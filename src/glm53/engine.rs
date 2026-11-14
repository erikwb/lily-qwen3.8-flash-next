//! GLM 5.3 Flash behind the engine's [`LanguageModel`] interface: what the
//! server and the bench drive. One session at a time, no draft head yet (the
//! MTP block is phase 5), no batching across sessions.

use std::path::Path;

use anyhow::{Result, ensure};

use super::model::{Glm53Model, Glm53State, LayerState, Logits, MAX_ROWS, Scratch};
use crate::engine::{
    DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, SnapshotApi,
};
use crate::kernels::sample::sample_f32;
use crate::metal::{EncodedPass, MetalContext};
use crate::tensor::{DType, Tensor};

/// Copies `src`'s first `bytes` into `dst` through host memory (both
/// shared-storage, the GPU idle on them).
fn host_copy(src: &Tensor, dst: &Tensor, bytes: usize) {
    assert!(bytes <= src.byte_len() && bytes <= dst.byte_len());
    // SAFETY: both buffers are host-visible and hold at least `bytes`; the
    // GPU is idle on them by the callers' contracts.
    unsafe {
        core::ptr::copy_nonoverlapping(src.contents_ptr(), dst.contents_ptr(), bytes);
    }
}

fn clone_tensor(ctx: &MetalContext, t: &Tensor) -> Result<Tensor> {
    let c = Tensor::zeros(ctx, t.shape(), t.dtype())?;
    host_copy(t, &c, t.byte_len());
    Ok(c)
}

/// The recurrent part of a state: KDA conv windows and states, the
/// indexer's incomplete-pool rows.
pub struct Glm53Snapshot {
    pos: usize,
    tensors: Vec<Tensor>,
}

impl SnapshotApi for Glm53Snapshot {
    fn pos(&self) -> usize {
        self.pos
    }

    fn bytes(&self) -> usize {
        self.tensors.iter().map(Tensor::byte_len).sum()
    }
}

/// The tensors a snapshot holds, in a fixed order.
fn recurrent(st: &Glm53State) -> Vec<&Tensor> {
    st.layers
        .iter()
        .flat_map(|l| match l {
            LayerState::Kda { conv, s } => vec![conv, s],
            LayerState::Dsa { tail_k, tail_g, .. } => vec![tail_k, tail_g],
        })
        .collect()
}

impl DecodeStateApi for Glm53State {
    type Snapshot = Glm53Snapshot;

    fn pos(&self) -> usize {
        self.pos
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn reset(&mut self) -> Result<()> {
        Glm53State::reset(self);
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure_capacity(&mut self, ctx: &MetalContext, tokens: usize) -> Result<()> {
        if tokens <= self.capacity {
            return Ok(());
        }
        let capacity = tokens.max(self.capacity * 2).next_multiple_of(4);
        for l in &mut self.layers {
            if let LayerState::Dsa { latent, pooled, .. } = l {
                let per_tok = latent.numel() / self.capacity;
                let per_pool = pooled.numel() / (self.capacity / 4);
                let nl = Tensor::zeros(ctx, &[capacity * per_tok], DType::F16)?;
                let np = Tensor::zeros(ctx, &[capacity / 4 * per_pool], DType::F16)?;
                host_copy(latent, &nl, self.pos * per_tok * 2);
                host_copy(pooled, &np, self.pos / 4 * per_pool * 2);
                *latent = nl;
                *pooled = np;
            }
        }
        self.pool_cap = capacity / 4 + 1;
        self.pool_scores = Tensor::zeros(ctx, &[MAX_ROWS * self.pool_cap], DType::F32)?;
        self.capacity = capacity;
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerState::Kda { conv, s } => conv.byte_len() + s.byte_len(),
                LayerState::Dsa { latent, pooled, tail_k, tail_g } => {
                    latent.byte_len()
                        + pooled.byte_len()
                        + tail_k.byte_len()
                        + tail_g.byte_len()
                }
            })
            .sum::<usize>()
            + self.pool_scores.byte_len()
    }

    fn snapshot(&self, ctx: &MetalContext) -> Result<Glm53Snapshot> {
        let tensors = recurrent(self)
            .into_iter()
            .map(|t| clone_tensor(ctx, t))
            .collect::<Result<_>>()?;
        Ok(Glm53Snapshot { pos: self.pos, tensors })
    }

    fn restore(&mut self, _ctx: &MetalContext, snapshot: &Glm53Snapshot) -> Result<()> {
        ensure!(snapshot.pos <= self.capacity, "snapshot past the state's capacity");
        for (dst, src) in recurrent(self).into_iter().zip(&snapshot.tensors) {
            host_copy(src, dst, src.byte_len());
        }
        self.pos = snapshot.pos;
        Ok(())
    }

    fn copy_prefix_from(
        &mut self,
        ctx: &MetalContext,
        from: &Self,
        tokens: usize,
    ) -> Result<()> {
        ensure!(tokens <= from.pos, "prefix of {tokens} from a state at {}", from.pos);
        self.ensure_capacity(ctx, tokens)?;
        for (dst, src) in self.layers.iter().zip(&from.layers) {
            if let (
                LayerState::Dsa { latent: dl, pooled: dp, .. },
                LayerState::Dsa { latent: sl, pooled: sp, .. },
            ) = (dst, src)
            {
                let per_tok = sl.numel() / from.capacity;
                let per_pool = sp.numel() / (from.capacity / 4);
                host_copy(sl, dl, tokens * per_tok * 2);
                host_copy(sp, dp, tokens / 4 * per_pool * 2);
            }
        }
        Ok(())
    }
}

impl ScratchApi for Scratch {
    fn next_token(&self) -> &Tensor {
        &self.next_token
    }

    fn logits(&self) -> &Tensor {
        &self.logits_last
    }

    fn begin_request(&self) {
        self.sampler().reset_counts();
    }
}

impl LanguageModel for Glm53Model {
    type State = Glm53State;
    type Scratch = Scratch;

    const MODEL_ID: &'static str = "glm-5.3-flash";

    fn load(ctx: &MetalContext, path: &Path, options: &LoadOptions) -> Result<Self> {
        if options.mtp_drafts > 0 {
            eprintln!(
                "glm-5.3-flash: the MTP draft head is not wired up yet; decoding plainly"
            );
        }
        Glm53Model::load(ctx, path)
    }

    fn max_position_embeddings(&self) -> usize {
        self.cfg.context_length
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        vec![self.cfg.eos, self.cfg.eot, self.cfg.eom]
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }

    fn bytes_per_token(&self) -> usize {
        let dsa =
            self.cfg.kinds.iter().filter(|k| **k == super::LayerKind::Dsa).count();
        // f16 latent rows, and the trunk's f16 pooled keys per four tokens.
        dsa * self.cfg.kv_lora * 2 + (dsa - 1) * self.cfg.indexer_dim * 2 / 4
    }

    fn weight_buffers(&self) -> Vec<crate::metal::Buffer> {
        Glm53Model::weight_buffers(self)
    }

    fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<Glm53State> {
        Glm53Model::new_state(self, ctx, capacity.max(4))
    }

    fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        _capacity_tokens: usize,
    ) -> Result<Scratch> {
        self.new_scratch(ctx)
    }

    fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut Glm53State,
        scratch: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
    ) -> Result<()> {
        ensure!(!tokens.is_empty(), "empty prefill");
        state.ensure_capacity(ctx, state.pos + tokens.len())?;
        let chunks: Vec<&[u32]> = tokens.chunks(MAX_ROWS).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            let last = i + 1 == chunks.len();
            let mut words = [0u32; MAX_ROWS];
            words[..chunk.len()].copy_from_slice(chunk);
            scratch.tokens.write_bytes(bytemuck::cast_slice(&words))?;
            let pass = self.begin(ctx)?;
            let logits =
                if last && draw.is_some() { Logits::Last } else { Logits::None };
            self.encode(
                ctx,
                &pass,
                state,
                scratch,
                &scratch.tokens,
                chunk.len(),
                logits,
            )?;
            if let (true, Some(d)) = (last, draw) {
                let out = scratch.next_token.view(0, &[1])?;
                sample_f32(
                    ctx,
                    &pass,
                    &scratch.logits_last,
                    scratch.sampler(),
                    d.params,
                    d.step,
                    &out,
                )?;
            }
            pass.commit_wait()?;
            state.pos += chunk.len();
        }
        Ok(())
    }

    fn prepare_step_inputs(
        &self,
        _state: &mut Glm53State,
        _scratch: &Scratch,
        _token: u32,
    ) -> Result<()> {
        Ok(())
    }

    fn encode_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &Glm53State,
        scratch: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>> {
        ensure!(state.pos < state.capacity, "decode step past the state's capacity");
        let pass = self.begin(ctx)?;
        let token = scratch.next_token.view(slot_in, &[1])?;
        self.encode(ctx, &pass, state, scratch, &token, 1, Logits::Last)?;
        let out = scratch.next_token.view(slot_out, &[1])?;
        sample_f32(
            ctx,
            &pass,
            &scratch.logits_last,
            scratch.sampler(),
            draw.params,
            draw.step,
            &out,
        )?;
        pass.end()
    }
}
