//! GLM 5.3 Flash behind the engine's [`LanguageModel`] interface: what the
//! server and the bench drive. One session at a time, no draft head yet (the
//! MTP block is phase 5), no batching across sessions.

use std::path::Path;

use anyhow::{Result, ensure};

use super::model::{Glm53Model, Glm53State, LayerState, Logits, MAX_ROWS, Scratch};
use crate::engine::{
    DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, SnapshotApi,
};
use crate::kernels::sample::{SamplingParams, sample_f32};
use crate::metal::{EncodedPass, MetalContext, PendingPass};
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
            LayerState::Kda { conv, s, .. } => vec![conv, s],
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
                LayerState::Kda { conv, s, conv_alt, s_alt } => {
                    conv.byte_len()
                        + s.byte_len()
                        + conv_alt.byte_len()
                        + s_alt.byte_len()
                }
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

impl Glm53Model {
    /// Runs one MTP step at main position `p` for the token at `p + 1`,
    /// reading hidden row `row` of the scratch; returns the draft when
    /// asked. Waits for the GPU.
    #[allow(clippy::too_many_arguments)]
    fn mtp_step(
        &self,
        ctx: &MetalContext,
        state: &mut Glm53State,
        scratch: &Scratch,
        token: u32,
        row: usize,
        p: usize,
        draft: bool,
    ) -> Result<u32> {
        scratch.mtp_io.write_bytes(bytemuck::cast_slice(&[token, 0u32]))?;
        let hidden = scratch.hidden.view(row * self.cfg.hidden, &[self.cfg.hidden])?;
        let pass = self.begin(ctx)?;
        self.encode_mtp(ctx, &pass, state, scratch, &hidden, p, draft)?;
        pass.commit_wait()?;
        state.mtp_next = Some(p + 1);
        Ok(scratch.mtp_io.to_u32()?[1])
    }
}

impl LanguageModel for Glm53Model {
    type State = Glm53State;
    type Scratch = Scratch;

    const MODEL_ID: &'static str = "glm-5.3-flash";

    fn load(ctx: &MetalContext, path: &Path, options: &LoadOptions) -> Result<Self> {
        let mut model = Glm53Model::load(ctx, path)?;
        model.mtp = options.mtp_drafts > 0;
        Ok(model)
    }

    /// One draft a step: the MTP block predicts one token ahead (ds4 drafts
    /// one for GLM 5.3 too).
    fn max_drafts(&self) -> usize {
        usize::from(self.mtp)
    }

    fn draft_initial(
        &self,
        ctx: &MetalContext,
        state: &mut Glm53State,
        scratch: &mut Scratch,
        first: u32,
        _drafts: usize,
        _params: &SamplingParams,
        _step0: usize,
    ) -> Result<Vec<u32>> {
        ensure!(state.pos > 0, "drafting before any token was fed");
        // The last fed token's position; its stream mean is the scratch's
        // hidden row 0.
        let p = state.pos - 1;
        if state.mtp_next != Some(p) {
            state.mtp_min = Some(p);
        }
        let draft = self.mtp_step(ctx, state, scratch, first, 0, p, true)?;
        Ok(vec![draft])
    }

    fn verify<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut Glm53State,
        scratch: &mut Scratch,
        pending: u32,
        drafts: &[u32],
        params: &SamplingParams,
        step0: usize,
        _next_drafts: usize,
    ) -> Result<(Vec<u32>, PendingPass<'a>)> {
        let rows = 1 + drafts.len();
        ensure!(rows <= 2, "GLM verifies one draft a step, got {}", drafts.len());
        ensure!(state.spec.is_none(), "a verify pass is already pending");
        state.ensure_capacity(ctx, state.pos + rows)?;
        let mut words = [0u32; MAX_ROWS];
        words[0] = pending;
        words[1..rows].copy_from_slice(drafts);
        scratch.spec_tokens.write_bytes(bytemuck::cast_slice(&words))?;
        let pass = self.begin(ctx)?;
        // KDA keeps the state after row 0 aside: the rollback if row 1's
        // draft is rejected.
        let snap = (rows > 1).then_some(0);
        self.encode(
            ctx,
            &pass,
            state,
            scratch,
            &scratch.spec_tokens,
            rows,
            Logits::All,
            snap,
        )?;
        let vocab = self.cfg.vocab;
        for r in 0..rows {
            let logits = scratch.logits.view(r * vocab, &[vocab])?;
            let out = scratch.draws.view(r, &[1])?;
            sample_f32(
                ctx,
                &pass,
                &logits,
                scratch.sampler(),
                params,
                step0 + r,
                &out,
            )?;
            pass.memory_barrier()?;
        }
        pass.commit_wait()?;
        let draws = scratch.draws.to_u32()?[..rows].to_vec();
        state.spec = Some((state.pos, rows, draws.clone()));
        state.pos += rows;
        // Nothing runs behind the verify pass: the draft is proposed in
        // finish_speculation, once the kept count is known.
        Ok((draws, ctx.begin()?.commit()?))
    }

    fn finish_speculation(
        &self,
        ctx: &MetalContext,
        state: &mut Glm53State,
        scratch: &mut Scratch,
        accepted: usize,
        propose: bool,
        draft: PendingPass<'_>,
    ) -> Result<Vec<u32>> {
        draft.wait()?;
        let (pos0, rows, draws) = state
            .spec
            .take()
            .ok_or_else(|| anyhow::anyhow!("no verify pass pending"))?;
        ensure!(accepted < rows, "{accepted} accepted of {} drafts", rows - 1);
        state.pos = pos0 + 1 + accepted;
        if accepted + 1 < rows {
            // Back to the state after row 0.
            state.take_kda_snapshots();
        }
        for &token in &draws[accepted + 1..] {
            scratch.sampler().uncount(token);
        }
        let mut proposals = Vec::new();
        if propose {
            if state.mtp_next != Some(pos0) {
                state.mtp_min = Some(pos0);
            }
            // Catch the head up on the accepted rows, then draft from the
            // last one.
            for r in 0..accepted {
                self.mtp_step(ctx, state, scratch, draws[r], r, pos0 + r, false)?;
            }
            proposals.push(self.mtp_step(
                ctx,
                state,
                scratch,
                draws[accepted],
                accepted,
                pos0 + accepted,
                true,
            )?);
        } else {
            state.mtp_next = None;
        }
        // The last fed row's stream mean becomes hidden row 0 (what the next
        // draft reads).
        let hidden = self.cfg.hidden;
        if accepted > 0 {
            let row = scratch.hidden.view(accepted * hidden, &[hidden])?;
            let first = scratch.hidden.view(0, &[hidden])?;
            first.write_bytes(row.contents())?;
        }
        Ok(proposals)
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
                None,
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
        self.encode(ctx, &pass, state, scratch, &token, 1, Logits::Last, None)?;
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
