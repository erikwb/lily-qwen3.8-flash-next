//! GLM 5.3 Flash bring-up tool: compares the GPU forward pass with the CPU
//! reference layer by layer on real weights, and runs greedy generation.
//!
//! ```sh
//! lily-glm53 compare --model GLM-5.3-Flash-Q2.gguf --tokens 154822,9703,11 --layers 4
//! lily-glm53 generate --model GLM-5.3-Flash-Q2.gguf --tokens ... --n 32
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use lily::glm53::reference::{RefModel, Trace};
use lily::glm53::{Glm53Model, LayerTrace};
use lily::metal::MetalContext;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// GPU against the CPU reference, layer by layer.
    Compare {
        #[arg(long)]
        model: PathBuf,
        /// Comma-separated token ids.
        #[arg(long)]
        tokens: String,
        /// Trunk layers to compare (the CPU reference is slow).
        #[arg(long, default_value_t = 4)]
        layers: usize,
        /// Feed the tokens one at a time instead of in runs of four.
        #[arg(long)]
        single: bool,
    },
    /// Greedy generation from token ids.
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokens: String,
        #[arg(long, default_value_t = 16)]
        n: usize,
    },
}

fn parse_tokens(s: &str) -> Result<Vec<u32>> {
    s.split(',').map(|t| t.trim().parse::<u32>().context("token id")).collect()
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Compare { model, tokens, layers, single } => {
            compare(&model, &parse_tokens(&tokens)?, layers, single)
        }
        Cmd::Generate { model, tokens, n } => {
            generate(&model, &parse_tokens(&tokens)?, n)
        }
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
}

/// Relative L2 error and max abs error of `a` against `b`.
fn err(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut d2 = 0f64;
    let mut n2 = 0f64;
    let mut mx = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = f64::from(x) - f64::from(y);
        d2 += d * d;
        n2 += f64::from(y) * f64::from(y);
        mx = mx.max(d.abs());
    }
    ((d2 / n2.max(1e-30)).sqrt(), mx)
}

fn compare(path: &PathBuf, tokens: &[u32], layers: usize, single: bool) -> Result<()> {
    let ctx = MetalContext::new()?;
    let t0 = Instant::now();
    let model = Glm53Model::load(&ctx, path)?;
    eprintln!("mapped in {:.1} s", t0.elapsed().as_secs_f64());
    let cfg = model.cfg.clone();
    let mut st = model.new_state(&ctx, tokens.len() + 8)?;
    let mut sc = model.new_scratch(&ctx)?;
    // GPU: per layer, the rows of every step concatenated in token order.
    let mut gpu: Vec<LayerTrace> = vec![LayerTrace::default(); cfg.n_trunk];
    let step = if single { 1 } else { lily::glm53::model::MAX_ROWS };
    let mut last_logits = Vec::new();
    for chunk in tokens.chunks(step) {
        let mut tr = Vec::new();
        model.forward_traced(&ctx, &mut st, &mut sc, chunk, &mut tr)?;
        for (g, t) in gpu.iter_mut().zip(tr) {
            g.attn_in.extend(t.attn_in);
            g.attn_out.extend(t.attn_out);
            g.ffn_in.extend(t.ffn_in);
            g.ffn_out.extend(t.ffn_out);
            g.h.extend(t.h);
        }
        last_logits = sc.logits.to_f32()?;
    }
    eprintln!("GPU traced forward done; top token {}", argmax(&last_logits));
    // CPU, layer by layer from the GPU's own input to each layer, so every
    // layer is compared alone.
    let rm = RefModel::new(cfg.clone(), &model.gguf);
    let hd = cfg.hc_dim();
    let mut hs = rm.embed(tokens)?;
    println!(
        "layer  kind   attn_in        attn_out       ffn_in         ffn_out        h (rel L2 / max abs)"
    );
    for il in 0..layers.min(cfg.n_trunk) {
        let t = Instant::now();
        let traces: Vec<Trace> = rm.trunk_layer(il, &mut hs)?;
        let g = &gpu[il];
        let cat = |f: fn(&Trace) -> &Vec<f32>| -> Vec<f32> {
            traces.iter().flat_map(|t| f(t).clone()).collect()
        };
        let h_cpu: Vec<f32> = hs.iter().flatten().copied().collect();
        let cols = [
            err(&g.attn_in, &cat(|t| &t.attn_in)),
            err(&g.attn_out, &cat(|t| &t.attn_out)),
            err(&g.ffn_in, &cat(|t| &t.ffn_in)),
            err(&g.ffn_out, &cat(|t| &t.ffn_out)),
            err(&g.h, &h_cpu),
        ];
        print!("{il:5}  {:5?}", cfg.kinds[il]);
        for (r, m) in cols {
            print!("  {r:.2e}/{m:.1e}");
        }
        println!("   ({:.1} s)", t.elapsed().as_secs_f64());
        let sel: Vec<String> =
            traces.iter().map(|t| format!("{:?}", t.selected)).collect();
        if !traces[0].selected.is_empty() {
            println!("       experts {}", sel.join(" "));
        }
        // Continue from the GPU's state so errors do not accumulate.
        for (i, h) in hs.iter_mut().enumerate() {
            h.copy_from_slice(&g.h[i * hd..(i + 1) * hd]);
        }
    }
    Ok(())
}

fn generate(path: &PathBuf, prompt: &[u32], n: usize) -> Result<()> {
    let ctx = MetalContext::new()?;
    let t0 = Instant::now();
    let model = Glm53Model::load(&ctx, path)?;
    eprintln!("mapped in {:.1} s", t0.elapsed().as_secs_f64());
    let mut st = model.new_state(&ctx, prompt.len() + n + 8)?;
    let mut sc = model.new_scratch(&ctx)?;
    let t = Instant::now();
    model.prefill(&ctx, &mut st, &mut sc, prompt)?;
    let pf = t.elapsed().as_secs_f64();
    eprintln!(
        "prefill {} tokens in {:.2} s ({:.1} tok/s)",
        prompt.len(),
        pf,
        prompt.len() as f64 / pf
    );
    let mut out = Vec::new();
    let t = Instant::now();
    let mut next = argmax(&sc.logits.to_f32()?) as u32;
    for _ in 0..n {
        out.push(next);
        model.forward(&ctx, &mut st, &mut sc, &[next], true)?;
        next = argmax(&sc.logits.to_f32()?) as u32;
    }
    let dt = t.elapsed().as_secs_f64();
    println!("{}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
    eprintln!("decode {n} tokens in {dt:.2} s ({:.2} tok/s)", n as f64 / dt);
    Ok(())
}
