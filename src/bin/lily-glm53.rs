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
    /// Teacher-forced scoring of ds4's official GLM 5.3 Flash continuations:
    /// mean NLL per token and greedy agreement, per case and overall.
    Score {
        #[arg(long)]
        model: PathBuf,
        /// A fixture directory (manifest.tsv, prompts/, continuations/,
        /// responses/); paths in the manifest are taken by file name.
        #[arg(long)]
        fixture: PathBuf,
        /// Reasoning effort the fixture was collected with.
        #[arg(long, default_value = "low")]
        effort: String,
        /// Score only the first N cases.
        #[arg(long)]
        limit: Option<usize>,
        /// ds4's legacy no-thinking prompt (`[gMASK]<sop><|user|>...
        /// <|assistant|><think></think>`, no effort line, no reasoning): the
        /// rendering ds4's 100-case references were scored with.
        #[arg(long)]
        legacy: bool,
    },
    /// Chat: renders one user message with the GGUF's template and decodes
    /// greedily, printing the text.
    Chat {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        n: usize,
        /// Open a reasoning block (the template's default).
        #[arg(long)]
        think: bool,
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
        Cmd::Score { model, fixture, effort, limit, legacy } => {
            score(&model, &fixture, &effort, limit, legacy)
        }
        Cmd::Chat { model, prompt, n, think } => chat(&model, &prompt, n, think),
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
        last_logits = sc.logits.to_f32()?[..cfg.vocab].to_vec();
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
    // Profile only the decode steps.
    let _ = lily::metal::profile::take();
    let t = Instant::now();
    let vocab = model.cfg.vocab;
    let mut next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
    for _ in 0..n {
        out.push(next);
        model.forward(&ctx, &mut st, &mut sc, &[next], true)?;
        next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
    }
    let dt = t.elapsed().as_secs_f64();
    if ctx.profiling() {
        profile_report(n);
    }
    println!("{}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
    eprintln!("decode {n} tokens in {dt:.2} s ({:.2} tok/s)", n as f64 / dt);
    Ok(())
}

fn log_softmax_at(logits: &[f32], target: usize) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| f64::from(l - m).exp()).sum();
    f64::from(logits[target] - m) - sum.ln()
}

fn score(
    path: &PathBuf,
    fixture: &PathBuf,
    effort: &str,
    limit: Option<usize>,
    legacy: bool,
) -> Result<()> {
    use lily::tokenizer::Tokenizer;
    let tok = Tokenizer::from_gguf(path)?;
    let ctx = MetalContext::new()?;
    let model = Glm53Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let manifest = std::fs::read_to_string(fixture.join("manifest.tsv"))?;
    let mut sc = model.new_scratch(&ctx)?;
    let (mut tot_nll, mut tot_n, mut tot_agree) = (0f64, 0usize, 0usize);
    let mut token_count_matches = 0;
    let (mut first_matches, mut total_lcp) = (0usize, 0usize);
    let mut case_nll_sum = 0f64;
    let mut cases = 0;
    let effort_cap = {
        let mut c = effort.chars();
        c.next()
            .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
            .unwrap_or_default()
    };
    println!("case       prompt(api) cont   nll/tok  agree   first lcp  prefill_s");
    for line in manifest.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        if limit.is_some_and(|n| cases >= n) {
            break;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let case = cols[0];
        let file = |dir: &str, col: &str| {
            fixture.join(dir).join(std::path::Path::new(col).file_name().unwrap())
        };
        let prompt = std::fs::read_to_string(file("prompts", cols[1]))?;
        let cont = std::fs::read_to_string(file("continuations", cols[2]))?;
        let resp: serde_json::Value =
            serde_json::from_slice(&std::fs::read(file("responses", cols[3]))?)?;
        let msg = &resp["choices"][0]["message"];
        let reasoning = msg["reasoning"]
            .as_str()
            .or(msg["reasoning_content"].as_str())
            .unwrap_or("");
        let api_prompt = resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize;
        // ds4's render_glm_references.py: the official template with the
        // API's reasoning in the prefix.
        let rendered = if legacy {
            format!("[gMASK]<sop><|user|>{prompt}<|assistant|><think></think>")
        } else {
            format!(
                "[gMASK]<sop><|system|>Reasoning Effort: {effort_cap}<|user|>{prompt}<|assistant|><think>{reasoning}</think>"
            )
        };
        let p_ids = tok.encode(&rendered)?;
        let c_ids = tok.encode(&cont)?;
        let mut st = model.new_state(&ctx, p_ids.len() + c_ids.len() + 8)?;
        let t = Instant::now();
        // All but the prompt's last token, then teacher-force from there.
        model.prefill(&ctx, &mut st, &mut sc, &p_ids[..p_ids.len() - 1])?;
        let pf = t.elapsed().as_secs_f64();
        let feed: Vec<u32> = std::iter::once(*p_ids.last().unwrap())
            .chain(c_ids.iter().copied())
            .collect();
        let (mut nll, mut agree) = (0f64, 0usize);
        let (mut lcp, mut prefix_open) = (0usize, true);
        let mut first = false;
        for (ci, chunk) in
            feed[..c_ids.len()].chunks(lily::glm53::model::MAX_ROWS).enumerate()
        {
            model.forward_all_logits(&ctx, &mut st, &mut sc, chunk)?;
            let logits = sc.logits.to_f32()?;
            for r in 0..chunk.len() {
                let row = &logits[r * vocab..(r + 1) * vocab];
                let target = c_ids[ci * lily::glm53::model::MAX_ROWS + r] as usize;
                nll -= log_softmax_at(row, target);
                let hit = argmax(row) == target;
                agree += usize::from(hit);
                if ci == 0 && r == 0 {
                    first = hit;
                }
                prefix_open &= hit;
                lcp += usize::from(prefix_open);
            }
        }
        // The reasoning-free prompt of a case without reasoning has the API's
        // token count; with reasoning the API counts it as completion.
        let reasoning_tokens = if legacy || reasoning.is_empty() {
            0
        } else {
            tok.encode(reasoning)?.len() + 2
        };
        token_count_matches +=
            usize::from(p_ids.len() - reasoning_tokens == api_prompt);
        println!(
            "{case}  {:5}({:5}) {:4}  {:7.4}  {:5.1}%  {:5} {:3}  {:6.1}",
            p_ids.len(),
            api_prompt,
            c_ids.len(),
            nll / c_ids.len() as f64,
            100.0 * agree as f64 / c_ids.len() as f64,
            first,
            lcp,
            pf
        );
        tot_nll += nll;
        tot_n += c_ids.len();
        tot_agree += agree;
        first_matches += usize::from(first);
        total_lcp += lcp;
        case_nll_sum += nll / c_ids.len() as f64;
        cases += 1;
    }
    println!(
        "total: {cases} cases, {tot_n} tokens, avg NLL {:.6}/token (case mean {:.6}), first-token match {first_matches}/{cases}, avg greedy prefix {:.2}, greedy agreement {:.2}%, prompt token count = API's in {token_count_matches}/{cases}",
        tot_nll / tot_n as f64,
        case_nll_sum / cases as f64,
        total_lcp as f64 / cases as f64,
        100.0 * tot_agree as f64 / tot_n as f64
    );
    Ok(())
}

fn chat(path: &PathBuf, prompt: &str, n: usize, think: bool) -> Result<()> {
    use lily::tokenizer::{ChatRender, Tokenizer};
    let tok = Tokenizer::from_gguf(path)?;
    let messages = [serde_json::json!({"role": "user", "content": prompt})];
    let render = ChatRender {
        messages: &messages,
        tools: None,
        enable_thinking: think,
        reasoning_effort: None,
        preserve_thinking: None,
    };
    eprintln!("{:?}", tok.render(&render)?);
    let ids = tok.encode_chat(&render)?;
    let ctx = MetalContext::new()?;
    let model = Glm53Model::load(&ctx, path)?;
    let vocab = model.cfg.vocab;
    let mut st = model.new_state(&ctx, ids.len() + n + 8)?;
    let mut sc = model.new_scratch(&ctx)?;
    let t = Instant::now();
    model.prefill(&ctx, &mut st, &mut sc, &ids)?;
    let pf = t.elapsed().as_secs_f64();
    let mut out = Vec::new();
    let t = Instant::now();
    let mut next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
    while out.len() < n && !tok.stop_tokens().contains(&next) {
        out.push(next);
        model.forward(&ctx, &mut st, &mut sc, &[next], true)?;
        next = argmax(&sc.logits.to_f32()?[..vocab]) as u32;
    }
    let dt = t.elapsed().as_secs_f64();
    println!("{}", tok.decode(&out, false)?);
    eprintln!(
        "prompt {} tokens, prefill {:.2} s ({:.1} tok/s); decode {} tokens in {:.2} s ({:.2} tok/s); stop {next}",
        ids.len(),
        pf,
        ids.len() as f64 / pf,
        out.len(),
        dt,
        out.len() as f64 / dt
    );
    Ok(())
}

/// With `LILY_KERNEL_PROFILE=1`: GPU time per kernel name per decode step
/// (each dispatch timed alone, so the sum exceeds a real step).
fn profile_report(steps: usize) {
    let passes = lily::metal::profile::take();
    let mut by: std::collections::HashMap<&str, (usize, f64)> = Default::default();
    let mut total = 0.0;
    for p in &passes {
        for k in &p.kernels {
            let e = by.entry(k.name).or_default();
            e.0 += 1;
            e.1 += k.gpu_secs;
            total += k.gpu_secs;
        }
    }
    let mut rows: Vec<_> = by.into_iter().collect();
    rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    let n = steps as f64;
    eprintln!("  ms/step  share  calls/step  kernel");
    for (name, (calls, secs)) in rows {
        eprintln!(
            "  {:7.3}  {:4.1}%  {:9.1}  {name}",
            secs / n * 1e3,
            100.0 * secs / total,
            calls as f64 / n
        );
    }
    eprintln!("  {:7.3}  kernels summed per step", total / n * 1e3);
}
