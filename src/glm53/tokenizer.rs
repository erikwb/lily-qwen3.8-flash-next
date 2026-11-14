//! GLM's tokenizer and chat template from the GGUF metadata: the
//! `tokenizers` JSON lily's [`crate::tokenizer::Tokenizer`] loads, built in
//! memory from `tokenizer.ggml.*` (byte-level BPE with the `glm4`
//! pre-tokenizer, the split rule ds4's `bpe_tokenize_text_glm4` implements),
//! and the template from `tokenizer.chat_template`.

use anyhow::{Context as _, Result, ensure};
use serde_json::{Value, json};

use crate::gguf::{Gguf, Value as GValue};

/// The `glm4` pre-tokenizer split (llama.cpp's `LLAMA_VOCAB_PRE_TYPE_GLM4`,
/// the regex of GLM-4's `tokenizer.json`).
const GLM4_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// GGUF token types.
const TYPE_CONTROL: u64 = 3;
const TYPE_USER_DEFINED: u64 = 4;

fn strings(g: &Gguf, key: &str) -> Result<Vec<String>> {
    match g.get(key) {
        Some(GValue::Array(v)) => v
            .iter()
            .map(|s| s.as_str().map(str::to_string).context("non-string entry"))
            .collect::<Result<_>>()
            .with_context(|| format!("reading {key}")),
        _ => anyhow::bail!("GGUF metadata {key} missing"),
    }
}

/// The `tokenizer.json` of the file's vocabulary.
pub fn tokenizer_json(g: &Gguf) -> Result<String> {
    ensure!(
        g.str("tokenizer.ggml.model")? == "gpt2",
        "GLM tokenizer model is not gpt2 BPE"
    );
    ensure!(g.str("tokenizer.ggml.pre")? == "glm4", "GLM pre-tokenizer is not glm4");
    let tokens = strings(g, "tokenizer.ggml.tokens")?;
    let vocab = g.u64(&format!("{}.vocab_size", super::config::ARCH))? as usize;
    ensure!(
        tokens.len() == vocab,
        "the tokenizer has {} tokens, the model's vocabulary {vocab}",
        tokens.len()
    );
    let merges = strings(g, "tokenizer.ggml.merges")?;
    let types: Vec<u64> = match g.get("tokenizer.ggml.token_type") {
        Some(GValue::Array(v)) => v.iter().map(|t| t.as_u64().unwrap_or(1)).collect(),
        _ => vec![1; tokens.len()],
    };
    let mut vocab = serde_json::Map::with_capacity(tokens.len());
    let mut added = Vec::new();
    for (id, tok) in tokens.iter().enumerate() {
        vocab.insert(tok.clone(), json!(id));
        let ty = types.get(id).copied().unwrap_or(1);
        if ty == TYPE_CONTROL || ty == TYPE_USER_DEFINED {
            added.push(json!({
                "id": id,
                "content": tok,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": ty == TYPE_CONTROL,
            }));
        }
    }
    let doc = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": GLM4_SPLIT}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
            ],
        },
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": Value::Object(vocab),
            "merges": merges,
        },
    });
    Ok(doc.to_string())
}

/// The file's chat template with an `enable_thinking` switch on the
/// generation prompt: the template always opens `<think>`; with thinking
/// off the prompt closes an empty block, `<think></think>`, which is how the
/// template itself renders an assistant turn without reasoning.
pub fn chat_template(g: &Gguf) -> Result<String> {
    let t = g.str("tokenizer.chat_template")?;
    let open = "<|assistant|>{{- '<think>' -}}";
    ensure!(t.contains(open), "GLM chat template lacks its generation prompt");
    Ok(t.replace(
        open,
        "<|assistant|>{{- '<think>' if enable_thinking is not defined or enable_thinking else '<think></think>' -}}",
    ))
}

/// Turn-ending ids: `<|endoftext|>`, `<|user|>` and `<|observation|>` (ds4's
/// eos, eot and eom).
pub fn stop_ids(g: &Gguf) -> Result<Vec<u32>> {
    let vocab = g.u64(&format!("{}.vocab_size", super::config::ARCH))?;
    ["eos", "eot", "eom"]
        .iter()
        .map(|k| {
            let id = g.u64(&format!("tokenizer.ggml.{k}_token_id"))?;
            ensure!(id < vocab, "{k} token {id} is outside the vocabulary of {vocab}");
            Ok(id as u32)
        })
        .collect()
}
