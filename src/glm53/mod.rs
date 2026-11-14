//! GLM 5.3 Flash (`glm5-next` GGUF): configuration, the CPU reference, and
//! the model graph. Logic and kernels follow antirez's ds4 (MIT), which runs
//! this model on Metal, CUDA and ROCm (`ds4.c`, `metal/glm53_*.metal`); lily
//! loads ds4's published GGUF as it is.

pub mod config;
pub mod engine;
pub mod model;
pub mod reference;
pub mod schema;
pub mod tokenizer;

pub use config::{Config, LayerKind};
pub use model::{Glm53Model, Glm53State, LayerTrace, Logits, Scratch};
