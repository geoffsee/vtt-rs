//! Voice-to-Text Streaming Library
//!
//! This library provides real-time audio transcription capabilities for AI agents
//! and other applications that need situational awareness through speech recognition.
//!
//! # Architecture
//!
//! The library is organized into several key components:
//!
//! - [`TranscriptionService`]: Main service that orchestrates audio capture and transcription
//! - [`Config`]: Configuration for chunk duration, API endpoint, and model selection
//! - [`TranscriptionEvent`]: Events emitted for successful transcriptions and errors
//!
//! Audio is captured from the default input device, chunked into configurable segments,
//! and sent to OpenAI-compatible transcription APIs. Results are delivered via an
//! event-driven async channel.
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```no_run
//! use vtt_rs::{TranscriptionService, Config, TranscriptionEvent};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = Config::default();
//!     let api_key = std::env::var("OPENAI_API_KEY")?;
//!
//!     let mut service = TranscriptionService::new(config, api_key)?;
//!
//!     // Start listening and transcribing
//!     let (mut receiver, _stream) = service.start().await?;
//!
//!     // Process transcription events
//!     while let Some(event) = receiver.recv().await {
//!         match event {
//!             TranscriptionEvent::Transcription { chunk_id, text } => {
//!                 println!("Heard: {}", text);
//!             }
//!             TranscriptionEvent::Error { chunk_id, error } => {
//!                 eprintln!("Error: {}", error);
//!             }
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Custom Configuration
//!
//! ```no_run
//! use vtt_rs::{Config, TranscriptionService};
//! use std::path::PathBuf;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let config = Config {
//!     chunk_duration_secs: 3,
//!     model: "whisper-1".to_string(),
//!     endpoint: "https://api.openai.com/v1/audio/transcriptions".to_string(),
//!     out_file: Some(PathBuf::from("transcripts.log")),
//! };
//!
//! let api_key = std::env::var("OPENAI_API_KEY")?;
//! let mut service = TranscriptionService::new(config, api_key)?;
//! # Ok(())
//! # }
//! ```

mod audio;
mod config;
mod transcription;

pub use config::{Config, OnDeviceConfig};
pub use transcription::{TranscriptionEvent, TranscriptionService};

// Re-export commonly used types
pub use anyhow::{Context, Result};
