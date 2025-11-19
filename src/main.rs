//! Command-line interface for the vtt-rs transcription service.
//!
//! This binary provides a standalone CLI tool for real-time audio transcription.
//! See the library documentation for programmatic usage.

use anyhow::Result;
use std::path::PathBuf;
use tokio::signal;
use vtt_rs::{Config, TranscriptionEvent, TranscriptionService};

#[tokio::main]
async fn main() -> Result<()> {
    let config_arg = std::env::args().nth(1);
    let (config, config_path) = load_config(config_arg)?;

    if let Some(path) = config_path.as_deref() {
        println!("Loaded configuration from {}", path.display());
    } else {
        println!("Using default configuration");
    }

    let resolved_out_path = config.resolve_out_path(config_path.as_deref());
    let on_device = config.uses_on_device();
    if let Some(path) = resolved_out_path.as_ref() {
        println!("Saving transcripts to {}", path.display());
    }

    if on_device {
        let model_name = config
            .on_device
            .as_ref()
            .map(|c| c.model.clone())
            .unwrap_or_else(|| "tiny.en".to_string());
        println!("Using on-device Whisper backend ({model_name}).");
    }

    println!("Starting transcription service...");
    println!("Press Ctrl+C to stop.");

    let mut service = if on_device {
        TranscriptionService::new_on_device(config)?
    } else {
        match std::env::var("OPENAI_API_KEY") {
            Ok(api_key) => {
                println!("Using remote transcription with API key (Authorization: Bearer ...)");
                TranscriptionService::new(config, api_key)?
            }
            Err(_) => {
                println!(
                    "No OPENAI_API_KEY found. Proceeding without Authorization header.\n\
                    If your endpoint requires a key, set OPENAI_API_KEY.\n\
                    This is suitable for local OpenAI-compatible servers (e.g. MLX Parakeet)."
                );
                TranscriptionService::new_no_api(config)?
            }
        }
    };
    let (mut receiver, _stream) = service.start().await?;

    let ctrl_c = signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        tokio::select! {
            _ = &mut ctrl_c => {
                println!("Stopping capture...");
                break;
            }
            Some(event) = receiver.recv() => {
                match event {
                    TranscriptionEvent::Transcription { chunk_id, text } => {
                        if text.is_empty() {
                            println!("Chunk {chunk_id}: <silence>");
                        } else {
                            println!("Chunk {chunk_id}: {text}");
                        }
                    }
                    TranscriptionEvent::Error { chunk_id, error } => {
                        eprintln!("Chunk {chunk_id} error: {error}");
                    }
                }
            }
        }
    }

    Ok(())
}

fn load_config(arg_path: Option<String>) -> Result<(Config, Option<PathBuf>)> {
    if let Some(path) = arg_path {
        let path = PathBuf::from(path);
        let config = Config::from_file(&path)?;
        return Ok((config, Some(path)));
    }

    Config::load_or_default()
}
