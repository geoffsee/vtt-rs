//! Command-line interface for the vtt-rs transcription service.
//!
//! This binary provides a standalone CLI tool for real-time audio transcription.
//! See the library documentation for programmatic usage.

use anyhow::{Context, Result};
use std::path::PathBuf;
use tokio::signal;
use vtt_rs::{Config, TranscriptionEvent, TranscriptionService};

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .context("Set OPENAI_API_KEY to call OpenAI transcription")?;

    let config_arg = std::env::args().nth(1);
    let (config, config_path) = load_config(config_arg)?;

    if let Some(path) = config_path.as_deref() {
        println!("Loaded configuration from {}", path.display());
    } else {
        println!("Using default configuration");
    }

    let resolved_out_path = config.resolve_out_path(config_path.as_deref());
    if let Some(path) = resolved_out_path.as_ref() {
        println!("Saving transcripts to {}", path.display());
    }

    println!("Starting transcription service...");
    println!("Press Ctrl+C to stop.");

    let mut service = TranscriptionService::new(config, api_key)?;
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
