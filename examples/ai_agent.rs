//! Example: Using vtt-rs to give an AI agent situational awareness.
//!
//! This example demonstrates how to integrate the vtt-rs library into an AI agent
//! to provide real-time audio transcription capabilities.
//!
//! # Running
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example ai_agent
//! ```

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use vtt_rs::{Config, TranscriptionEvent, TranscriptionService};

/// Simple AI agent with situational awareness through audio
struct AIAgent {
    /// Recent transcriptions for context
    context: Arc<RwLock<Vec<String>>>,
    /// Maximum context size
    max_context_items: usize,
}

impl AIAgent {
    fn new(max_context_items: usize) -> Self {
        Self {
            context: Arc::new(RwLock::new(Vec::new())),
            max_context_items,
        }
    }

    /// Process a transcription and update agent's awareness
    async fn process_transcription(&self, text: String) {
        if text.is_empty() {
            return; // Ignore silence
        }

        let mut context = self.context.write().await;
        context.push(text.clone());

        // Keep only the most recent items
        if context.len() > self.max_context_items {
            let drain_count = context.len() - self.max_context_items;
            context.drain(0..drain_count);
        }

        // Here you would integrate with your AI agent's decision-making
        println!("[Agent] Heard: {}", text);
        self.analyze_and_respond(&text).await;
    }

    /// Analyze the transcription and respond accordingly
    async fn analyze_and_respond(&self, text: &str) {
        let text_lower = text.to_lowercase();

        // Example: Detect commands or keywords
        if text_lower.contains("hey agent") || text_lower.contains("hello agent") {
            println!("[Agent] Acknowledged greeting!");
            // Here you could trigger a response action
        } else if text_lower.contains("help") {
            println!("[Agent] User needs assistance with: {}", text);
            // Here you could analyze the request and provide help
        } else if text_lower.contains("status") {
            let context = self.context.read().await;
            println!("[Agent] Current context has {} items", context.len());
        }

        // Example: Sentiment analysis
        if text_lower.contains("problem") || text_lower.contains("issue") {
            println!("[Agent] Detected potential problem, increasing alertness");
        }
    }

    /// Get the current context for decision-making
    async fn get_context(&self) -> Vec<String> {
        self.context.read().await.clone()
    }

    /// Get a summary of recent audio context
    async fn get_context_summary(&self) -> String {
        let context = self.get_context().await;
        if context.is_empty() {
            "No recent audio context".to_string()
        } else {
            format!(
                "Recent audio (last {} items):\n{}",
                context.len(),
                context.join("\n")
            )
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== AI Agent with Situational Awareness ===\n");

    // Get API key
    let api_key = std::env::var("OPENAI_API_KEY").expect("Set OPENAI_API_KEY environment variable");

    // Configure transcription
    let config = Config {
        chunk_duration_secs: 3, // Shorter chunks for faster response
        model: "whisper-1".to_string(),
        endpoint: "https://api.openai.com/v1/audio/transcriptions".to_string(),
        out_file: None, // Agent doesn't need file logging
    };

    // Create AI agent
    let agent = Arc::new(AIAgent::new(20)); // Keep last 20 transcriptions

    // Start transcription service
    let mut service = TranscriptionService::new(config, api_key)?;
    println!("Starting audio transcription...");
    let (mut receiver, _stream) = service.start().await?;
    println!("Listening for audio. Try saying 'Hey agent', 'help', or 'status'.");
    println!("Press Ctrl+C to stop.\n");

    // Process transcription events
    while let Some(event) = receiver.recv().await {
        match event {
            TranscriptionEvent::Transcription { chunk_id, text } => {
                if !text.is_empty() {
                    println!("[Chunk {}] Transcribed: {}", chunk_id, text);
                    agent.process_transcription(text).await;
                }
            }
            TranscriptionEvent::Error { chunk_id, error } => {
                eprintln!("[Error] Chunk {}: {}", chunk_id, error);
            }
        }
    }

    // Show final context summary
    println!("\n=== Final Context Summary ===");
    println!("{}", agent.get_context_summary().await);

    Ok(())
}
