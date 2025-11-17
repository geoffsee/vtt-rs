//! Example: Streaming AI agent with continuous awareness.
//!
//! This example shows a more sophisticated integration where the agent
//! maintains a sliding window of context and can respond to voice commands.
//!
//! # Running
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example streaming_agent
//! ```

use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tokio::time;
use vtt_rs::{Config, TranscriptionEvent, TranscriptionService};

#[derive(Clone)]
struct TimestampedTranscription {
    text: String,
    timestamp: SystemTime,
    #[allow(dead_code)]
    chunk_id: usize,
}

/// Advanced AI agent with temporal awareness
struct StreamingAgent {
    /// Transcriptions with timestamps
    history: Arc<RwLock<Vec<TimestampedTranscription>>>,
    /// Window duration for context
    context_window: Duration,
}

impl StreamingAgent {
    fn new(context_window_secs: u64) -> Self {
        Self {
            history: Arc::new(RwLock::new(Vec::new())),
            context_window: Duration::from_secs(context_window_secs),
        }
    }

    /// Add a new transcription to the history
    async fn add_transcription(&self, chunk_id: usize, text: String) {
        if text.is_empty() {
            return;
        }

        let transcription = TimestampedTranscription {
            text,
            timestamp: SystemTime::now(),
            chunk_id,
        };

        let mut history = self.history.write().await;
        history.push(transcription);

        // Clean old entries outside the context window
        self.cleanup_old_entries(&mut history).await;
    }

    /// Remove transcriptions older than the context window
    async fn cleanup_old_entries(&self, history: &mut Vec<TimestampedTranscription>) {
        let cutoff = SystemTime::now() - self.context_window;
        history.retain(|t| t.timestamp >= cutoff);
    }

    /// Get recent context within the time window
    async fn get_recent_context(&self) -> Vec<TimestampedTranscription> {
        let history = self.history.read().await;
        history.clone()
    }

    /// Analyze recent speech patterns
    async fn analyze_patterns(&self) {
        let context = self.get_recent_context().await;

        if context.is_empty() {
            return;
        }

        // Count words in recent context
        let total_words: usize = context
            .iter()
            .map(|t| t.text.split_whitespace().count())
            .sum();

        let avg_words_per_chunk = if !context.is_empty() {
            total_words / context.len()
        } else {
            0
        };

        println!(
            "[Analysis] Recent activity: {} chunks, {} total words, avg {:.1} words/chunk",
            context.len(),
            total_words,
            avg_words_per_chunk as f32
        );

        // Detect if conversation is active
        if avg_words_per_chunk > 5 {
            println!("[Analysis] Active conversation detected");
        } else {
            println!("[Analysis] Sparse audio detected");
        }
    }

    /// Search for keywords in recent context
    async fn search_context(&self, keyword: &str) -> Vec<String> {
        let context = self.get_recent_context().await;
        context
            .iter()
            .filter(|t| t.text.to_lowercase().contains(keyword))
            .map(|t| t.text.clone())
            .collect()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Streaming AI Agent ===\n");

    // Get API key
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("Set OPENAI_API_KEY environment variable");

    // Configure with shorter chunks for real-time feel
    let config = Config {
        chunk_duration_secs: 2,
        model: "whisper-1".to_string(),
        endpoint: "https://api.openai.com/v1/audio/transcriptions".to_string(),
        out_file: None,
    };

    // Create streaming agent with 30-second context window
    let agent = Arc::new(StreamingAgent::new(30));

    // Start transcription service
    let mut service = TranscriptionService::new(config, api_key)?;
    let (mut receiver, _stream) = service.start().await?;

    println!("Agent is listening...");
    println!("Context window: 30 seconds");
    println!("Press Ctrl+C to stop.\n");

    // Spawn periodic analysis task
    let analysis_agent = agent.clone();
    tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(10));
        loop {
            interval.tick().await;
            analysis_agent.analyze_patterns().await;
        }
    });

    // Process transcription events
    while let Some(event) = receiver.recv().await {
        match event {
            TranscriptionEvent::Transcription { chunk_id, text } => {
                if !text.is_empty() {
                    println!("[{}] {}", chunk_id, text);
                    agent.add_transcription(chunk_id, text.clone()).await;

                    // Example: Search for specific keywords
                    if text.to_lowercase().contains("important") {
                        println!("[Alert] Important keyword detected!");
                        let related = agent.search_context("important").await;
                        println!("[Alert] Related context: {:?}", related);
                    }
                }
            }
            TranscriptionEvent::Error { chunk_id, error } => {
                eprintln!("[Error] Chunk {}: {}", chunk_id, error);
            }
        }
    }

    Ok(())
}
