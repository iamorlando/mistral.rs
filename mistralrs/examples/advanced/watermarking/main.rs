//! Generate and score text with an independent SynthID-Text watermark key.

use anyhow::Result;
use either::Either;
use mistralrs::{
    IsqBits, ModelBuilder, RequestBuilder, SynthIdTextWatermark, SynthIdTextWatermarkConfig,
    TextMessageRole,
};

const MAX_TOKENS: usize = 512;

#[tokio::main]
async fn main() -> Result<()> {
    let config = SynthIdTextWatermarkConfig::new(std::env::var("MISTRALRS_WATERMARK_KEY")?)?;
    let detector = SynthIdTextWatermark::new(&config)?;
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Eight)
        .build()
        .await?;
    let request = RequestBuilder::new()
        .add_message(
            TextMessageRole::User,
            "Write a long story about a lunar garden.",
        )
        .set_sampler_temperature(0.8)
        .set_sampler_topk(40)
        .set_sampler_max_len(MAX_TOKENS)
        .set_sampler_watermark(config);
    let response = model.send_chat_request(request).await?;
    let text = response.choices[0].message.content.as_deref().unwrap_or("");
    println!("{text}");
    let tokens = model
        .tokenize(Either::Right(text.to_owned()), None, false, false, None)
        .await?;
    let evidence = detector.detect(&tokens, 0, &[])?;
    println!(
        "Mean g-value: {:?}; tokens scored: {} (requires threshold calibration)",
        evidence.mean_g_value, evidence.tokens_scored
    );
    Ok(())
}
