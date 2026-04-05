# LFM2.5-VL-1.6B ANE Native Swift

Swift package and CLI for running the chunked CoreML export of LFM2.5-VL-1.6B on Apple Silicon with the Neural Engine as the primary compute target.

CoreML bundle and tokenizer assets: https://huggingface.co/mweinbach/LFM2.5-VL-1.6B-CoreML

## What is in this repo

- `Sources/LFM2CoreML`: reusable Swift library
- `Sources/ANEInferenceCLI`: runnable CLI wrapper
- `Tests/LFM2CoreMLTests`: basic formatter test coverage

## Requirements

- macOS 15+ or iOS 18+
- Xcode 16+ / Swift 6.1+
- Apple Silicon recommended

## Build

```bash
swift build
```

## Run

First download the CoreML bundle from Hugging Face so you have a local folder containing `CoreMLModels/`, `tokenizer.json`, `tokenizer_config.json`, `config.json`, `processor_config.json`, `generation_config.json`, and `chat_template.jinja`.

Text-only example:

```bash
swift run ANEInferenceCLI \
  --bundle-root /path/to/LFM2.5-VL-1.6B-CoreML \
  --prompt "Write a haiku about Apple silicon" \
  --max-new-tokens 64
```

Multimodal example:

```bash
swift run ANEInferenceCLI \
  --bundle-root /path/to/LFM2.5-VL-1.6B-CoreML \
  --image /path/to/image.png \
  --prompt "Describe the image briefly." \
  --max-new-tokens 64
```

Optional flags:

- `--system "..."`
- `--image /path/to/another-image.png` (repeatable)
- `--temperature 0.0`
- `--top-k 40`

## Notes

- Supports text-only and multimodal prompts, including multi-image inputs and the model's tiling/thumbnail preprocessing path.
- The vision path uses split CoreML models for patch embedding, encoder, and projector, plus shipped positional embeddings.
- The prompt formatter follows the shipped ChatML-style template from the original model assets.
- The CoreML export is fixed to a 4096-token context window.

## CoreML package

The matching CoreML weights and tokenizer assets live on Hugging Face:
https://huggingface.co/mweinbach/LFM2.5-VL-1.6B-CoreML
