# Model Download Guide

This directory contains scripts to download pre-exported ONNX models from Hugging Face.

## Quick Start

### Using Python Script (Recommended)

The Python script `download_model.py` provides a unified interface to download all supported models:

```bash
# 1. Ensure virtual environment is set up (only needed once)
source ./setup.sh

# 2. List available models
python download_model.py --list

# 3. Download a model (all variants)
python download_model.py --model google/flan-t5-small

# 4. Download specific variants only (faster)
python download_model.py --model google/flan-t5-large --variants full,int8,q4
```

## Model Variants

Each model comes in multiple quantization variants:

- **FULL (FP32)**: Best quality, largest size (~400MB), baseline performance
- **FP16**: Half precision, ~200MB, 2-3x GPU speedup, <0.1% quality loss
- **INT8**: 8-bit quantization, ~100MB, 30-40% CPU speedup, 98-99% quality
- **Q4**: 4-bit quantization, ~50MB, 50-60% CPU speedup, 95-97% quality

## Usage Examples

### Download All Variants

```bash
# Download flan-t5-small with all quantization variants (FULL, FP16, INT8, Q4)
python download_model.py --model flan-t5-small
```

This will download all model files to `models/flan-t5-small-HF/`.

## Using Downloaded Models in Java

After downloading, use the models in your Java code:

```java
ModelConfig config = ModelConfig.builder()
    .modelPath("models/flan-t5-small-HF")
    .variant(ModelVariant.INT8)
    .flavour("onnx")
    .build();

GenerationConfig genConfig = GenerationConfig.builder()
    .maxOutputTokens(128)
    .temperature(0.7f)
    .build();

try (OnnxInference inference = OnnxInference.create(config, genConfig)) {
    InferenceResponse response = inference.generate("Summarize: ...");
    System.out.println(response.getResponseText());
}
```

**Recommendation**: Download only the variants you need using `--variants` flag.

## Troubleshooting

### Python Virtual Environment

If you get `huggingface_hub not found`, ensure the virtual environment is set up:

```bash
source ./setup.sh
```

The script will automatically install `huggingface_hub` if needed.

### Network Issues

If downloads fail due to network issues, the script can be re-run. It will skip files that already exist.

### Disk Space

Check available disk space before downloading:

```bash
df -h .
```

### Authentication (for gated models)

Some models may require Hugging Face authentication:

```bash
# Install huggingface-cli
pip install huggingface_hub[cli]

# Login (creates token in ~/.cache/huggingface/token)
huggingface-cli login
```

Then re-run the download script.

## Advanced Usage

### Download Multiple Models

```bash
# Download multiple models in sequence
for model in flan-t5-small qwen3 llama3; do
    python download_model.py --model $model --variants int8,q4
done
```

### Verify Download

```bash
# Check if model files exist
ls -lh Qwen3-1.7B-HF/onnx/

### Clean Up Unused Variants

```bash
# Remove FP16 variants to save space (if not using GPU)
find . -name "*_fp16.onnx*" -delete

# Remove FULL variants to save space (if using quantized models)
find . -name "encoder_model.onnx" -o -name "decoder_model.onnx" -delete
```

## Model Sources

All models are downloaded from official Hugging Face repositories:

- **Flan-T5**: [Xenova/flan-t5-small](https://huggingface.co/Xenova/flan-t5-small), [Xenova/flan-t5-base](https://huggingface.co/Xenova/flan-t5-base)
- **DistilBART**: [Xenova/distilbart-cnn-12-6](https://huggingface.co/Xenova/distilbart-cnn-12-6)
- **Qwen3**: [onnx-community/Qwen3-1.7B-ONNX](https://huggingface.co/onnx-community/Qwen3-1.7B-ONNX)
- **Llama 3.2**: [onnx-community/Llama-3.2-1B-Instruct-ONNX](https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-ONNX)
- **Phi-3**: [microsoft/Phi-3-mini-128k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx)

## License

Each model has its own license. Check the model's Hugging Face page for license details:

- Flan-T5: Apache 2.0
- DistilBART: Apache 2.0
- Qwen3: Apache 2.0
- Llama 3.2: Llama 3.2 Community License
- Phi-3: MIT License

Always review and comply with the model's license terms.
