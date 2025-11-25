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
python download_model.py --model qwen3

# 4. Download specific variants only (faster)
python download_model.py --model qwen3 --variants full,int8,q4
```

### Using Shell Scripts (Legacy)

Individual bash scripts are available for each model:

```bash
./downloadQwen3Model.sh
./downloadFlanT5SmallModel.sh
./downloadLlama32Model.sh
./downloadPhi3Model.sh
```

## Supported Models

| Model | Name | Params | Architecture | Description |
|-------|------|--------|--------------|-------------|
| `flan-t5-small` | Flan-T5-Small | 80M | Encoder-Decoder | Fast, good for summarization |
| `flan-t5-base` | Flan-T5-Base | 250M | Encoder-Decoder | Better quality than Small |
| `distilbart` | DistilBART-CNN-12-6 | 306M | Encoder-Decoder | Optimized for summarization |
| `qwen3` | Qwen3-1.7B | 1.7B | Decoder-Only | Powerful multilingual LLM |
| `llama3` | Llama-3.2-1B | 1B | Decoder-Only | Meta's compact LLM |
| `phi3` | Phi-3-Mini-128k | 3.8B | Decoder-Only | Microsoft's efficient LLM |

## Model Variants

Each model comes in multiple quantization variants:

- **FULL (FP32)**: Best quality, largest size (~400MB), baseline performance
- **FP16**: Half precision, ~200MB, 2-3x GPU speedup, <0.1% quality loss
- **INT8**: 8-bit quantization, ~100MB, 30-40% CPU speedup, 98-99% quality
- **Q4**: 4-bit quantization, ~50MB, 50-60% CPU speedup, 95-97% quality

## Usage Examples

### Download All Variants

```bash
# Download Qwen3 with all quantization variants (FULL, FP16, INT8, Q4)
python download_model.py --model qwen3
```

This will download all model files to `models/Qwen3-1.7B-ONNX/`.

### Download Specific Variants (Recommended)

```bash
# Download only INT8 and Q4 variants (smaller, faster)
python download_model.py --model qwen3 --variants int8,q4

# Download only FULL variant for best quality
python download_model.py --model flan-t5-small --variants full
```

### Download to Custom Location

```bash
# Download to a specific directory
python download_model.py --model llama3 --output /custom/path/models
```

## Model Directories

After download, models are organized as follows:

### Encoder-Decoder Models (T5, BART)

```
flan-t5-small-ONNX/
├── config.json
├── tokenizer.json
└── onnx/
    ├── encoder_model.onnx          # FULL variant
    ├── encoder_model_int8.onnx     # INT8 variant
    ├── decoder_model.onnx          # FULL variant
    ├── decoder_model_int8.onnx     # INT8 variant
    ├── decoder_with_past_model.onnx
    └── ...
```

### Decoder-Only Models (Qwen3)

```
Qwen3-1.7B-ONNX/
├── config.json
├── tokenizer.json
└── onnx/
    ├── model.onnx                  # FULL variant
    ├── model.onnx_data
    ├── model_int8.onnx             # INT8 variant
    ├── model_q4.onnx               # Q4 variant
    └── ...
```

### Decoder-Only Models (Llama, Phi-3)

```
Llama-3.2-1B-Instruct-ONNX/
├── config.json
└── cpu_and_mobile/
    └── cpu-int4-rtn-block-32-acc-level-4/
        ├── genai_config.json
        ├── tokenizer.json
        ├── model.onnx
        └── model.onnx.data
```

## Using Downloaded Models in Java

After downloading, use the models in your Java code:

### Encoder-Decoder Models (T5, BART)

```java
ModelConfig config = ModelConfig.builder()
    .modelPath("models/flan-t5-small-ONNX")
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

### Decoder-Only Models (Qwen3)

```java
ModelConfig config = ModelConfig.builder()
    .modelPath("models/Qwen3-1.7B-ONNX")
    .variant(ModelVariant.INT8)
    .build();

GenerationConfig genConfig = GenerationConfig.builder()
    .maxOutputTokens(50)
    .temperature(0.7f)
    .build();

try (OnnxInference inference = OnnxInference.create(config, genConfig)) {
    InferenceResponse response = inference.generate("What is AI?");
    System.out.println(response.getResponseText());
}
```

### Decoder-Only Models (Llama 3.2, Phi-3)

```java
ModelConfig config = ModelConfig.builder()
    .modelPath("models/Llama-3.2-1B-Instruct-ONNX/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4")
    .variant(ModelVariant.Q4)
    .build();

GenerationConfig genConfig = GenerationConfig.builder()
    .maxOutputTokens(50)
    .temperature(0.7f)
    .build();

try (OnnxInference inference = OnnxInference.create(config, genConfig)) {
    InferenceResponse response = inference.generate("Explain quantum computing");
    System.out.println(response.getResponseText());
}
```

## Disk Space Requirements

Downloading all variants of a model can require significant disk space:

| Model | All Variants | INT8 + Q4 Only | FULL Only |
|-------|--------------|----------------|-----------|
| Flan-T5-Small | ~2.5 GB | ~600 MB | ~400 MB |
| Flan-T5-Base | ~4.5 GB | ~1.2 GB | ~900 MB |
| DistilBART | ~5.0 GB | ~1.5 GB | ~1.2 GB |
| Qwen3-1.7B | ~7.0 GB | ~1.8 GB | ~3.5 GB |
| Llama-3.2-1B | ~700 MB | ~700 MB | N/A |
| Phi-3-Mini | ~2.5 GB | ~2.5 GB | N/A |

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
ls -lh Qwen3-1.7B-ONNX/onnx/

# Test with Java
./mvnw test -Pgpu,integration-tests -Dtest=Qwen3IntegrationTest
```

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
