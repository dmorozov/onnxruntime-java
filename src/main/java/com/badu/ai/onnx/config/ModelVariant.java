package com.badu.ai.onnx.config;

/**
 * Model variant enum representing different quantization levels.
 * Each variant specifies the encoder and decoder ONNX model files for T5-based models.
 */
public enum ModelVariant {
    /**
     * Full precision model (highest quality, largest size, slowest).
     * Files: encoder_model.onnx + decoder_model.onnx (+ model.onnx_data for large models)
     */
    FULL("encoder_model.onnx", "decoder_model.onnx"),

    /**
     * Half-precision (FP16) model (GPU-optimized, 2-3x faster on Tensor Cores, <0.1% quality loss).
     * Recommended for GPU inference with NVIDIA T4, V100, A100, or RTX 30xx/40xx.
     * Files: encoder_model_fp16.onnx + decoder_model_fp16.onnx
     */
    FP16("encoder_model_fp16.onnx", "decoder_model_fp16.onnx"),

    /**
     * 4-bit quantized model (balanced quality/performance/size).
     * Recommended default for most use cases.
     */
    Q4("encoder_model_q4.onnx", "decoder_model_q4.onnx"),

    /**
     * 8-bit quantized model (faster, smaller, slightly lower quality).
     */
    INT8("encoder_model_int8.onnx", "decoder_model_int8.onnx");

    private final String encoderFileName;
    private final String decoderFileName;

    ModelVariant(String encoderFileName, String decoderFileName) {
        this.encoderFileName = encoderFileName;
        this.decoderFileName = decoderFileName;
    }

    /**
     * Gets the encoder ONNX model file name for this variant.
     *
     * @return the encoder model file name
     */
    public String getEncoderFileName() {
        return encoderFileName;
    }

    /**
     * Gets the decoder ONNX model file name for this variant.
     *
     * @return the decoder model file name
     */
    public String getDecoderFileName() {
        return decoderFileName;
    }

    /**
     * Gets the encoder model file name for this variant.
     * Alias for getEncoderFileName() for backward compatibility.
     *
     * @return the encoder model file name
     */
    public String getFileName() {
        return encoderFileName;
    }
}
