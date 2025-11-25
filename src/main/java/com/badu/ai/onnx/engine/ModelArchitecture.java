package com.badu.ai.onnx.engine;

/**
 * Enum representing supported model architectures.
 *
 * <p>Each architecture defines a specific inference pattern:
 * <ul>
 *   <li>SIMPLE_GENAI: Legacy single-session generation (deprecated)</li>
 *   <li>T5_ENCODER_DECODER: Encoder-decoder models like T5, BART</li>
 *   <li>DECODER_ONLY: Decoder-only models like GPT, Qwen, Phi-3 (future)</li>
 * </ul>
 */
public enum ModelArchitecture {

    /**
     * Encoder-decoder architecture (e.g., T5, BART, DistilBART).
     * Uses separate encoder and decoder ONNX sessions with auto-regressive decoding.
     */
    T5_ENCODER_DECODER,

    /**
     * Decoder-only architecture (e.g., GPT, Qwen, Phi-3).
     * Future implementation for causal language models.
     */
    DECODER_ONLY
}
