package com.badu.ai.onnx.config;

import java.nio.file.Path;

/**
 * Immutable record holding paths to ONNX model files.
 *
 * <p>This record encapsulates the essential files for encoder-decoder models:
 * <ul>
 *   <li>Encoder model (.onnx file)</li>
 *   <li>Decoder model (.onnx file) - for first step without KV-cache</li>
 *   <li>Decoder with past model (.onnx file, optional) - for subsequent steps with KV-cache</li>
 *   <li>Tokenizer (tokenizer.json file)</li>
 * </ul>
 *
 * <p><b>Dual-Session Decoder Support:</b><br>
 * For optimal KV-cache performance, the decoder can be split into two models:
 * <ul>
 *   <li><b>decoder</b> (decoder_model.onnx): First decoding step without KV-cache inputs</li>
 *   <li><b>decoderWithPast</b> (decoder_with_past_model.onnx): Subsequent steps with KV-cache</li>
 * </ul>
 *
 * <p>If decoderWithPast is null, the decoder model is used for all steps (merged or non-cached mode).
 *
 * <p>Used by {@link ModelFileDiscovery} to return discovered model files
 * and by {@link ModelConfig.ModelConfigBuilder} to construct configurations.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Single decoder (merged or non-cached)
 * ModelFiles files = new ModelFiles(
 *     Paths.get("models/encoder_model_int8.onnx"),
 *     Paths.get("models/decoder_model_merged_int8.onnx"),
 *     null,  // No separate with_past decoder
 *     Paths.get("models/tokenizer.json")
 * );
 *
 * // Dual decoder (optimal KV-cache performance)
 * ModelFiles files = new ModelFiles(
 *     Paths.get("models/encoder_model_int8.onnx"),
 *     Paths.get("models/decoder_model_int8.onnx"),
 *     Paths.get("models/decoder_with_past_model_int8.onnx"),
 *     Paths.get("models/tokenizer.json")
 * );
 * }</pre>
 *
 * @param encoder path to encoder ONNX model file
 * @param decoder path to decoder ONNX model file (first step)
 * @param decoderWithPast path to decoder with past ONNX model file (optional, for subsequent steps)
 * @param tokenizer path to tokenizer.json file
 */
public record ModelFiles(Path encoder, Path decoder, Path decoderWithPast, Path tokenizer) {

    /**
     * Creates a ModelFiles record with validation.
     *
     * @param encoder path to encoder ONNX model file (required)
     * @param decoder path to decoder ONNX model file (required)
     * @param decoderWithPast path to decoder with past ONNX model file (optional, can be null)
     * @param tokenizer path to tokenizer.json file (required)
     * @throws IllegalArgumentException if any required path is null
     */
    public ModelFiles {
        if (encoder == null) {
            throw new IllegalArgumentException("Encoder path cannot be null");
        }
        if (decoder == null) {
            throw new IllegalArgumentException("Decoder path cannot be null");
        }
        if (tokenizer == null) {
            throw new IllegalArgumentException("Tokenizer path cannot be null");
        }
        // decoderWithPast is optional (can be null)
    }

    /**
     * Checks if dual-session decoder is available.
     *
     * @return true if decoderWithPast is not null (dual-session mode), false otherwise (single-session mode)
     */
    public boolean hasDualDecoder() {
        return decoderWithPast != null;
    }
}
