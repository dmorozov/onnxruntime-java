package com.badu.ai.onnx.config;

import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Whisper generation configuration.
 *
 * <p>This class represents the Whisper-specific generation configuration
 * typically found in generation_config.json. It includes parameters for
 * decoding, token suppression, and multilingual support.
 *
 * <p><strong>Key Parameters:</strong>
 * <ul>
 *   <li>is_multilingual: Whether model supports multiple languages or English-only</li>
 *   <li>max_length: Maximum sequence length (typically 448 for Whisper)</li>
 *   <li>suppress_tokens: Token IDs to suppress during generation</li>
 *   <li>eos_token_id: End-of-sequence token (typically 50256)</li>
 * </ul>
 *
 * <p><strong>Usage:</strong>
 * <pre>{@code
 * WhisperGenerationConfig config = WhisperGenerationConfig.builder()
 *     .maxLength(448)
 *     .eosTokenId(50256L)
 *     .isMultilingual(false)
 *     .build();
 * }</pre>
 *
 * @see WhisperConfigParser
 */
@Data
@Builder
public class WhisperGenerationConfig {

    /**
     * Whether the model is multilingual (supports multiple languages) or English-only.
     * <p>Default: false (English-only)
     */
    @Builder.Default
    private boolean isMultilingual = false;

    /**
     * Maximum sequence length for decoder.
     * <p>Default: 448 (Whisper standard)
     */
    @Builder.Default
    private int maxLength = 448;

    /**
     * End-of-sequence token ID.
     * <p>Default: 50256 (Whisper standard)
     */
    @Builder.Default
    private long eosTokenId = 50256;

    /**
     * Beginning-of-sequence token ID.
     * <p>Default: 50257 (Whisper standard)
     */
    @Builder.Default
    private long bosTokenId = 50257;

    /**
     * Padding token ID.
     * <p>Default: 50256 (same as EOS for Whisper)
     */
    @Builder.Default
    private long padTokenId = 50256;

    /**
     * Decoder start token ID.
     * <p>Default: 50257 (same as BOS for Whisper)
     */
    @Builder.Default
    private long decoderStartTokenId = 50257;

    /**
     * No timestamps token ID.
     * <p>Default: 50362 (Whisper standard)
     */
    @Builder.Default
    private long noTimestampsTokenId = 50362;

    /**
     * Previous start-of-transcript token ID.
     * <p>Default: 50360 (Whisper standard)
     */
    @Builder.Default
    private long prevSotTokenId = 50360;

    /**
     * Whether to return timestamps in transcription.
     * <p>Default: false
     */
    @Builder.Default
    private boolean returnTimestamps = false;

    /**
     * Maximum initial timestamp index.
     * <p>Default: 50 (Whisper standard)
     */
    @Builder.Default
    private int maxInitialTimestampIndex = 50;

    /**
     * List of token IDs to suppress during generation.
     * <p>These tokens are typically special tokens or tokens that should not appear
     * in transcription output (e.g., punctuation, numbers, etc.)
     */
    private List<Long> suppressTokens;

    /**
     * List of token IDs to suppress at the beginning of generation.
     * <p>Default: [220, 50256] (Whisper standard)
     */
    private List<Long> beginSuppressTokens;

    /**
     * Forced decoder ID pairs [position, token_id].
     * <p>Example: [[1, 50362]] forces no_timestamps token at position 1
     */
    private List<List<Integer>> forcedDecoderIds;

    /**
     * Alignment heads configuration for timestamp prediction.
     * <p>Format: [[layer, head], ...]
     */
    private List<List<Integer>> alignmentHeads;

    /**
     * Hallucination detection threshold.
     * <p>If a token sequence repeats more than this many times, stop generation.
     * <p>Default: 3 (reasonable for detecting hallucination loops)
     */
    @Builder.Default
    private int hallucinationRepetitionThreshold = 3;

    /**
     * Length of token sequence to check for repetition.
     * <p>Default: 5 tokens
     */
    @Builder.Default
    private int hallucinationSequenceLength = 5;

    /**
     * Creates a default Whisper generation config.
     *
     * @return Default configuration for English-only Whisper tiny model
     */
    public static WhisperGenerationConfig getDefault() {
        return WhisperGenerationConfig.builder()
                .isMultilingual(false)
                .maxLength(448)
                .eosTokenId(50256)
                .bosTokenId(50257)
                .padTokenId(50256)
                .decoderStartTokenId(50257)
                .noTimestampsTokenId(50362)
                .prevSotTokenId(50360)
                .returnTimestamps(false)
                .maxInitialTimestampIndex(50)
                .hallucinationRepetitionThreshold(3)
                .hallucinationSequenceLength(5)
                .build();
    }

    /**
     * Checks if a token should be suppressed.
     *
     * @param tokenId Token ID to check
     * @return true if token should be suppressed
     */
    public boolean shouldSuppressToken(long tokenId) {
        return suppressTokens != null && suppressTokens.contains(tokenId);
    }

    /**
     * Checks if a token should be suppressed at the beginning of generation.
     *
     * @param tokenId Token ID to check
     * @return true if token should be suppressed at the beginning
     */
    public boolean shouldSuppressTokenAtBegin(long tokenId) {
        return beginSuppressTokens != null && beginSuppressTokens.contains(tokenId);
    }

    @Override
    public String toString() {
        return String.format("WhisperGenerationConfig{multilingual=%s, maxLen=%d, eosToken=%d, " +
                        "suppressTokens=%d, hallucinationThreshold=%d}",
                isMultilingual, maxLength, eosTokenId,
                suppressTokens != null ? suppressTokens.size() : 0,
                hallucinationRepetitionThreshold);
    }
}
