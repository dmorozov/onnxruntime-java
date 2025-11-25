package com.badu.ai.onnx.metrics;

import lombok.Builder;
import lombok.Value;

/**
 * Immutable data class capturing timing and throughput metrics for inference operations.
 *
 * <p>Metrics are collected throughout the inference pipeline:
 * <ul>
 *   <li><b>Initialization</b>: Model loading time (cold start only)</li>
 *   <li><b>Tokenization</b>: Input text to token IDs conversion time</li>
 *   <li><b>Encoder</b>: Encoder forward pass time (T5/BART models)</li>
 *   <li><b>Decoder</b>: Auto-regressive decoding loop time</li>
 *   <li><b>TTFT</b>: Time to first token (streaming mode only)</li>
 *   <li><b>Throughput</b>: Tokens generated per second</li>
 * </ul>
 *
 * <p>This class is immutable and thread-safe.
 *
 * @see com.badu.ai.onnx.metrics.PerformanceTracker
 */
@Value
@Builder
public class PerformanceMetrics {
    /**
     * Model loading time in milliseconds (cold start).
     * Zero for warm inference (model already loaded).
     */
    long initTimeMs;

    /**
     * Input tokenization time in milliseconds.
     * Includes text to token IDs conversion.
     */
    long tokenizationTimeMs;

    /**
     * Encoder forward pass time in milliseconds.
     * Applies to encoder-decoder models (T5, BART).
     */
    long encoderTimeMs;

    /**
     * Total decoder loop time in milliseconds.
     * Includes all auto-regressive iterations until EOS or max tokens.
     */
    long decoderTimeMs;

    /**
     * Time to first token in milliseconds (streaming mode only).
     * Measures latency from API call to first token emission.
     * Zero for batch mode.
     */
    long ttftMs;

    /**
     * End-to-end latency in milliseconds.
     * Calculated as: tokenizationTimeMs + encoderTimeMs + decoderTimeMs.
     */
    long totalTimeMs;

    /**
     * Generation throughput in tokens per second.
     * Calculated as: outputTokenCount / (decoderTimeMs / 1000.0f).
     */
    float tokensPerSecond;

    /**
     * Number of input tokens processed.
     */
    int inputTokenCount;

    /**
     * Number of output tokens generated.
     */
    int outputTokenCount;

    /**
     * Creates a PerformanceMetrics instance with calculated derived fields.
     *
     * <p>This builder automatically calculates:
     * <ul>
     *   <li>totalTimeMs = tokenizationTimeMs + encoderTimeMs + decoderTimeMs</li>
     *   <li>tokensPerSecond = outputTokenCount / (decoderTimeMs / 1000.0f)</li>
     * </ul>
     */
    public static class PerformanceMetricsBuilder {
        /**
         * Builds the PerformanceMetrics with calculated fields.
         *
         * @return Immutable PerformanceMetrics instance
         */
        public PerformanceMetrics build() {
            // Calculate total time if not set
            if (totalTimeMs == 0) {
                totalTimeMs = tokenizationTimeMs + encoderTimeMs + decoderTimeMs;
            }

            // Calculate throughput if not set
            if (tokensPerSecond == 0.0f && decoderTimeMs > 0) {
                tokensPerSecond = outputTokenCount / (decoderTimeMs / 1000.0f);
            }

            return new PerformanceMetrics(
                    initTimeMs,
                    tokenizationTimeMs,
                    encoderTimeMs,
                    decoderTimeMs,
                    ttftMs,
                    totalTimeMs,
                    tokensPerSecond,
                    inputTokenCount,
                    outputTokenCount
            );
        }
    }
}
