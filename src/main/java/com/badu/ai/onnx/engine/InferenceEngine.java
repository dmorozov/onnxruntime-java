package com.badu.ai.onnx.engine;

import com.badu.ai.onnx.InferenceException;
import com.badu.ai.onnx.InferenceResponse;
import com.badu.ai.onnx.TokenCallback;
import com.badu.ai.onnx.config.ModelConfig;
import com.badu.ai.onnx.config.GenerationConfig;
import com.badu.ai.onnx.metrics.PerformanceMetrics;

/**
 * Common interface for all inference engine implementations.
 *
 * <p>Each inference engine handles a specific model architecture:
 * <ul>
 *   <li>SimpleGenAIEngine: Legacy single-session generation</li>
 *   <li>T5EncoderDecoderEngine: Encoder-decoder models (T5, BART, DistilBART)</li>
 *   <li>DecoderOnlyEngine: Future support for GPT-style models</li>
 * </ul>
 *
 * <p>Usage example:
 * <pre>{@code
 * InferenceEngine engine = InferenceEngineFactory.createEngine(ModelArchitecture.T5_ENCODER_DECODER);
 * engine.initialize(modelConfig, generationConfig);
 * InferenceResponse response = engine.generate("Summarize: This is a test");
 * System.out.println(response.getResponseText());
 * engine.close();
 * }</pre>
 *
 * <p>Thread Safety: Implementations are not required to be thread-safe. Callers should
 * synchronize access if the same engine instance is used from multiple threads.
 *
 * <p>Resource Management: All implementations must properly release resources in the
 * {@link #close()} method. Use try-with-resources for automatic cleanup.
 */
public interface InferenceEngine extends AutoCloseable {

    /**
     * Initializes the inference engine with model and generation configuration.
     *
     * <p>This method loads model files, initializes tokenizers, and prepares
     * the engine for inference. Must be called before any generation methods.
     *
     * @param modelConfig model configuration (model path, variant, device type, etc.)
     * @param generationConfig generation parameters (temperature, top-k, max tokens, etc.)
     * @throws InferenceException if initialization fails (model loading, tokenizer error, etc.)
     */
    void initialize(ModelConfig modelConfig, GenerationConfig generationConfig)
        throws InferenceException;

    /**
     * Generates text from input prompt (blocking mode).
     *
     * <p>This method blocks until generation completes. For real-time token streaming,
     * use {@link #generateStreaming(String, TokenCallback)} instead.
     *
     * @param prompt input text prompt
     * @return inference response with generated text, metrics, and validation results
     * @throws InferenceException if generation fails
     * @throws IllegalStateException if engine not initialized
     */
    InferenceResponse generate(String prompt) throws InferenceException;

    /**
     * Generates text with streaming token callback.
     *
     * <p>This method invokes the callback for each token as it's generated,
     * enabling real-time UI updates. The callback receives:
     * <ul>
     *   <li>Token ID and text</li>
     *   <li>Position in sequence</li>
     *   <li>isLast flag for final token</li>
     * </ul>
     *
     * <p>Performance targets:
     * <ul>
     *   <li>TTFT (time to first token): <50ms</li>
     *   <li>Token intervals: <100ms between consecutive tokens</li>
     * </ul>
     *
     * @param prompt input text prompt
     * @param callback token callback for progressive updates
     * @return inference response with complete text and metrics (including TTFT)
     * @throws InferenceException if generation fails
     * @throws IllegalStateException if engine not initialized
     * @throws IllegalArgumentException if callback is null
     */
    InferenceResponse generateStreaming(String prompt, TokenCallback callback)
        throws InferenceException;

    /**
     * Returns current performance metrics.
     *
     * <p>Metrics include:
     * <ul>
     *   <li>Initialization time</li>
     *   <li>Tokenization time</li>
     *   <li>Encoder time (for encoder-decoder models)</li>
     *   <li>Decoder/generation time</li>
     *   <li>TTFT (time to first token, for streaming)</li>
     *   <li>Total time and throughput (tokens/sec)</li>
     * </ul>
     *
     * @return performance metrics from last generation
     */
    PerformanceMetrics getMetrics();

    /**
     * Checks if the engine is initialized and ready for inference.
     *
     * @return true if initialized, false otherwise
     */
    boolean isInitialized();

    /**
     * Closes the engine and releases all resources.
     *
     * <p>After calling this method, the engine cannot be used for inference.
     * Implementations must ensure all ONNX sessions, tokenizers, and native
     * resources are properly released.
     *
     * <p>Implementations should be idempotent - calling close() multiple times
     * should be safe and not throw exceptions.
     */
    @Override
    void close();
}
