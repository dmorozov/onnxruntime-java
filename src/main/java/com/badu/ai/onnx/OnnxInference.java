package com.badu.ai.onnx;

import com.badu.ai.onnx.config.ModelConfig;
import com.badu.ai.onnx.engine.InferenceEngine;
import com.badu.ai.onnx.engine.InferenceEngineFactory;
import com.badu.ai.onnx.engine.ModelArchitecture;
import com.badu.ai.onnx.config.GenerationConfig;
import com.badu.ai.onnx.metrics.PerformanceMetrics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main inference class for ONNX model text generation. Provides simple API: prompt(String) →
 * generated text.
 *
 * <p>This class acts as a facade over different inference engines:
 * <ul>
 *   <li>T5EncoderDecoderEngine: For encoder-decoder models (T5, BART, DistilBART)</li>
 *   <li>SimpleGenAIEngine: Legacy single-session generation (deprecated)</li>
 *   <li>DecoderOnlyEngine: Future support for GPT-style models</li>
 * </ul>
 *
 * <p>The appropriate engine is automatically selected based on model files present.
 * For manual control, use {@link #create(ModelConfig, GenerationConfig, ModelArchitecture)}.
 *
 * <p>Usage example (automatic engine detection):
 * <pre>{@code
 * ModelConfig config = ModelConfig.builder()
 *     .modelPath("models/flan-t5-small-ONNX")
 *     .variant(ModelVariant.INT8)
 *     .build();
 *
 * OnnxInference inference = OnnxInference.create(config, GenerationConfig.DEFAULT);
 * InferenceResponse response = inference.generate("Summarize: The quick brown fox...");
 * System.out.println(response.getResponseText());
 * inference.close();
 * }</pre>
 *
 * <p>Usage example (streaming):
 * <pre>{@code
 * inference.generateStreaming("Summarize: This is a test", new TokenCallback() {
 *     public void onToken(int tokenId, String tokenText, int position, boolean isLast) {
 *         System.out.print(tokenText);
 *     }
 *     public void onComplete(String summary, PerformanceMetrics metrics) {
 *         System.out.println("\nDone! TTFT: " + metrics.getTimeToFirstTokenMs() + "ms");
 *     }
 *     public void onError(Exception e) {
 *         System.err.println("Error: " + e.getMessage());
 *     }
 * });
 * }</pre>
 *
 * <p>Uses singleton pattern for model lifecycle management. The model is loaded once and reused
 * across prompts.
 *
 * <p>Thread Safety: This class is not thread-safe. Synchronize access if using from multiple threads.
 *
 * <p>Resource Management: Always call {@link #close()} or use try-with-resources to release
 * native resources (ONNX sessions, tokenizers).
 */
public class OnnxInference implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(OnnxInference.class);

    private static volatile OnnxInference instance;
    private static final Object LOCK = new Object();

    private final ModelConfig modelConfig;
    private final GenerationConfig generationConfig;
    private final InferenceEngine engine;

    /**
     * Private constructor for singleton pattern. Uses default configurations.
     */
    private OnnxInference() {
        this(ModelConfig.builder().modelPath("models/flan-t5-small-ONNX").build(),
            GenerationConfig.DEFAULT);
    }

    /**
     * Private constructor with custom configurations. Auto-detects model architecture.
     *
     * @param modelConfig model configuration
     * @param generationConfig generation configuration
     */
    private OnnxInference(ModelConfig modelConfig, GenerationConfig generationConfig) {
        this(modelConfig, generationConfig, null);
    }

    /**
     * Private constructor with explicit engine selection.
     *
     * @param modelConfig model configuration
     * @param generationConfig generation configuration
     * @param architecture model architecture (null for auto-detection)
     */
    private OnnxInference(ModelConfig modelConfig, GenerationConfig generationConfig,
                          ModelArchitecture architecture) {
        this.modelConfig = modelConfig;
        this.generationConfig = generationConfig;

        // Create appropriate engine
        if (architecture != null) {
            logger.trace("Creating inference engine with explicit architecture: {}", architecture);
            this.engine = InferenceEngineFactory.createEngine(architecture);
        } else {
            logger.trace("Auto-detecting inference engine from model files");
            this.engine = InferenceEngineFactory.autoDetect(modelConfig);
        }

        // Initialize engine
        try {
            engine.initialize(modelConfig, generationConfig);
            logger.debug("OnnxInference initialized successfully with {} engine",
                engine.getClass().getSimpleName());
        } catch (InferenceException e) {
            logger.error("Failed to initialize inference engine", e);
            throw new RuntimeException("Failed to initialize inference engine: " + e.getMessage(), e);
        }
    }

    /**
     * Gets the singleton instance with default configuration. Thread-safe double-checked locking.
     *
     * @return singleton OnnxInference instance
     */
    public static OnnxInference getInstance() {
        if (instance == null) {
            synchronized (LOCK) {
                if (instance == null) {
                    instance = new OnnxInference();
                }
            }
        }
        return instance;
    }

    /**
     * Creates a new instance with custom configuration. Does not use singleton pattern.
     * Auto-detects model architecture based on files present.
     *
     * @param modelConfig model configuration
     * @param generationConfig generation configuration
     * @return new OnnxInference instance
     * @throws IllegalArgumentException if modelConfig or generationConfig is null
     */
    public static OnnxInference create(ModelConfig modelConfig, GenerationConfig generationConfig) {
        if (modelConfig == null) {
            throw new IllegalArgumentException("modelConfig cannot be null");
        }
        if (generationConfig == null) {
            throw new IllegalArgumentException("generationConfig cannot be null");
        }
        return new OnnxInference(modelConfig, generationConfig);
    }

    /**
     * Creates a new instance with explicit architecture selection.
     * Use this method when you want to override auto-detection.
     *
     * @param modelConfig model configuration
     * @param generationConfig generation configuration
     * @param architecture model architecture to use
     * @return new OnnxInference instance
     * @throws IllegalArgumentException if any parameter is null
     */
    public static OnnxInference create(ModelConfig modelConfig, GenerationConfig generationConfig,
                                        ModelArchitecture architecture) {
        if (modelConfig == null) {
            throw new IllegalArgumentException("modelConfig cannot be null");
        }
        if (generationConfig == null) {
            throw new IllegalArgumentException("generationConfig cannot be null");
        }
        if (architecture == null) {
            throw new IllegalArgumentException("architecture cannot be null");
        }
        return new OnnxInference(modelConfig, generationConfig, architecture);
    }

    /**
     * Simple prompt interface: takes prompt text, returns generated text. This is the main public API
     * method.
     *
     * <p>Internally delegates to the appropriate inference engine based on model architecture.
     *
     * @param promptText user prompt text
     * @return generated text
     * @throws RuntimeException if generation fails
     */
    public String prompt(String promptText) {
        InferenceResponse response = generate(promptText);
        if (!response.isSuccess()) {
            throw new RuntimeException("Generation failed: " + response.getError());
        }
        return response.getResponseText();
    }

    /**
     * Generates text in blocking mode with full metrics. Returns InferenceResponse with generated
     * text and performance metrics.
     *
     * <p><b>Error Handling:</b> Throws InferenceException for unrecoverable errors (model loading,
     * ONNX Runtime errors, etc.). Returns InferenceResponse with success=false for recoverable
     * issues (validation failures, empty output, etc.).
     *
     * @param userPrompt user prompt text (will be formatted appropriately by the engine)
     * @return inference response with text and metrics
     * @throws InferenceException if generation fails due to unrecoverable error
     */
    public InferenceResponse generate(String userPrompt) throws InferenceException {
        return engine.generate(userPrompt);
    }

    /**
     * Generates text with custom system prompt.
     *
     * <p><b>Note:</b> System prompt support depends on the underlying engine.
     * Some engines may ignore or handle system prompts differently.
     *
     * <p><b>Error Handling:</b> Throws InferenceException for unrecoverable errors (model loading,
     * ONNX Runtime errors, etc.). Returns InferenceResponse with success=false for recoverable
     * issues (validation failures, empty output, etc.).
     *
     * @param systemPrompt custom system prompt (may be ignored by some engines)
     * @param userPrompt user prompt text
     * @return inference response with text and metrics
     * @throws InferenceException if generation fails due to unrecoverable error
     */
    public InferenceResponse generate(String systemPrompt, String userPrompt) throws InferenceException {
        // Note: Current engines don't use system prompt separately
        // This is kept for backward compatibility
        return generate(userPrompt);
    }

    /**
     * Generates text in streaming mode with token callback.
     *
     * <p>The callback receives each token as it's generated, enabling real-time UI updates.
     *
     * <p><b>Error Handling:</b> Throws InferenceException for unrecoverable errors (model loading,
     * ONNX Runtime errors, etc.). Returns InferenceResponse with success=false for recoverable
     * issues (validation failures, empty output, etc.). Errors are also passed to callback.onError().
     *
     * @param userPrompt user prompt text
     * @param callback token callback for streaming
     * @return inference response with complete text and metrics
     * @throws InferenceException if generation fails due to unrecoverable error
     * @throws IllegalArgumentException if callback is null
     */
    public InferenceResponse generateStreaming(String userPrompt, TokenCallback callback)
        throws InferenceException {
        if (callback == null) {
            throw new IllegalArgumentException("Callback cannot be null for streaming generation");
        }
        return generate(null, userPrompt, callback);
    }

    /**
     * Generates text with full control over system prompt and streaming.
     *
     * <p><b>Error Handling:</b> Throws InferenceException for unrecoverable errors.
     * For streaming mode, errors are also passed to callback.onError() before being thrown.
     *
     * @param systemPrompt custom system prompt (may be ignored by some engines)
     * @param userPrompt user prompt text (required)
     * @param callback token callback for streaming (null for blocking mode)
     * @return inference response with text and metrics
     * @throws InferenceException if generation fails due to unrecoverable error
     */
    public InferenceResponse generate(String systemPrompt, String userPrompt,
                                       TokenCallback callback) throws InferenceException {
        try {
            if (callback != null) {
                return engine.generateStreaming(userPrompt, callback);
            } else {
                return engine.generate(userPrompt);
            }
        } catch (InferenceException e) {
            logger.error("Generation failed", e);
            if (callback != null) {
                callback.onError(e);
            }
            throw e;
        }
    }

    /**
     * Initializes T5 components for encoder-decoder inference.
     *
     * @deprecated This method is no longer needed. The appropriate engine is automatically
     *             initialized based on model files. Use {@link #create(ModelConfig, GenerationConfig)}
     *             or {@link #getInstance()} instead.
     * @throws InferenceException if initialization fails
     */
    @Deprecated
    public void initializeT5Components() throws InferenceException {
        logger.warn("initializeT5Components() is deprecated and no longer needed. " +
            "Engine is automatically initialized based on model architecture.");
        // No-op for backward compatibility
    }

    /**
     * Generates text using T5 encoder-decoder architecture.
     *
     * @deprecated Use {@link #generate(String)} instead. The appropriate engine is automatically
     *             selected based on model architecture.
     * @param inputText input text to process
     * @return inference response with generated text and metrics
     * @throws InferenceException if generation fails due to unrecoverable error
     */
    @Deprecated
    public InferenceResponse generateT5(String inputText) throws InferenceException {
        logger.debug("generateT5() is deprecated, using generate() instead");
        return generate(inputText);
    }

    /**
     * Streaming generation with T5 encoder-decoder model.
     *
     * @deprecated Use {@link #generateStreaming(String, TokenCallback)} instead.
     * @param inputText User prompt text
     * @param callback Callback invoked for each generated token
     * @return InferenceResponse with complete generated text and metrics
     * @throws InferenceException if generation fails due to unrecoverable error
     */
    @Deprecated
    public InferenceResponse generateT5Streaming(String inputText, TokenCallback callback)
        throws InferenceException {
        logger.debug("generateT5Streaming() is deprecated, using generateStreaming() instead");
        return generateStreaming(inputText, callback);
    }

    /**
     * Returns the model configuration.
     *
     * @return model configuration
     */
    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    /**
     * Returns the generation configuration.
     *
     * @return generation configuration
     */
    public GenerationConfig getGenerationConfig() {
        return generationConfig;
    }

    /**
     * Returns current performance metrics from the underlying engine.
     *
     * @return performance metrics from last generation
     */
    public PerformanceMetrics getMetrics() {
        return engine.getMetrics();
    }

    /**
     * Checks if the inference engine is initialized and ready.
     *
     * @return true if initialized
     */
    public boolean isInitialized() {
        return engine.isInitialized();
    }

    /**
     * Closes the inference engine and releases all resources.
     *
     * <p>After closing, this instance cannot be used for inference.
     * All ONNX sessions, tokenizers, and native resources are properly released.
     */
    @Override
    public void close() {
        if (engine != null) {
            try {
                engine.close();
                logger.trace("OnnxInference resources released");
            } catch (Exception e) {
                logger.error("Error closing inference engine", e);
            }
        }
    }

    /**
     * Resets the singleton instance. Useful for testing or when you want to reload the model.
     * Thread-safe.
     */
    public static void resetInstance() {
        synchronized (LOCK) {
            if (instance != null) {
                instance.close();
                instance = null;
            }
        }
    }
}
