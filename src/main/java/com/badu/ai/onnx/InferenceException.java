package com.badu.ai.onnx;

import com.badu.ai.onnx.config.ModelVariant;
import com.badu.ai.onnx.config.GenerationConfig;

/**
 * Exception thrown during inference operations with context-rich error messages.
 *
 * <p>This exception provides detailed context about the failure including:
 * <ul>
 *   <li>Model variant being used</li>
 *   <li>Input token count (if available)</li>
 *   <li>Generation configuration</li>
 *   <li>Root cause exception</li>
 *   <li>Corrective actions for common issues</li>
 * </ul>
 *
 * <p>Usage example:
 * <pre>{@code
 * try {
 *   InferenceResponse response = inference.generateT5(input);
 * } catch (InferenceException e) {
 *   System.err.println("Inference failed: " + e.getMessage());
 *   System.err.println("Model variant: " + e.getModelVariant());
 *   System.err.println("Input tokens: " + e.getInputTokenCount());
 *   System.err.println("Corrective action: " + e.getCorrectiveAction());
 * }
 * }</pre>
 *
 * @see com.badu.ai.onnx.OnnxInference
 */
public class InferenceException extends RuntimeException {

  private final ModelVariant modelVariant;
  private final Integer inputTokenCount;
  private final GenerationConfig generationConfig;
  private final String correctiveAction;
  private final ErrorType errorType;

  /**
   * Error types for categorizing inference failures.
   */
  public enum ErrorType {
    /** Input validation failed (invalid UTF-8, control characters, etc.) */
    INPUT_VALIDATION,

    /** Output validation failed (token ID leakage, low readability, etc.) */
    OUTPUT_VALIDATION,

    /** Tokenization failed (invalid input, encoding error, etc.) */
    TOKENIZATION,

    /** Encoder execution failed (ONNX Runtime error, invalid input shape, etc.) */
    ENCODER,

    /** Decoder execution failed (ONNX Runtime error, invalid config, etc.) */
    DECODER,

    /** Model initialization failed (missing files, invalid configuration, etc.) */
    INITIALIZATION,

    /** Unknown error type */
    UNKNOWN
  }

  /**
   * Creates an InferenceException with full context.
   *
   * @param message Error message
   * @param errorType Type of error
   * @param modelVariant Model variant (FULL/Q4/INT8)
   * @param inputTokenCount Number of input tokens (null if unknown)
   * @param generationConfig Generation configuration
   * @param correctiveAction Suggested corrective action
   * @param cause Root cause exception
   */
  public InferenceException(String message, ErrorType errorType, ModelVariant modelVariant,
                             Integer inputTokenCount, GenerationConfig generationConfig,
                             String correctiveAction, Throwable cause) {
    super(buildFullMessage(message, errorType, modelVariant, inputTokenCount, generationConfig,
        correctiveAction), cause);
    this.errorType = errorType;
    this.modelVariant = modelVariant;
    this.inputTokenCount = inputTokenCount;
    this.generationConfig = generationConfig;
    this.correctiveAction = correctiveAction;
  }

  /**
   * Creates an InferenceException with partial context (no generation config).
   */
  public InferenceException(String message, ErrorType errorType, ModelVariant modelVariant,
                             Integer inputTokenCount, String correctiveAction, Throwable cause) {
    this(message, errorType, modelVariant, inputTokenCount, null, correctiveAction, cause);
  }

  /**
   * Creates an InferenceException with minimal context.
   */
  public InferenceException(String message, ErrorType errorType, String correctiveAction,
                             Throwable cause) {
    this(message, errorType, null, null, null, correctiveAction, cause);
  }

  /**
   * Creates an InferenceException with just message and type.
   */
  public InferenceException(String message, ErrorType errorType, Throwable cause) {
    this(message, errorType, null, cause);
  }

  /**
   * Creates an InferenceException with just message and type (no cause).
   */
  public InferenceException(String message, ErrorType errorType) {
    this(message, errorType, null, null);
  }

  /**
   * Builds the full error message with all available context.
   */
  private static String buildFullMessage(String message, ErrorType errorType,
                                          ModelVariant modelVariant, Integer inputTokenCount,
                                          GenerationConfig generationConfig,
                                          String correctiveAction) {
    StringBuilder sb = new StringBuilder();
    sb.append("[").append(errorType).append("] ").append(message);

    if (modelVariant != null) {
      sb.append("\n  Model variant: ").append(modelVariant);
    }

    if (inputTokenCount != null) {
      sb.append("\n  Input tokens: ").append(inputTokenCount);
    }

    if (generationConfig != null) {
      sb.append("\n  Generation config: maxTokens=").append(generationConfig.getMaxOutputTokens())
          .append(", temperature=").append(generationConfig.getTemperature());
    }

    if (correctiveAction != null) {
      sb.append("\n  Corrective action: ").append(correctiveAction);
    }

    return sb.toString();
  }

  /**
   * Gets the error type.
   *
   * @return error type
   */
  public ErrorType getErrorType() {
    return errorType;
  }

  /**
   * Gets the model variant.
   *
   * @return model variant (may be null)
   */
  public ModelVariant getModelVariant() {
    return modelVariant;
  }

  /**
   * Gets the input token count.
   *
   * @return input token count (may be null)
   */
  public Integer getInputTokenCount() {
    return inputTokenCount;
  }

  /**
   * Gets the generation configuration.
   *
   * @return generation config (may be null)
   */
  public GenerationConfig getGenerationConfig() {
    return generationConfig;
  }

  /**
   * Gets the suggested corrective action.
   *
   * @return corrective action (may be null)
   */
  public String getCorrectiveAction() {
    return correctiveAction;
  }
}
