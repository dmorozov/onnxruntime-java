package com.badu.ai.onnx;

import com.badu.ai.onnx.metrics.PerformanceMetrics;

/**
 * Immutable response object containing inference results and performance metrics. Created via
 * Builder pattern with validation.
 * <p>
 * All instances are immutable and thread-safe.
 *
 * @see OnnxInference
 */
public final class InferenceResponse {

  private final boolean isSuccess;
  private final String responseText;
  private final String error;
  private final PerformanceMetrics metrics;

  private InferenceResponse(Builder builder) {
    this.isSuccess = builder.isSuccess;
    this.responseText = builder.responseText;
    this.error = builder.error;
    this.metrics = builder.metrics;
  }

  /**
   * Returns success/failure flag for inference operation.
   *
   * @return true if inference succeeded, false if error occurred
   */
  public boolean isSuccess() {
    return isSuccess;
  }

  /**
   * Returns generated text (null if isSuccess=false and no partial output).
   *
   * @return generated text or null
   */
  public String getResponseText() {
    return responseText;
  }

  /**
   * Returns error message (null if isSuccess=true).
   *
   * @return error message or null
   */
  public String getError() {
    return error;
  }

  /**
   * Returns performance metrics for this inference operation.
   *
   * @return performance metrics (may be null for error cases)
   */
  public PerformanceMetrics getMetrics() {
    return metrics;
  }

  /**
   * Returns model initialization time in milliseconds.
   *
   * @return initialization time in ms (0 if model already loaded)
   */
  public long getInitTimeMs() {
    return metrics != null ? metrics.getInitTimeMs() : 0;
  }

  /**
   * Returns tokenization time in milliseconds.
   *
   * @return tokenization time in ms
   */
  public long getTokenizationTimeMs() {
    return metrics != null ? metrics.getTokenizationTimeMs() : 0;
  }

  /**
   * Returns time to first token generation in milliseconds.
   *
   * @return time to first token in ms
   */
  public long getTimeToFirstTokenMs() {
    return metrics != null ? metrics.getTtftMs() : 0;
  }

  /**
   * Returns generation throughput in tokens per second.
   *
   * @return tokens per second
   */
  public double getTokensPerSecond() {
    return metrics != null ? metrics.getTokensPerSecond() : 0.0;
  }

  /**
   * Returns total inference time in milliseconds (tokenization + generation).
   *
   * @return total time in ms
   */
  public long getTotalTimeMs() {
    return metrics != null ? metrics.getTotalTimeMs() : 0;
  }

  /**
   * Returns human-readable string representation with all fields.
   *
   * @return formatted string with success status, text/error, and metrics
   */
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("InferenceResponse{");
    sb.append("isSuccess=").append(isSuccess);
    if (isSuccess) {
      sb.append(", responseText='")
          .append(responseText != null && responseText.length() > 100
              ? responseText.substring(0, 100) + "..."
              : responseText)
          .append("'");
    } else {
      sb.append(", error='").append(error).append("'");
    }
    if (metrics != null) {
      sb.append(", metrics={");
      sb.append("init=").append(metrics.getInitTimeMs()).append("ms, ");
      sb.append("tokenization=").append(metrics.getTokenizationTimeMs()).append("ms, ");
      sb.append("encoder=").append(metrics.getEncoderTimeMs()).append("ms, ");
      sb.append("decoder=").append(metrics.getDecoderTimeMs()).append("ms, ");
      sb.append("ttft=").append(metrics.getTtftMs()).append("ms, ");
      sb.append("tps=").append(String.format("%.1f", metrics.getTokensPerSecond())).append(", ");
      sb.append("total=").append(metrics.getTotalTimeMs()).append("ms");
      sb.append("}");
    }
    sb.append("}");
    return sb.toString();
  }

  /**
   * Creates a new builder for InferenceResponse.
   *
   * @return a new builder instance
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Builder pattern for InferenceResponse with validation.
   */
  public static class Builder {
    private boolean isSuccess;
    private String responseText;
    private String error;
    private PerformanceMetrics metrics;

    /**
     * Sets success flag.
     *
     * @param success true if inference succeeded
     * @return this builder
     */
    public Builder success(boolean success) {
      this.isSuccess = success;
      return this;
    }

    /**
     * Sets response text.
     *
     * @param text generated text
     * @return this builder
     */
    public Builder responseText(String text) {
      this.responseText = text;
      return this;
    }

    /**
     * Sets error message.
     *
     * @param error error description
     * @return this builder
     */
    public Builder error(String error) {
      this.error = error;
      return this;
    }

    /**
     * Sets performance metrics.
     *
     * @param metrics performance metrics
     * @return this builder
     */
    public Builder metrics(PerformanceMetrics metrics) {
      this.metrics = metrics;
      return this;
    }

    /**
     * Builds InferenceResponse with validation.
     *
     * @return validated InferenceResponse instance
     * @throws IllegalStateException if validation fails
     */
    public InferenceResponse build() {
      // Success invariant: success requires responseText
      if (isSuccess && (responseText == null || responseText.isEmpty())) {
        throw new IllegalStateException("success=true requires non-empty responseText");
      }

      // Failure invariant: failure requires error message
      if (!isSuccess && (error == null || error.isEmpty())) {
        throw new IllegalStateException("success=false requires non-empty error message");
      }

      return new InferenceResponse(this);
    }
  }
}
