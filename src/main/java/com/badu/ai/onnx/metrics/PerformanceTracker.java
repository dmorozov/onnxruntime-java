package com.badu.ai.onnx.metrics;

/**
 * Tracks timing metrics for inference operations. Measures initialization, tokenization,
 * generation, and overall performance.
 * <p>
 * All timing methods record timestamps in nanoseconds internally, but getter methods return
 * milliseconds for consistency.
 */
public class PerformanceTracker {

  private long initStart = 0;
  private long initEnd = 0;
  private boolean initTimeReported = false;
  private long tokenizationStart = 0;
  private long tokenizationEnd = 0;
  private long firstTokenTime = 0;
  private long generationStart = 0;
  private long generationEnd = 0;
  private int tokenCount = 0;
  private long encoderStart = 0;
  private long encoderEnd = 0;
  private long decoderStart = 0;
  private long decoderEnd = 0;
  private int inputTokenCount = 0;
  private long ttftMs = 0; // Time to first token (for streaming mode)

  /**
   * Marks the start of model initialization.
   */
  public void startInit() {
    this.initStart = System.nanoTime();
  }

  /**
   * Marks the end of model initialization.
   */
  public void endInit() {
    this.initEnd = System.nanoTime();
  }

  /**
   * Marks the start of tokenization.
   */
  public void startTokenization() {
    this.tokenizationStart = System.nanoTime();
  }

  /**
   * Marks the end of tokenization.
   */
  public void endTokenization() {
    this.tokenizationEnd = System.nanoTime();
  }

  /**
   * Records the time when first token was generated.
   */
  public void recordFirstToken() {
    if (this.firstTokenTime == 0) { // Only record once
      this.firstTokenTime = System.nanoTime();
    }
  }

  /**
   * Marks the start of generation.
   */
  public void startGeneration() {
    this.generationStart = System.nanoTime();
  }

  /**
   * Marks the end of generation.
   *
   * @param tokenCount number of tokens generated
   */
  public void endGeneration(int tokenCount) {
    this.generationEnd = System.nanoTime();
    this.tokenCount = tokenCount;
  }

  /**
   * Marks the start of encoder execution.
   */
  public void startEncoder() {
    this.encoderStart = System.nanoTime();
  }

  /**
   * Marks the end of encoder execution.
   */
  public void endEncoder() {
    this.encoderEnd = System.nanoTime();
  }

  /**
   * Marks the start of decoder execution.
   */
  public void startDecoder() {
    this.decoderStart = System.nanoTime();
  }

  /**
   * Marks the end of decoder execution.
   */
  public void endDecoder() {
    this.decoderEnd = System.nanoTime();
  }

  /**
   * Records the number of input tokens.
   *
   * @param inputTokenCount number of input tokens
   */
  public void recordInputTokens(int inputTokenCount) {
    this.inputTokenCount = inputTokenCount;
  }

  /**
   * Records the time to first token (TTFT) for streaming mode.
   * This is used when TTFT is calculated externally (e.g., by DecoderExecutor).
   *
   * @param ttftMs time to first token in milliseconds
   */
  public void recordTTFT(long ttftMs) {
    this.ttftMs = ttftMs;
  }

  /**
   * Resets all timing data while preserving initialization times. Used between consecutive prompts
   * when model is already loaded. After reset, getInitTimeMs() will return 0 since model is already
   * initialized.
   */
  public void reset() {
    // Keep initStart/initEnd if already initialized, but mark as reported
    this.initTimeReported = true;
    // Reset other timings
    this.tokenizationStart = 0;
    this.tokenizationEnd = 0;
    this.firstTokenTime = 0;
    this.generationStart = 0;
    this.generationEnd = 0;
    this.tokenCount = 0;
    this.encoderStart = 0;
    this.encoderEnd = 0;
    this.decoderStart = 0;
    this.decoderEnd = 0;
    this.inputTokenCount = 0;
    this.ttftMs = 0;
  }

  /**
   * Gets model initialization time in milliseconds. Returns 0 if model was already initialized
   * (subsequent calls after reset).
   *
   * @return initialization time in ms (0 for subsequent requests)
   */
  public long getInitTimeMs() {
    if (initStart == 0 || initEnd == 0) {
      return 0;
    }
    // Return 0 for subsequent requests after reset (model already loaded)
    if (initTimeReported) {
      return 0;
    }
    return (initEnd - initStart) / 1_000_000;
  }

  /**
   * Gets tokenization time in milliseconds.
   *
   * @return tokenization time in ms
   */
  public long getTokenizationTimeMs() {
    if (tokenizationStart == 0 || tokenizationEnd == 0) {
      return 0;
    }
    return (tokenizationEnd - tokenizationStart) / 1_000_000;
  }

  /**
   * Gets time to first token in milliseconds. Measures from generation start to first token.
   * If TTFT was set explicitly via recordTTFT(), returns that value. Otherwise calculates
   * from timestamps.
   *
   * @return time to first token in ms
   */
  public long getTimeToFirstTokenMs() {
    // Return explicit TTFT if set (streaming mode)
    if (ttftMs > 0) {
      return ttftMs;
    }
    // Otherwise calculate from timestamps (batch mode)
    if (generationStart == 0 || firstTokenTime == 0) {
      return 0;
    }
    return (firstTokenTime - generationStart) / 1_000_000;
  }

  /**
   * Gets generation throughput in tokens per second.
   *
   * @return tokens per second (0.0 if no tokens generated)
   */
  public double getTokensPerSecond() {
    if (generationStart == 0 || generationEnd == 0 || tokenCount == 0) {
      return 0.0;
    }

    long generationTimeNs = generationEnd - generationStart;
    if (generationTimeNs == 0) {
      return 0.0;
    }

    double generationTimeSec = generationTimeNs / 1_000_000_000.0;
    return tokenCount / generationTimeSec;
  }

  /**
   * Gets total end-to-end time in milliseconds. Includes tokenization + generation.
   *
   * @return total time in ms
   */
  public long getTotalTimeMs() {
    long total = 0;

    // Add tokenization time
    total += getTokenizationTimeMs();

    // Add generation time
    if (generationStart != 0 && generationEnd != 0) {
      total += (generationEnd - generationStart) / 1_000_000;
    }

    return total;
  }

  /**
   * Gets encoder execution time in milliseconds.
   *
   * @return encoder time in ms
   */
  public long getEncoderTimeMs() {
    if (encoderStart == 0 || encoderEnd == 0) {
      return 0;
    }
    return (encoderEnd - encoderStart) / 1_000_000;
  }

  /**
   * Gets decoder execution time in milliseconds.
   *
   * @return decoder time in ms
   */
  public long getDecoderTimeMs() {
    if (decoderStart == 0 || decoderEnd == 0) {
      return 0;
    }
    return (decoderEnd - decoderStart) / 1_000_000;
  }

  /**
   * Builds a PerformanceMetrics object from current timing data.
   *
   * @return immutable PerformanceMetrics snapshot
   */
  public PerformanceMetrics toMetrics() {
    return PerformanceMetrics.builder()
        .initTimeMs(getInitTimeMs())
        .tokenizationTimeMs(getTokenizationTimeMs())
        .encoderTimeMs(getEncoderTimeMs())
        .decoderTimeMs(getDecoderTimeMs())
        .ttftMs(getTimeToFirstTokenMs())
        .totalTimeMs(getTotalTimeMs())
        .tokensPerSecond((float) getTokensPerSecond())
        .inputTokenCount(inputTokenCount)
        .outputTokenCount(tokenCount)
        .build();
  }
}
