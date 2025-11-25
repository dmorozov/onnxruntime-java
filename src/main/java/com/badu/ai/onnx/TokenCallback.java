package com.badu.ai.onnx;

import com.badu.ai.onnx.metrics.PerformanceMetrics;

/**
 * Callback interface for real-time streaming token generation.
 *
 * <p>Allows progressive display of generated text as tokens are decoded, improving UX
 * responsiveness for long-running summarization tasks.
 *
 * <p>Usage example:
 * <pre>{@code
 * TokenCallback callback = new TokenCallback() {
 *   @Override
 *   public void onToken(int tokenId, String tokenText, int position, boolean isLast) {
 *     System.out.print(tokenText); // Progressive display
 *   }
 *
 *   @Override
 *   public void onComplete(String summary, PerformanceMetrics metrics) {
 *     System.out.println("\nComplete! TTFT: " + metrics.getTimeToFirstTokenMs() + "ms");
 *   }
 *
 *   @Override
 *   public void onError(Exception e) {
 *     System.err.println("Error: " + e.getMessage());
 *   }
 * };
 *
 * inference.streamGenerate("summarize: Long document...", callback);
 * }</pre>
 *
 * <p><b>Thread Safety:</b> Callbacks are invoked sequentially from the decoder thread.
 * Implementations should avoid blocking operations to maintain streaming throughput.
 *
 * <p><b>Error Handling:</b> If any callback method throws an exception, streaming stops
 * immediately and {@link #onError(Exception)} is invoked with the exception.
 *
 * @see OnnxInference#streamGenerate(String, TokenCallback)
 * @see PerformanceMetrics#getTimeToFirstTokenMs()
 */
public interface TokenCallback {

  /**
   * Called for each generated token during streaming.
   *
   * <p>This method is invoked sequentially for each token as it's decoded. The token text
   * is already decoded from token ID to UTF-8 string.
   *
   * <p><b>Performance Target:</b> TTFT (time to first token) should be <50ms from
   * {@code streamGenerate()} API call. Subsequent tokens should have <100ms intervals.
   *
   * @param tokenId the vocabulary token ID (for debugging/logging)
   * @param tokenText the decoded UTF-8 text for this token (may be empty for special tokens)
   * @param position zero-based position in the output sequence
   * @param isLast true if this is the last token (EOS reached or max length)
   */
  void onToken(int tokenId, String tokenText, int position, boolean isLast);

  /**
   * Called when streaming completes successfully.
   *
   * <p>Invoked after the final token with {@code isLast=true} has been delivered.
   * The complete summary and performance metrics are provided for verification and logging.
   *
   * <p><b>Consistency Guarantee:</b> The {@code summary} parameter contains the concatenation
   * of all {@code tokenText} values from {@link #onToken} calls, and should be identical to
   * batch mode output for the same input and generation config.
   *
   * @param summary the complete generated summary (all tokens concatenated)
   * @param metrics performance metrics including TTFT, total time, tokens/sec throughput
   */
  void onComplete(String summary, PerformanceMetrics metrics);

  /**
   * Called when an error occurs during streaming generation.
   *
   * <p>This includes:
   * <ul>
   *   <li>ONNX Runtime errors during encoder/decoder execution</li>
   *   <li>Tokenization errors (invalid vocabulary, encoding issues)</li>
   *   <li>Validation errors (output quality checks)</li>
   *   <li>Exceptions thrown by {@link #onToken} callback</li>
   * </ul>
   *
   * <p>After {@code onError()} is called, no further callbacks will be invoked.
   * The streaming session is terminated and resources are cleaned up.
   *
   * @param e the exception that caused the streaming to fail
   */
  void onError(Exception e);
}
