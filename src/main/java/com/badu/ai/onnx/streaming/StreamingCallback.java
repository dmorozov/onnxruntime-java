package com.badu.ai.onnx.streaming;

/**
 * Callback interface for receiving real-time transcription results during streaming.
 *
 * <p>Implementations of this interface receive incremental transcription updates
 * as audio chunks are processed, enabling real-time speech-to-text applications.
 *
 * <p><strong>Usage Example:</strong>
 * <pre>{@code
 * StreamingCallback callback = new StreamingCallback() {
 *     @Override
 *     public void onPartialResult(String partialText, long timestamp, boolean isFinal) {
 *         if (isFinal) {
 *             System.out.println("Final: " + partialText);
 *         } else {
 *             System.out.print("Partial: " + partialText + "\r");
 *         }
 *     }
 *
 *     @Override
 *     public void onSegmentComplete(TranscriptionSegment segment) {
 *         System.out.println("[" + segment.getStartTime() + " - " +
 *                            segment.getEndTime() + "] " + segment.getText());
 *     }
 *
 *     @Override
 *     public void onError(Exception error) {
 *         System.err.println("Transcription error: " + error.getMessage());
 *     }
 *
 *     @Override
 *     public void onComplete(StreamingResult result) {
 *         System.out.println("Total transcription: " + result.getFullText());
 *     }
 * };
 *
 * engine.transcribeStreaming(audioStream, callback);
 * }</pre>
 *
 * <p><strong>Callback Sequence:</strong>
 * <ol>
 *   <li>Multiple {@code onPartialResult()} calls with isFinal=false (unstable text)</li>
 *   <li>Final {@code onPartialResult()} call with isFinal=true (stable text)</li>
 *   <li>{@code onSegmentComplete()} when a complete sentence/phrase is detected</li>
 *   <li>Repeat for next audio chunks</li>
 *   <li>{@code onComplete()} when stream ends</li>
 * </ol>
 *
 * @see TranscriptionSegment
 * @see StreamingResult
 */
public interface StreamingCallback {

  /**
   * Called when a partial transcription result is available.
   *
   * <p>Partial results are unstable and may change as more audio is processed.
   * When {@code isFinal} is true, the result is stable and won't change.
   *
   * @param partialText The partial transcription text
   * @param timestamp Timestamp in milliseconds from stream start
   * @param isFinal true if this is a final (stable) result, false for intermediate results
   */
  void onPartialResult(String partialText, long timestamp, boolean isFinal);

  /**
   * Called when a complete transcription segment is finalized.
   *
   * <p>A segment represents a complete sentence, phrase, or speech utterance
   * with accurate timing information.
   *
   * @param segment The completed transcription segment with timing
   */
  void onSegmentComplete(TranscriptionSegment segment);

  /**
   * Called when an error occurs during streaming transcription.
   *
   * <p>After this callback, no further transcription callbacks will be invoked
   * for the current stream.
   *
   * @param error The exception that occurred
   */
  void onError(Exception error);

  /**
   * Called when the streaming transcription completes successfully.
   *
   * <p>This is invoked after all segments have been processed and
   * {@code onSegmentComplete()} has been called for each segment.
   *
   * @param result The final streaming result with all segments and metadata
   */
  void onComplete(StreamingResult result);

  /**
   * Called when voice activity is detected in the audio stream.
   *
   * <p>This callback is optional and may be used to provide visual feedback
   * or trigger actions based on speech detection.
   *
   * @param isVoiceActive true if voice is detected, false during silence
   * @param timestamp Timestamp in milliseconds from stream start
   */
  default void onVoiceActivityChange(boolean isVoiceActive, long timestamp) {
    // Default: no-op (optional callback)
  }

  /**
   * Called periodically to report streaming progress.
   *
   * <p>This can be used to update progress indicators or monitor latency.
   *
   * @param audioProcessed Total audio processed in milliseconds
   * @param latencyMs Current processing latency (audio time - real time)
   */
  default void onProgress(long audioProcessed, long latencyMs) {
    // Default: no-op (optional callback)
  }
}
