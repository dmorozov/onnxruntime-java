package com.badu.ai.onnx.streaming;

import lombok.Builder;
import lombok.Singular;
import lombok.Value;

import java.util.List;

/**
 * Final result of a streaming transcription session.
 *
 * <p>Contains all transcription segments, timing information, and performance metrics.
 *
 * <p><strong>Usage Example:</strong>
 * <pre>{@code
 * StreamingResult result = StreamingResult.builder()
 *     .segment(TranscriptionSegment.builder()
 *         .text("Hello world")
 *         .startTime(0)
 *         .endTime(1000)
 *         .build())
 *     .segment(TranscriptionSegment.builder()
 *         .text("This is a test")
 *         .startTime(1500)
 *         .endTime(3000)
 *         .build())
 *     .totalAudioDuration(5000)
 *     .processingTime(1200)
 *     .build();
 *
 * System.out.println("Full text: " + result.getFullText());
 * System.out.println("Real-time factor: " + result.getRealTimeFactor());
 * }</pre>
 */
@Value
@Builder
public class StreamingResult {

  /**
   * List of all transcription segments in chronological order.
   */
  @Singular
  List<TranscriptionSegment> segments;

  /**
   * Total audio duration processed in milliseconds.
   */
  long totalAudioDuration;

  /**
   * Total processing time in milliseconds.
   */
  long processingTime;

  /**
   * Average latency in milliseconds (time between audio capture and transcription).
   */
  @Builder.Default
  long averageLatency = 0;

  /**
   * Number of audio chunks processed.
   */
  @Builder.Default
  int chunksProcessed = 0;

  /**
   * Whether voice activity detection was used.
   */
  @Builder.Default
  boolean vadEnabled = false;

  /**
   * Gets the complete transcription text from all segments.
   *
   * @return Full transcription with segments joined by spaces
   */
  public String getFullText() {
    if (segments == null || segments.isEmpty()) {
      return "";
    }

    StringBuilder fullText = new StringBuilder();
    for (int i = 0; i < segments.size(); i++) {
      if (i > 0) {
        fullText.append(" ");
      }
      fullText.append(segments.get(i).getText());
    }
    return fullText.toString();
  }

  /**
   * Gets the real-time factor (RTF) for this transcription.
   *
   * <p>RTF = processing_time / audio_duration
   * <ul>
   *   <li>RTF < 1.0: Faster than real-time (can process live audio)</li>
   *   <li>RTF = 1.0: Exactly real-time</li>
   *   <li>RTF > 1.0: Slower than real-time (cannot keep up with live audio)</li>
   * </ul>
   *
   * @return Real-time factor
   */
  public double getRealTimeFactor() {
    if (totalAudioDuration == 0) {
      return 0.0;
    }
    return (double) processingTime / totalAudioDuration;
  }

  /**
   * Gets the average confidence across all segments.
   *
   * @return Average confidence score (0.0 to 1.0)
   */
  public double getAverageConfidence() {
    if (segments == null || segments.isEmpty()) {
      return 0.0;
    }

    double sum = 0.0;
    for (TranscriptionSegment segment : segments) {
      sum += segment.getConfidence();
    }
    return sum / segments.size();
  }

  /**
   * Gets the number of segments in this result.
   *
   * @return Number of transcription segments
   */
  public int getSegmentCount() {
    return segments != null ? segments.size() : 0;
  }

  @Override
  public String toString() {
    return String.format("StreamingResult{segments=%d, audio=%dms, processing=%dms, RTF=%.2f}",
        getSegmentCount(), totalAudioDuration, processingTime, getRealTimeFactor());
  }
}
