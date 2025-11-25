package com.badu.ai.onnx.streaming;

import lombok.Builder;
import lombok.Value;

/**
 * Represents a complete transcription segment with timing information.
 *
 * <p>A segment typically corresponds to a sentence, phrase, or utterance
 * with precise start and end timestamps.
 *
 * <p><strong>Usage Example:</strong>
 * <pre>{@code
 * TranscriptionSegment segment = TranscriptionSegment.builder()
 *     .text("Hello, how are you?")
 *     .startTime(1000)   // 1 second
 *     .endTime(2500)     // 2.5 seconds
 *     .confidence(0.95)
 *     .build();
 *
 * System.out.println("[" + segment.getStartTime() + "ms - " +
 *                    segment.getEndTime() + "ms] " + segment.getText());
 * }</pre>
 */
@Value
@Builder
public class TranscriptionSegment {

  /**
   * The transcribed text for this segment.
   */
  String text;

  /**
   * Start time of this segment in milliseconds from stream start.
   */
  long startTime;

  /**
   * End time of this segment in milliseconds from stream start.
   */
  long endTime;

  /**
   * Confidence score for this segment (0.0 to 1.0).
   * Higher values indicate more confident transcription.
   * Default: 1.0 (if not computed)
   */
  @Builder.Default
  double confidence = 1.0;

  /**
   * Optional speaker ID for speaker diarization.
   * Null if speaker diarization is not enabled.
   */
  Integer speakerId;

  /**
   * Gets the duration of this segment in milliseconds.
   *
   * @return Segment duration (endTime - startTime)
   */
  public long getDuration() {
    return endTime - startTime;
  }

  /**
   * Checks if this segment overlaps with another segment in time.
   *
   * @param other The other segment to check
   * @return true if segments overlap, false otherwise
   */
  public boolean overlaps(TranscriptionSegment other) {
    return this.startTime < other.endTime && other.startTime < this.endTime;
  }

  @Override
  public String toString() {
    return String.format("[%dms - %dms] (%.2f) %s",
        startTime, endTime, confidence, text);
  }
}
