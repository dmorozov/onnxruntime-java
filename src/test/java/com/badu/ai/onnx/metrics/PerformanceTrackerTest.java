package com.badu.ai.onnx.metrics;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for PerformanceTracker timing metrics.
 */
class PerformanceTrackerTest {

  private PerformanceTracker tracker;

  @BeforeEach
  void setUp() {
    tracker = new PerformanceTracker();
  }

  @Test
  @DisplayName("New tracker has zero times")
  void newTracker_hasZeroTimes() {
    assertEquals(0, tracker.getInitTimeMs());
    assertEquals(0, tracker.getTokenizationTimeMs());
    assertEquals(0, tracker.getTimeToFirstTokenMs());
    assertEquals(0.0, tracker.getTokensPerSecond(), 0.001);
    assertEquals(0, tracker.getTotalTimeMs());
  }

  @Test
  @DisplayName("Tracks initialization time correctly")
  void trackInitTime_calculatesCorrectly() throws InterruptedException {
    tracker.startInit();
    Thread.sleep(10); // Sleep 10ms
    tracker.endInit();

    long initTime = tracker.getInitTimeMs();
    assertTrue(initTime >= 10, "Init time should be at least 10ms, got: " + initTime);
    assertTrue(initTime < 100, "Init time should be less than 100ms, got: " + initTime);
  }

  @Test
  @DisplayName("Tracks tokenization time correctly")
  void trackTokenizationTime_calculatesCorrectly() throws InterruptedException {
    tracker.startTokenization();
    Thread.sleep(5); // Sleep 5ms
    tracker.endTokenization();

    long tokenizationTime = tracker.getTokenizationTimeMs();
    assertTrue(tokenizationTime >= 5, "Tokenization time should be at least 5ms");
    assertTrue(tokenizationTime < 50, "Tokenization time should be less than 50ms");
  }

  @Test
  @DisplayName("Tracks time to first token correctly")
  void trackTimeToFirstToken_calculatesCorrectly() throws InterruptedException {
    tracker.startGeneration();
    Thread.sleep(8); // Sleep 8ms
    tracker.recordFirstToken();

    long ttft = tracker.getTimeToFirstTokenMs();
    assertTrue(ttft >= 8, "Time to first token should be at least 8ms");
    assertTrue(ttft < 80, "Time to first token should be less than 80ms");
  }

  @Test
  @DisplayName("Records first token only once")
  void recordFirstToken_recordsOnlyOnce() throws InterruptedException {
    tracker.startGeneration();
    Thread.sleep(5);
    tracker.recordFirstToken();
    long firstTime = tracker.getTimeToFirstTokenMs();

    Thread.sleep(10);
    tracker.recordFirstToken(); // Second call should be ignored
    long secondTime = tracker.getTimeToFirstTokenMs();

    assertEquals(firstTime, secondTime, "First token time should not change on second call");
  }

  @Test
  @DisplayName("Calculates tokens per second correctly")
  void calculateTokensPerSecond_correctValue() throws InterruptedException {
    tracker.startGeneration();
    Thread.sleep(100); // 100ms = 0.1 seconds
    tracker.endGeneration(10); // 10 tokens

    double tps = tracker.getTokensPerSecond();
    // 10 tokens / ~0.1 seconds = ~100 tokens/sec
    assertTrue(tps >= 80, "TPS should be at least 80, got: " + tps);
    assertTrue(tps <= 120, "TPS should be at most 120, got: " + tps);
  }

  @Test
  @DisplayName("Calculates total time correctly")
  void calculateTotalTime_sumsAllTimes() throws InterruptedException {
    tracker.startTokenization();
    Thread.sleep(5);
    tracker.endTokenization();

    tracker.startGeneration();
    Thread.sleep(10);
    tracker.endGeneration(5);

    long totalTime = tracker.getTotalTimeMs();
    assertTrue(totalTime >= 15, "Total time should be at least 15ms, got: " + totalTime);
    assertTrue(totalTime < 50, "Total time should be less than 50ms, got: " + totalTime);
  }

  @Test
  @DisplayName("Reset clears timing data except init")
  void reset_clearsTimingData() throws InterruptedException {
    // Set some initial data
    tracker.startInit();
    Thread.sleep(5);
    tracker.endInit();
    long initTime = tracker.getInitTimeMs();

    tracker.startTokenization();
    Thread.sleep(5);
    tracker.endTokenization();

    tracker.startGeneration();
    tracker.recordFirstToken();
    tracker.endGeneration(10);

    // Verify data exists
    assertTrue(tracker.getTokenizationTimeMs() > 0);
    assertTrue(tracker.getTokensPerSecond() > 0);

    // Reset
    tracker.reset();

    // Init time returns 0 after reset (model already loaded for subsequent requests)
    assertEquals(0, tracker.getInitTimeMs());

    // Other times cleared
    assertEquals(0, tracker.getTokenizationTimeMs());
    assertEquals(0, tracker.getTimeToFirstTokenMs());
    assertEquals(0.0, tracker.getTokensPerSecond(), 0.001);
  }

  @Test
  @DisplayName("Returns zero for incomplete measurements")
  void incompleteMeasurements_returnZero() {
    // Start init but don't end
    tracker.startInit();
    assertEquals(0, tracker.getInitTimeMs());

    // Start tokenization but don't end
    tracker.startTokenization();
    assertEquals(0, tracker.getTokenizationTimeMs());

    // Start generation but don't record first token
    tracker.startGeneration();
    assertEquals(0, tracker.getTimeToFirstTokenMs());
  }

  @Test
  @DisplayName("Handles zero token generation")
  void zeroTokens_returnsZeroTPS() {
    tracker.startGeneration();
    tracker.endGeneration(0); // 0 tokens

    assertEquals(0.0, tracker.getTokensPerSecond(), 0.001);
  }

  @Test
  @DisplayName("Handles multiple measurement cycles")
  void multipleCycles_worksCorrectly() throws InterruptedException {
    // First cycle
    tracker.startTokenization();
    Thread.sleep(5);
    tracker.endTokenization();
    long firstTokenizationTime = tracker.getTokenizationTimeMs();

    tracker.reset();

    // Second cycle
    tracker.startTokenization();
    Thread.sleep(5);
    tracker.endTokenization();
    long secondTokenizationTime = tracker.getTokenizationTimeMs();

    assertTrue(firstTokenizationTime > 0);
    assertTrue(secondTokenizationTime > 0);
  }
}
