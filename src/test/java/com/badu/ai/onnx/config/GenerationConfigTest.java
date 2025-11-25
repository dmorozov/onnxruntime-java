package com.badu.ai.onnx.config;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for GenerationConfig builder and validation.
 */
class GenerationConfigTest {

  @Test
  @DisplayName("Builder creates valid config with defaults")
  void builder_defaults_createsValidConfig() {
    GenerationConfig config = GenerationConfig.builder().build();

    assertEquals(512, config.getMaxOutputTokens());
    assertEquals(0.7f, config.getTemperature(), 0.001);
    assertEquals(50, config.getTopK());
    assertEquals(0.9f, config.getTopP(), 0.001);
    assertEquals(1.0f, config.getRepetitionPenalty(), 0.001);
  }

  @Test
  @DisplayName("Builder creates valid config with all fields")
  void builder_allFields_createsValidConfig() {
    GenerationConfig config = GenerationConfig.builder()
        .maxOutputTokens(1024)
        .temperature(0.8f)
        .topK(50)
        .topP(0.95f)
        .repetitionPenalty(1.2f)
        .build();

    assertEquals(1024, config.getMaxOutputTokens());
    assertEquals(0.8f, config.getTemperature(), 0.001);
    assertEquals(50, config.getTopK());
    assertEquals(0.95f, config.getTopP(), 0.001);
    assertEquals(1.2f, config.getRepetitionPenalty(), 0.001);
  }

  @Test
  @DisplayName("Builder validates maxOutputTokens is positive")
  void builder_zeroMaxOutputTokens_throwsException() {
    IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
      GenerationConfig.builder()
          .maxOutputTokens(0)
          .temperature(0.7f)
          .topK(40)
          .topP(0.9f)
          .build();
    });

    assertTrue(exception.getMessage().contains("maxOutputTokens must be positive"));
  }

  @Test
  @DisplayName("Builder validates maxOutputTokens is positive (negative)")
  void builder_negativeMaxOutputTokens_throwsException() {
    IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
      GenerationConfig.builder()
          .maxOutputTokens(-100)
          .temperature(0.7f)
          .topK(40)
          .topP(0.9f)
          .build();
    });

    assertTrue(exception.getMessage().contains("maxOutputTokens must be positive"));
  }

  @Test
  @DisplayName("Builder validates temperature range (minimum)")
  void builder_temperatureTooLow_throwsException() {
    IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
      GenerationConfig.builder().maxOutputTokens(512).temperature(-0.1f).topK(40).topP(0.9f)
          .build();
    });

    assertTrue(exception.getMessage().contains("temperature must be in range [0.0, 2.0]"));
  }

  @Test
  @DisplayName("Builder validates temperature range (maximum)")
  void builder_temperatureTooHigh_throwsException() {
    IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
      GenerationConfig.builder().maxOutputTokens(512).temperature(2.1f).topK(40).topP(0.9f)
          .build();
    });

    assertTrue(exception.getMessage().contains("temperature must be in range [0.0, 2.0]"));
  }

  @Test
  @DisplayName("Builder allows temperature at boundaries")
  void builder_temperatureAtBoundaries_succeeds() {
    // Temperature = 0.0
    GenerationConfig config0 = GenerationConfig.builder().maxOutputTokens(512).temperature(0.0f)
        .topK(40).topP(0.9f).build();
    assertEquals(0.0, config0.getTemperature(), 0.001);

    // Temperature = 2.0
    GenerationConfig config2 = GenerationConfig.builder().maxOutputTokens(512).temperature(2.0f)
        .topK(40).topP(0.9f).build();
    assertEquals(2.0, config2.getTemperature(), 0.001);
  }

  @Test
  @DisplayName("Builder validates topK range (minimum)")
  void builder_topKTooLow_throwsException() {
    IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
      GenerationConfig.builder().maxOutputTokens(512).temperature(0.7f).topK(0).topP(0.9f)
          .build();
    });

    assertTrue(exception.getMessage().contains("topK must be in range [1, 100]"));
  }

  @Test
  @DisplayName("Builder validates topK range (maximum)")
  void builder_topKTooHigh_throwsException() {
    IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
      GenerationConfig.builder().maxOutputTokens(512).temperature(0.7f).topK(101).topP(0.9f)
          .build();
    });

    assertTrue(exception.getMessage().contains("topK must be in range [1, 100]"));
  }

  @Test
  @DisplayName("Builder allows topK at boundaries")
  void builder_topKAtBoundaries_succeeds() {
    // topK = 1
    GenerationConfig config1 = GenerationConfig.builder().maxOutputTokens(512).temperature(0.7f).topK(1)
        .topP(0.9f).build();
    assertEquals(1, config1.getTopK());

    // topK = 100
    GenerationConfig config100 = GenerationConfig.builder().maxOutputTokens(512).temperature(0.7f)
        .topK(100).topP(0.9f).build();
    assertEquals(100, config100.getTopK());
  }

  @Test
  @DisplayName("Builder validates topP range (minimum)")
  void builder_topPTooLow_throwsException() {
    IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
      GenerationConfig.builder().maxOutputTokens(512).temperature(0.7f).topK(40).topP(-0.01f)
          .build();
    });

    assertTrue(exception.getMessage().contains("topP must be in range [0.0, 1.0]"));
  }

  @Test
  @DisplayName("Builder validates topP range (maximum)")
  void builder_topPTooHigh_throwsException() {
    IllegalStateException exception = assertThrows(IllegalStateException.class, () -> {
      GenerationConfig.builder().maxOutputTokens(512).temperature(0.7f).topK(40).topP(1.01f)
          .build();
    });

    assertTrue(exception.getMessage().contains("topP must be in range [0.0, 1.0]"));
  }

  @Test
  @DisplayName("Builder allows topP at boundaries")
  void builder_topPAtBoundaries_succeeds() {
    // topP = 0.0
    GenerationConfig config0 = GenerationConfig.builder().maxOutputTokens(512).temperature(0.7f)
        .topK(40).topP(0.0f).build();
    assertEquals(0.0, config0.getTopP(), 0.001);

    // topP = 1.0
    GenerationConfig config1 = GenerationConfig.builder().maxOutputTokens(512).temperature(0.7f)
        .topK(40).topP(1.0f).build();
    assertEquals(1.0, config1.getTopP(), 0.001);
  }

  @Test
  @DisplayName("GenerationConfig is immutable")
  void generationConfig_immutability() {
    GenerationConfig config = GenerationConfig.builder().maxOutputTokens(512).temperature(0.7f).topK(40)
        .topP(0.9f).build();

    // Verify getters return same values
    assertEquals(512, config.getMaxOutputTokens());
    assertEquals(512, config.getMaxOutputTokens());

    assertEquals(0.7, config.getTemperature(), 0.001);
    assertEquals(0.7, config.getTemperature(), 0.001);

    assertFalse(config.isGreedy()); // isGreedy() returns true only when temp == 0.0
    assertFalse(config.isGreedy());
  }

  @Test
  @DisplayName("Builder supports greedy sampling (doSample=false)")
  void builder_greedySampling_succeeds() {
    GenerationConfig config = GenerationConfig.builder().maxOutputTokens(256).temperature(0.0f) // Temperature
                                                                                            // 0 for
                                                                                            // greedy
        .topK(1).topP(1.0f).build();

    assertTrue(config.isGreedy()); // isGreedy() returns true when temp == 0.0
    assertEquals(0.0, config.getTemperature(), 0.001);
  }

  @Test
  @DisplayName("Builder supports creative sampling (high temperature)")
  void builder_creativeSampling_succeeds() {
    GenerationConfig config = GenerationConfig.builder().maxOutputTokens(1024).temperature(1.5f) // Higher
                                                                                             // temperature
                                                                                             // for
                                                                                             // creativity
        .topK(80).topP(0.95f).build();

    assertFalse(config.isGreedy()); // isGreedy() returns true only when temp == 0.0
    assertEquals(1.5, config.getTemperature(), 0.001);
    assertEquals(80, config.getTopK());
    assertEquals(0.95, config.getTopP(), 0.001);
  }

  @Test
  @DisplayName("Builder supports nucleus sampling (topP-based)")
  void builder_nucleusSampling_succeeds() {
    GenerationConfig config =
        GenerationConfig.builder().maxOutputTokens(512).temperature(0.8f).topK(50).topP(0.92f) // Nucleus
                                                                                          // sampling
                                                                                          // with
                                                                                          // topP
            .build();

    assertFalse(config.isGreedy()); // isGreedy() returns true only when temp == 0.0
    assertEquals(0.92, config.getTopP(), 0.001);
  }
}
