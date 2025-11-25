package com.badu.ai.onnx.streaming;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simple energy-based Voice Activity Detector (VAD).
 *
 * <p>Detects speech in audio by analyzing energy levels and zero-crossing rate.
 * This is a lightweight implementation suitable for real-time streaming.
 *
 * <p><strong>Algorithm:</strong>
 * <ol>
 *   <li>Compute short-term energy (RMS)</li>
 *   <li>Compute zero-crossing rate</li>
 *   <li>Apply thresholds to detect voice activity</li>
 *   <li>Use hysteresis to avoid flickering</li>
 * </ol>
 *
 * <p><strong>Usage Example:</strong>
 * <pre>{@code
 * VoiceActivityDetector vad = new VoiceActivityDetector(
 *     16000,  // Sample rate
 *     0.02,   // Energy threshold
 *     0.3     // ZCR threshold
 * );
 *
 * while (audioAvailable) {
 *     float[] samples = captureAudio();
 *     boolean isVoice = vad.isVoiceActive(samples);
 *
 *     if (isVoice) {
 *         // Process as speech
 *     } else {
 *         // Skip silence
 *     }
 * }
 * }</pre>
 *
 * <p><strong>Limitations:</strong>
 * <ul>
 *   <li>Energy-based detection (may be fooled by loud background noise)</li>
 *   <li>No machine learning model (less accurate than ML-based VAD)</li>
 *   <li>Fixed thresholds (not adaptive to environment)</li>
 * </ul>
 *
 * <p>For production use, consider more advanced VAD like:
 * <ul>
 *   <li>WebRTC VAD</li>
 *   <li>Silero VAD (ONNX-based)</li>
 *   <li>PyAnnote VAD</li>
 * </ul>
 */
public class VoiceActivityDetector {

  private static final Logger logger = LoggerFactory.getLogger(VoiceActivityDetector.class);

  private final int sampleRate;
  private final double energyThreshold;
  private final double zcrThreshold;
  private final int frameSize;

  private boolean currentlyActive;
  private double runningEnergyAverage;
  private int activeFrames;
  private int inactiveFrames;

  // Hysteresis parameters to avoid flickering
  private static final int MIN_ACTIVE_FRAMES = 3;   // Need 3 consecutive active frames
  private static final int MIN_INACTIVE_FRAMES = 5; // Need 5 consecutive inactive frames

  /**
   * Creates a VAD with default parameters optimized for speech.
   *
   * <p>Default: 16kHz sample rate, 0.02 energy threshold, 0.3 ZCR threshold
   */
  public VoiceActivityDetector() {
    this(16000, 0.02, 0.3);
  }

  /**
   * Creates a VAD with custom parameters.
   *
   * @param sampleRate Audio sample rate in Hz
   * @param energyThreshold Energy threshold (0.0-1.0, typical: 0.01-0.05)
   * @param zcrThreshold Zero-crossing rate threshold (0.0-1.0, typical: 0.2-0.4)
   */
  public VoiceActivityDetector(int sampleRate, double energyThreshold, double zcrThreshold) {
    if (sampleRate <= 0) {
      throw new IllegalArgumentException("Sample rate must be positive");
    }
    if (energyThreshold < 0 || energyThreshold > 1) {
      throw new IllegalArgumentException("Energy threshold must be in [0, 1]");
    }
    if (zcrThreshold < 0 || zcrThreshold > 1) {
      throw new IllegalArgumentException("ZCR threshold must be in [0, 1]");
    }

    this.sampleRate = sampleRate;
    this.energyThreshold = energyThreshold;
    this.zcrThreshold = zcrThreshold;
    this.frameSize = sampleRate / 100;  // 10ms frames

    this.currentlyActive = false;
    this.runningEnergyAverage = 0.0;
    this.activeFrames = 0;
    this.inactiveFrames = 0;

    logger.debug("VAD initialized: sampleRate={}, energyThreshold={}, zcrThreshold={}",
        sampleRate, energyThreshold, zcrThreshold);
  }

  /**
   * Detects voice activity in the given audio samples.
   *
   * @param samples Audio samples (mono, normalized to [-1.0, 1.0])
   * @return true if voice is detected, false otherwise
   */
  public boolean isVoiceActive(float[] samples) {
    if (samples == null || samples.length == 0) {
      return currentlyActive;
    }

    // Compute features
    double energy = computeEnergy(samples);
    double zcr = computeZeroCrossingRate(samples);

    // Update running average (for adaptive threshold)
    runningEnergyAverage = 0.95 * runningEnergyAverage + 0.05 * energy;

    // Determine if frame is active
    boolean frameActive = energy > energyThreshold && zcr > zcrThreshold;

    // Apply hysteresis
    if (frameActive) {
      activeFrames++;
      inactiveFrames = 0;

      if (!currentlyActive && activeFrames >= MIN_ACTIVE_FRAMES) {
        currentlyActive = true;
        logger.debug("Voice activity START (energy={:.4f}, zcr={:.4f})", energy, zcr);
      }
    } else {
      inactiveFrames++;
      activeFrames = 0;

      if (currentlyActive && inactiveFrames >= MIN_INACTIVE_FRAMES) {
        currentlyActive = false;
        logger.debug("Voice activity STOP (energy={:.4f}, zcr={:.4f})", energy, zcr);
      }
    }

    return currentlyActive;
  }

  /**
   * Computes the short-term energy (RMS) of the audio samples.
   *
   * @param samples Audio samples
   * @return Energy value (0.0 to ~1.0)
   */
  private double computeEnergy(float[] samples) {
    double sumSquares = 0.0;

    for (float sample : samples) {
      sumSquares += sample * sample;
    }

    return Math.sqrt(sumSquares / samples.length);
  }

  /**
   * Computes the zero-crossing rate of the audio samples.
   *
   * <p>ZCR measures how often the signal crosses zero amplitude.
   * Speech typically has higher ZCR than noise.
   *
   * @param samples Audio samples
   * @return Zero-crossing rate (0.0 to 1.0)
   */
  private double computeZeroCrossingRate(float[] samples) {
    int zeroCrossings = 0;

    for (int i = 1; i < samples.length; i++) {
      if ((samples[i - 1] >= 0 && samples[i] < 0) ||
          (samples[i - 1] < 0 && samples[i] >= 0)) {
        zeroCrossings++;
      }
    }

    return (double) zeroCrossings / (samples.length - 1);
  }

  /**
   * Resets the VAD state.
   */
  public void reset() {
    currentlyActive = false;
    runningEnergyAverage = 0.0;
    activeFrames = 0;
    inactiveFrames = 0;
    logger.debug("VAD reset");
  }

  /**
   * Gets the current voice activity state.
   *
   * @return true if voice is currently active, false otherwise
   */
  public boolean isCurrentlyActive() {
    return currentlyActive;
  }

  /**
   * Gets the running energy average.
   *
   * @return Energy average (for adaptive thresholding)
   */
  public double getRunningEnergyAverage() {
    return runningEnergyAverage;
  }

  /**
   * Gets the configured energy threshold.
   *
   * @return Energy threshold
   */
  public double getEnergyThreshold() {
    return energyThreshold;
  }

  /**
   * Gets the configured ZCR threshold.
   *
   * @return ZCR threshold
   */
  public double getZcrThreshold() {
    return zcrThreshold;
  }
}
