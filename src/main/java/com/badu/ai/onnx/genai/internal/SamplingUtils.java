package com.badu.ai.onnx.genai.internal;

import java.util.*;

/**
 * Utility methods for token sampling and decoding strategies.
 *
 * <p>Provides implementations for:
 * <ul>
 *   <li>Greedy decoding (argmax)</li>
 *   <li>Temperature scaling</li>
 *   <li>Top-K sampling</li>
 *   <li>Top-P (nucleus) sampling</li>
 *   <li>Repetition penalty</li>
 * </ul>
 *
 * <p>All methods are stateless and thread-safe.
 *
 * @see com.badu.ai.onnx.genai.GenerationConfig
 */
public class SamplingUtils {

  /**
   * Greedy decoding - selects token with highest probability (argmax).
   *
   * <p>Used when temperature = 0.0 for deterministic output.
   *
   * <p><b>Special Token Handling:</b> To work around broken INT8 quantized models
   * where special tokens (pad, eos, unk) get artificially high logits, we mask
   * special tokens (IDs 0-3) by setting their logits to -Infinity. This prevents
   * them from being selected during greedy decoding.
   *
   * @param logits Logits array from model output [vocab_size]
   * @return Index of token with highest logit (excluding masked special tokens)
   */
  public static long argmax(float[] logits) {
    if (logits == null || logits.length == 0) {
      throw new IllegalArgumentException("Logits cannot be null or empty");
    }

    // Mask special tokens to prevent broken INT8 models from selecting them
    // T5 special tokens: 0=<pad>, 1=</s>, 2=<unk>, 3=<unk> variant
    // Create a masked copy to preserve original logits
    float[] maskedLogits = logits.clone();
    final int NUM_SPECIAL_TOKENS = 4;
    for (int i = 0; i < Math.min(NUM_SPECIAL_TOKENS, maskedLogits.length); i++) {
      maskedLogits[i] = Float.NEGATIVE_INFINITY;
    }

    int maxIndex = NUM_SPECIAL_TOKENS; // Start search after special tokens
    float maxValue = maskedLogits[maxIndex];

    for (int i = maxIndex + 1; i < maskedLogits.length; i++) {
      if (maskedLogits[i] > maxValue) {
        maxValue = maskedLogits[i];
        maxIndex = i;
      }
    }

    return maxIndex;
  }

  /**
   * Applies temperature scaling to logits.
   *
   * <p>Temperature controls randomness:
   * <ul>
   *   <li>temperature = 0.0: Greedy (use argmax instead)</li>
   *   <li>temperature = 1.0: No change (standard softmax)</li>
   *   <li>temperature &gt; 1.0: More random (flattens distribution)</li>
   *   <li>temperature &lt; 1.0: More confident (sharpens distribution)</li>
   * </ul>
   *
   * @param logits Logits array [vocab_size]
   * @param temperature Temperature value (0.0 to 2.0)
   * @return Temperature-scaled logits (modifies in place)
   */
  public static float[] applyTemperature(float[] logits, float temperature) {
    if (temperature <= 0.0f) {
      throw new IllegalArgumentException("Temperature must be > 0.0 (use argmax for greedy)");
    }

    if (temperature == 1.0f) {
      return logits; // No scaling needed
    }

    for (int i = 0; i < logits.length; i++) {
      logits[i] = logits[i] / temperature;
    }

    return logits;
  }

  /**
   * Applies repetition penalty to discourage repeated tokens.
   *
   * <p>For each token in generatedTokens, divides its logit by penalty:
   * <ul>
   *   <li>penalty = 1.0: No penalty</li>
   *   <li>penalty &gt; 1.0: Penalize repetition (recommended: 1.1-1.5)</li>
   *   <li>penalty &lt; 1.0: Encourage repetition (unusual)</li>
   * </ul>
   *
   * @param logits Logits array [vocab_size]
   * @param generatedTokens Previously generated token IDs
   * @param penalty Repetition penalty (1.0 to 2.0)
   * @return Penalized logits (modifies in place)
   */
  public static float[] applyRepetitionPenalty(float[] logits, List<Long> generatedTokens,
                                                float penalty) {
    if (penalty == 1.0f || generatedTokens.isEmpty()) {
      return logits; // No penalty
    }

    // Penalize each token that appears in generated sequence
    for (long tokenId : generatedTokens) {
      int idx = (int) tokenId;
      if (idx >= 0 && idx < logits.length) {
        logits[idx] = logits[idx] / penalty;
      }
    }

    return logits;
  }

  /**
   * Converts logits to probabilities using softmax.
   *
   * <p>Softmax: p(i) = exp(logit[i]) / sum(exp(logit[j]))
   *
   * @param logits Logits array [vocab_size]
   * @return Probability distribution [vocab_size] (sums to 1.0)
   */
  public static float[] softmax(float[] logits) {
    if (logits == null || logits.length == 0) {
      throw new IllegalArgumentException("Logits cannot be null or empty");
    }

    // Find max for numerical stability (prevents overflow)
    float maxLogit = logits[0];
    for (int i = 1; i < logits.length; i++) {
      if (logits[i] > maxLogit) {
        maxLogit = logits[i];
      }
    }

    // Compute exp(logit - max) and sum
    float[] probs = new float[logits.length];
    float sum = 0.0f;

    for (int i = 0; i < logits.length; i++) {
      probs[i] = (float) Math.exp(logits[i] - maxLogit);
      sum += probs[i];
    }

    // Normalize to sum to 1.0
    for (int i = 0; i < probs.length; i++) {
      probs[i] = probs[i] / sum;
    }

    return probs;
  }

  /**
   * Top-K sampling: samples from top-K most probable tokens.
   *
   * <p>Algorithm:
   * <ol>
   *   <li>Sort logits in descending order</li>
   *   <li>Keep only top-K logits, set rest to -Infinity</li>
   *   <li>Apply softmax to get probabilities</li>
   *   <li>Sample from the distribution</li>
   * </ol>
   *
   * @param logits Logits array [vocab_size]
   * @param k Number of top tokens to consider (1 to vocab_size)
   * @param rng Random number generator
   * @return Sampled token ID
   */
  public static long sampleTopK(float[] logits, int k, Random rng) {
    if (k <= 0) {
      throw new IllegalArgumentException("k must be > 0");
    }

    if (k >= logits.length) {
      // No filtering needed, sample from full distribution
      return sampleFromDistribution(logits, rng);
    }

    // Create index-value pairs for sorting
    List<IndexValue> indexedLogits = new ArrayList<>(logits.length);
    for (int i = 0; i < logits.length; i++) {
      indexedLogits.add(new IndexValue(i, logits[i]));
    }

    // Sort by value descending
    indexedLogits.sort((a, b) -> Float.compare(b.value, a.value));

    // Keep only top-K, set rest to -Infinity
    float[] filteredLogits = new float[logits.length];
    Arrays.fill(filteredLogits, Float.NEGATIVE_INFINITY);

    for (int i = 0; i < k; i++) {
      IndexValue iv = indexedLogits.get(i);
      filteredLogits[iv.index] = iv.value;
    }

    return sampleFromDistribution(filteredLogits, rng);
  }

  /**
   * Top-P (nucleus) sampling: samples from smallest set of tokens with cumulative probability ≥ p.
   *
   * <p>Algorithm:
   * <ol>
   *   <li>Sort logits in descending order</li>
   *   <li>Apply softmax to get probabilities</li>
   *   <li>Find smallest set where cumulative prob ≥ p</li>
   *   <li>Sample from this set</li>
   * </ol>
   *
   * @param logits Logits array [vocab_size]
   * @param p Cumulative probability threshold (0.0 to 1.0)
   * @param rng Random number generator
   * @return Sampled token ID
   */
  public static long sampleTopP(float[] logits, float p, Random rng) {
    if (p <= 0.0f || p > 1.0f) {
      throw new IllegalArgumentException("p must be in (0.0, 1.0]");
    }

    // Convert to probabilities
    float[] probs = softmax(logits);

    // Create index-probability pairs for sorting
    List<IndexValue> indexedProbs = new ArrayList<>(probs.length);
    for (int i = 0; i < probs.length; i++) {
      indexedProbs.add(new IndexValue(i, probs[i]));
    }

    // Sort by probability descending
    indexedProbs.sort((a, b) -> Float.compare(b.value, a.value));

    // Find nucleus: smallest set with cumulative prob ≥ p
    float cumulativeProb = 0.0f;
    float[] filteredLogits = new float[logits.length];
    Arrays.fill(filteredLogits, Float.NEGATIVE_INFINITY);

    for (IndexValue iv : indexedProbs) {
      filteredLogits[iv.index] = logits[iv.index];
      cumulativeProb += iv.value;

      if (cumulativeProb >= p) {
        break;
      }
    }

    return sampleFromDistribution(filteredLogits, rng);
  }

  /**
   * Samples a token from a probability distribution.
   *
   * <p>Uses categorical sampling: draws from multinomial distribution.
   *
   * @param logits Logits array (may contain -Infinity for filtered tokens)
   * @param rng Random number generator
   * @return Sampled token ID
   */
  private static long sampleFromDistribution(float[] logits, Random rng) {
    // Convert to probabilities
    float[] probs = softmax(logits);

    // Categorical sampling
    float randomValue = rng.nextFloat();
    float cumulativeProb = 0.0f;

    for (int i = 0; i < probs.length; i++) {
      cumulativeProb += probs[i];
      if (randomValue <= cumulativeProb) {
        return i;
      }
    }

    // Fallback (should never reach here due to floating point precision)
    return argmax(logits);
  }

  /**
   * Helper class for sorting logits/probabilities by value while preserving index.
   */
  private static class IndexValue {
    final int index;
    final float value;

    IndexValue(int index, float value) {
      this.index = index;
      this.value = value;
    }
  }
}
