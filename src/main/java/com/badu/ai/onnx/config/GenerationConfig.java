package com.badu.ai.onnx.config;

import lombok.Builder;
import lombok.Value;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Immutable configuration for decoding strategy (greedy/sampling).
 *
 * <p>Controls sampling behavior, token limits, and stopping criteria for text generation.
 *
 * <p>Decoding logic:
 * <ul>
 *   <li>If temperature == 0.0: Use greedy decoding (argmax)</li>
 *   <li>Else if topK > 0: Apply temperature → sample from top-K tokens</li>
 *   <li>Else if topP > 0: Apply temperature → nucleus sampling (cumulative prob ≥ topP)</li>
 *   <li>Else: Apply temperature → sample from full distribution</li>
 * </ul>
 *
 * <p>Builder pattern usage:
 * <pre>{@code
 * GenerationConfig config = GenerationConfig.builder()
 *     .temperature(0.0f)  // Greedy decoding
 *     .maxOutputTokens(256)
 *     .build();
 * }</pre>
 */
@Value
@Builder
public class GenerationConfig {

  /**
   * Default generation configuration with balanced settings.
   * temperature: 0.7, topK: 50, topP: 0.9, repetitionPenalty: 1.0, maxOutputTokens: 512
   */
  public static final GenerationConfig DEFAULT = builder().build();

  /**
   * Sampling temperature (0.0 = greedy).
   * Range: [0.0, 2.0]
   * Default: 0.7 (balanced creativity/consistency)
   */
  @Builder.Default
  float temperature = 0.7f;

  /**
   * Top-K sampling.
   * Range: [1, 100]
   * Default: 50
   */
  @Builder.Default
  int topK = 50;

  /**
   * Nucleus sampling threshold (0.0 = disabled).
   * Range: [0.0, 1.0]
   * Default: 0.9
   */
  @Builder.Default
  float topP = 0.9f;

  /**
   * Penalty for repeated tokens (1.0 = no penalty).
   * Range: [1.0, 2.0]
   * Default: 1.0
   */
  @Builder.Default
  float repetitionPenalty = 1.0f;

  /**
   * Maximum tokens to generate.
   * Range: [1, 4096]
   * Default: 512
   */
  @Builder.Default
  int maxOutputTokens = 512;

  /**
   * Minimum tokens to generate (prevents premature EOS).
   * Range: [0, maxOutputTokens]
   * Default: 0 (disabled)
   *
   * <p>When set > 0, the EOS token is masked (probability set to -inf) until
   * at least minOutputTokens have been generated. This prevents one-word
   * summaries and ensures sufficient detail in generated text.
   *
   * <p><b>Note:</b> For encoder-decoder models (T5, BART), counts only generated tokens.
   * For decoder-only models (Llama, Phi-3), counts total tokens including prompt.
   *
   * <p>Use cases:
   * <ul>
   *   <li>Prevent one-word summaries</li>
   *   <li>Ensure minimum response length</li>
   *   <li>Quality control for production outputs</li>
   * </ul>
   */
  @Builder.Default
  int minOutputTokens = 0;

  /**
   * Bad words filter (prevents specific token sequences).
   * Default: Empty list (no filtering)
   *
   * <p>List of token ID sequences that should not be generated. During generation,
   * if the current token sequence matches a bad word prefix, the next token is
   * masked (probability set to -inf) to prevent completion.
   *
   * <p><b>Multi-Token Handling:</b>
   * The filter supports both single-token and multi-token bad words. For example,
   * if "feck" tokenizes to [12345, 67890], both tokens must be tracked:
   * <ul>
   *   <li>When token 12345 is generated, check if next token is 67890</li>
   *   <li>If yes, mask 67890 to prevent "feck"</li>
   * </ul>
   *
   * <p><b>Usage Example:</b>
   * <pre>{@code
   * // Using tokenizer to get token IDs
   * T5Tokenizer tokenizer = new T5Tokenizer(...);
   * GenerationConfig config = GenerationConfig.builder()
   *     .addBadWord("profanity", tokenizer)
   *     .addBadWord("competitor", tokenizer)
   *     .build();
   * }</pre>
   *
   * <p>Use cases:
   * <ul>
   *   <li>Content safety (filter profanity)</li>
   *   <li>Brand compliance (prevent competitor mentions)</li>
   *   <li>Domain-specific filtering (medical, legal)</li>
   * </ul>
   */
  @Builder.Default
  List<List<Long>> badWordsIds = Collections.emptyList();

  /**
   * Number of beams for beam search (1 = greedy/sampling).
   * Range: [1, 16]
   * Default: 1 (disabled - use greedy or sampling)
   *
   * <p>Beam search maintains K candidate sequences and explores multiple paths
   * simultaneously. Higher beam width typically produces higher quality outputs
   * but increases computational cost linearly.
   *
   * <p><b>Performance Impact:</b>
   * <ul>
   *   <li>numBeams=1: Standard greedy/sampling (fastest)</li>
   *   <li>numBeams=4: 4x slower, typically better quality</li>
   *   <li>numBeams=8: 8x slower, diminishing returns</li>
   * </ul>
   *
   * <p><b>Usage Example:</b>
   * <pre>{@code
   * GenerationConfig config = GenerationConfig.builder()
   *     .numBeams(4)
   *     .lengthPenalty(0.8f)
   *     .build();
   * }</pre>
   */
  @Builder.Default
  int numBeams = 1;

  /**
   * Length penalty for beam search scoring.
   * Range: [0.0, 2.0]
   * Default: 1.0 (no penalty)
   *
   * <p>Adjusts beam scores based on sequence length to avoid bias toward
   * shorter or longer sequences:
   * <ul>
   *   <li>&lt; 1.0: Favors shorter sequences</li>
   *   <li>= 1.0: No length bias</li>
   *   <li>&gt; 1.0: Favors longer sequences</li>
   * </ul>
   *
   * <p><b>Formula:</b> score = log_prob / (length ^ lengthPenalty)
   */
  @Builder.Default
  float lengthPenalty = 1.0f;

  /**
   * Enable early stopping for beam search.
   * Default: true
   *
   * <p>When enabled, beam search stops when all beams end with EOS token.
   * When disabled, continues until maxOutputTokens is reached.
   */
  @Builder.Default
  boolean earlyStoppingBeamSearch = true;

  /**
   * Number of sequences to return from beam search.
   * Range: [1, numBeams]
   * Default: 1 (return only best sequence)
   *
   * <p>Returns the top-N scoring sequences from beam search. Useful for
   * generating multiple candidate outputs.
   */
  @Builder.Default
  int numReturnSequences = 1;

  /**
   * Custom builder with validation logic.
   */
  public static class GenerationConfigBuilder {
    /**
     * Builds the GenerationConfig with validation.
     *
     * @return validated GenerationConfig instance
     * @throws IllegalStateException if validation fails
     */
    public GenerationConfig build() {
      // Apply defaults if not set
      if (!this.temperature$set) {
        this.temperature$value = 0.7f;
        this.temperature$set = true;
      }
      if (!this.topK$set) {
        this.topK$value = 50;
        this.topK$set = true;
      }
      if (!this.topP$set) {
        this.topP$value = 0.9f;
        this.topP$set = true;
      }
      if (!this.repetitionPenalty$set) {
        this.repetitionPenalty$value = 1.0f;
        this.repetitionPenalty$set = true;
      }
      if (!this.maxOutputTokens$set) {
        this.maxOutputTokens$value = 512;
        this.maxOutputTokens$set = true;
      }
      if (!this.minOutputTokens$set) {
        this.minOutputTokens$value = 0;
        this.minOutputTokens$set = true;
      }
      if (!this.badWordsIds$set) {
        this.badWordsIds$value = Collections.emptyList();
        this.badWordsIds$set = true;
      }
      if (!this.numBeams$set) {
        this.numBeams$value = 1;
        this.numBeams$set = true;
      }
      if (!this.lengthPenalty$set) {
        this.lengthPenalty$value = 1.0f;
        this.lengthPenalty$set = true;
      }
      if (!this.earlyStoppingBeamSearch$set) {
        this.earlyStoppingBeamSearch$value = true;
        this.earlyStoppingBeamSearch$set = true;
      }
      if (!this.numReturnSequences$set) {
        this.numReturnSequences$value = 1;
        this.numReturnSequences$set = true;
      }

      // Validate ranges
      if (this.temperature$value < 0.0f || this.temperature$value > 2.0f) {
        throw new IllegalStateException(
            "temperature must be in range [0.0, 2.0], got: " + this.temperature$value);
      }

      if (this.topK$value < 1 || this.topK$value > 100) {
        throw new IllegalStateException(
            "topK must be in range [1, 100], got: " + this.topK$value);
      }

      if (this.topP$value < 0.0f || this.topP$value > 1.0f) {
        throw new IllegalStateException(
            "topP must be in range [0.0, 1.0], got: " + this.topP$value);
      }

      if (this.repetitionPenalty$value < 1.0f || this.repetitionPenalty$value > 2.0f) {
        throw new IllegalStateException(
            "repetitionPenalty must be in range [1.0, 2.0], got: " + this.repetitionPenalty$value);
      }

      if (this.maxOutputTokens$value <= 0) {
        throw new IllegalStateException(
            "maxOutputTokens must be positive, got: " + this.maxOutputTokens$value);
      }

      if (this.maxOutputTokens$value > 4096) {
        throw new IllegalStateException(
            "maxOutputTokens exceeds maximum 4096, got: " + this.maxOutputTokens$value);
      }

      if (this.minOutputTokens$value < 0) {
        throw new IllegalStateException(
            "minOutputTokens must be >= 0, got: " + this.minOutputTokens$value);
      }

      if (this.minOutputTokens$value > this.maxOutputTokens$value) {
        throw new IllegalStateException(
            "minOutputTokens (" + this.minOutputTokens$value + ") " +
            "must be <= maxOutputTokens (" + this.maxOutputTokens$value + ")");
      }

      if (this.badWordsIds$value == null) {
        throw new IllegalStateException("badWordsIds cannot be null (use empty list instead)");
      }

      // Validate beam search parameters
      if (this.numBeams$value < 1 || this.numBeams$value > 16) {
        throw new IllegalStateException(
            "numBeams must be in range [1, 16], got: " + this.numBeams$value);
      }

      if (this.lengthPenalty$value < 0.0f || this.lengthPenalty$value > 2.0f) {
        throw new IllegalStateException(
            "lengthPenalty must be in range [0.0, 2.0], got: " + this.lengthPenalty$value);
      }

      if (this.numReturnSequences$value < 1 || this.numReturnSequences$value > this.numBeams$value) {
        throw new IllegalStateException(
            "numReturnSequences must be in range [1, " + this.numBeams$value + "], got: " +
            this.numReturnSequences$value);
      }

      // Make badWordsIds immutable
      List<List<Long>> immutableBadWords = new ArrayList<>();
      for (List<Long> badWord : this.badWordsIds$value) {
        if (badWord == null || badWord.isEmpty()) {
          throw new IllegalStateException("Bad word sequences cannot be null or empty");
        }
        immutableBadWords.add(Collections.unmodifiableList(new ArrayList<>(badWord)));
      }

      return new GenerationConfig(this.temperature$value, this.topK$value, this.topP$value,
          this.repetitionPenalty$value, this.maxOutputTokens$value, this.minOutputTokens$value,
          Collections.unmodifiableList(immutableBadWords), this.numBeams$value,
          this.lengthPenalty$value, this.earlyStoppingBeamSearch$value,
          this.numReturnSequences$value);
    }
  }

  /**
   * Checks if this configuration uses greedy decoding.
   *
   * @return true if temperature is 0.0 (greedy)
   */
  public boolean isGreedy() {
    return temperature == 0.0f;
  }

  /**
   * Checks if top-K sampling is enabled.
   *
   * @return true if topK > 0
   */
  public boolean isTopKEnabled() {
    return topK > 0;
  }

  /**
   * Checks if nucleus (top-P) sampling is enabled.
   *
   * @return true if topP > 0.0
   */
  public boolean isTopPEnabled() {
    return topP > 0.0f;
  }

  /**
   * Checks if repetition penalty is applied.
   *
   * @return true if repetitionPenalty > 1.0
   */
  public boolean hasRepetitionPenalty() {
    return repetitionPenalty > 1.0f;
  }

  /**
   * Checks if minimum length enforcement is enabled.
   *
   * @return true if minOutputTokens > 0
   */
  public boolean hasMinLength() {
    return minOutputTokens > 0;
  }

  /**
   * Checks if bad words filtering is enabled.
   *
   * @return true if badWordsIds is not empty
   */
  public boolean hasBadWords() {
    return badWordsIds != null && !badWordsIds.isEmpty();
  }

  /**
   * Checks if beam search is enabled.
   *
   * @return true if numBeams > 1
   */
  public boolean isBeamSearchEnabled() {
    return numBeams > 1;
  }
}
