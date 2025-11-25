package com.badu.ai.onnx.tokenization;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Whisper tokenizer wrapper using DJL HuggingFace Tokenizers.
 *
 * <p>Handles tokenization and decoding for Whisper (Speech-to-Text) models.
 * Uses the HuggingFace tokenizer.json format for compatibility with byte-level BPE.
 *
 * <p>Whisper uses a unique set of special tokens:
 * <ul>
 *   <li><code>&lt;|startoftranscript|&gt;</code> - Marks beginning of transcription</li>
 *   <li><code>&lt;|en|&gt;</code>, <code>&lt;|zh|&gt;</code>, etc. - Language tokens</li>
 *   <li><code>&lt;|translate|&gt;</code> - Translation task token</li>
 *   <li><code>&lt;|transcribe|&gt;</code> - Transcription task token</li>
 *   <li><code>&lt;|notimestamps|&gt;</code> - No timestamp token</li>
 *   <li><code>&lt;|endoftext|&gt;</code> - Used for BOS, EOS, PAD, and UNK</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>{@code
 * try (WhisperTokenizer tokenizer = new WhisperTokenizer(
 *         Paths.get("models/whisper-tiny.en-ONNX/tokenizer.json"))) {
 *
 *     // Prepare decoder input tokens (for forced decoding)
 *     long[] decoderInput = tokenizer.createDecoderInputTokens(
 *         Language.ENGLISH, Task.TRANSCRIBE, false);
 *
 *     // Decode model output
 *     String transcription = tokenizer.decode(outputTokens);
 * }
 * }</pre>
 *
 * @see HuggingFaceTokenizer
 */
public class WhisperTokenizer implements AutoCloseable {

  private static final Logger logger = LoggerFactory.getLogger(WhisperTokenizer.class);

  private final HuggingFaceTokenizer tokenizer;
  private final Path tokenizerPath;

  // Whisper special tokens
  public static final String TOKEN_START_OF_TRANSCRIPT = "<|startoftranscript|>";
  public static final String TOKEN_END_OF_TEXT = "<|endoftext|>";
  public static final String TOKEN_TRANSLATE = "<|translate|>";
  public static final String TOKEN_TRANSCRIBE = "<|transcribe|>";
  public static final String TOKEN_NO_TIMESTAMPS = "<|notimestamps|>";
  public static final String TOKEN_START_OF_LM = "<|startoflm|>";
  public static final String TOKEN_NO_SPEECH = "<|nocaptions|>";

  // Language tokens
  public enum Language {
    ENGLISH("en"),
    SPANISH("es"),
    FRENCH("fr"),
    GERMAN("de"),
    ITALIAN("it"),
    PORTUGUESE("pt"),
    CHINESE("zh"),
    JAPANESE("ja"),
    KOREAN("ko"),
    RUSSIAN("ru");

    private final String code;

    Language(String code) {
      this.code = code;
    }

    public String getToken() {
      return "<|" + code + "|>";
    }
  }

  // Task tokens
  public enum Task {
    TRANSCRIBE(TOKEN_TRANSCRIBE),
    TRANSLATE(TOKEN_TRANSLATE);

    private final String token;

    Task(String token) {
      this.token = token;
    }

    public String getToken() {
      return token;
    }
  }

  /**
   * Creates a WhisperTokenizer from a tokenizer.json file.
   *
   * @param tokenizerPath Path to tokenizer.json file
   * @throws IOException if tokenizer file cannot be loaded
   */
  public WhisperTokenizer(Path tokenizerPath) throws IOException {
    this.tokenizerPath = tokenizerPath;

    logger.debug("Loading Whisper tokenizer from: {}", tokenizerPath);

    try {
      this.tokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath);
      logger.debug("WhisperTokenizer loaded successfully from: {}", tokenizerPath);
    } catch (IOException e) {
      logger.error("Failed to load tokenizer from: {}", tokenizerPath, e);
      throw new IOException("Failed to load Whisper tokenizer from: " + tokenizerPath, e);
    }
  }

  /**
   * Creates decoder input tokens for Whisper model.
   *
   * <p>Whisper requires specific prefix tokens for the decoder:
   * <pre>
   * [&lt;|startoftranscript|&gt;, &lt;|en|&gt;, &lt;|transcribe|&gt;, &lt;|notimestamps|&gt;]
   * </pre>
   *
   * @param language Language to transcribe (e.g., ENGLISH)
   * @param task Task to perform (TRANSCRIBE or TRANSLATE)
   * @param includeTimestamps Whether to include timestamps in output
   * @return Array of token IDs for decoder input
   */
  public long[] createDecoderInputTokens(Language language, Task task,
      boolean includeTimestamps) {

    logger.debug("Creating decoder input tokens: language={}, task={}, timestamps={}",
        language, task, includeTimestamps);

    // Build prefix sequence for Whisper decoder
    //
    // NOTE: For English-only models (whisper-tiny.en, whisper-base.en, etc.),
    // the generation_config.json specifies:
    //   - decoder_start_token_id: 50257 (<|startoftranscript|>)
    //   - forced_decoder_ids: [[1, 50362]] (forces <|notimestamps|> at position 1)
    //   - suppress_tokens: includes 50357, 50358 (<|translate|>, <|transcribe|>)
    //
    // So for English-only models, we only need:
    //   [<|startoftranscript|>, <|notimestamps|>]
    //
    // For multilingual models, we would include language and task tokens.
    //
    // TODO: Detect model type (English-only vs multilingual) from config

    java.util.List<Long> tokenIdsList = new java.util.ArrayList<>();

    // Start of transcript token (always first)
    tokenIdsList.add(getSpecialTokenId(TOKEN_START_OF_TRANSCRIPT));

    // For now, assume English-only model (based on generation_config.json)
    // Just add notimestamps token if requested
    if (!includeTimestamps) {
      tokenIdsList.add(getSpecialTokenId(TOKEN_NO_TIMESTAMPS));
    }

    // Convert to array
    long[] tokenIds = new long[tokenIdsList.size()];
    for (int i = 0; i < tokenIdsList.size(); i++) {
      tokenIds[i] = tokenIdsList.get(i);
    }

    logger.info("Decoder input token IDs: {}", java.util.Arrays.toString(tokenIds));

    return tokenIds;
  }

  /**
   * Encodes text to token IDs (for testing/validation only).
   *
   * <p>Note: For Whisper inference, you typically don't encode text input.
   * Instead, you provide audio features (mel spectrogram) to the encoder.
   * This method is mainly for tokenizer testing and validation.
   *
   * @param text Input text to encode
   * @return Array of token IDs
   * @throws IllegalArgumentException if text is null or empty
   */
  public long[] encode(String text) {
    if (text == null || text.isEmpty()) {
      throw new IllegalArgumentException("Text cannot be null or empty");
    }

    logger.trace("Encoding text: {}", text.substring(0, Math.min(100, text.length())));

    Encoding encoding = tokenizer.encode(text);
    long[] tokenIds = encoding.getIds();

    logger.trace("Encoded {} characters to {} tokens", text.length(), tokenIds.length);

    return tokenIds;
  }

  /**
   * Decodes token IDs back to text.
   *
   * <p>This is the primary method for converting Whisper model output
   * (token IDs) back to transcribed text.
   *
   * <p>The tokenizer automatically:
   * <ul>
   *   <li>Skips special tokens (SOT, language, task markers)</li>
   *   <li>Handles byte-level BPE decoding</li>
   *   <li>Reconstructs text with proper spacing</li>
   * </ul>
   *
   * @param tokenIds Array of token IDs to decode
   * @return Decoded transcription text
   * @throws IllegalArgumentException if tokenIds is null or empty
   */
  public String decode(long[] tokenIds) {
    return decode(tokenIds, true);
  }

  /**
   * Decodes token IDs with special token skipping control.
   *
   * @param tokenIds Array of token IDs to decode
   * @param skipSpecialTokens Whether to skip special tokens in output
   * @return Decoded text
   */
  public String decode(long[] tokenIds, boolean skipSpecialTokens) {
    if (tokenIds == null || tokenIds.length == 0) {
      throw new IllegalArgumentException("Token IDs cannot be null or empty");
    }

    logger.trace("Decoding {} tokens (skipSpecialTokens={})", tokenIds.length, skipSpecialTokens);

    String decoded = tokenizer.decode(tokenIds, skipSpecialTokens);

    logger.trace("Decoded {} tokens to {} characters", tokenIds.length, decoded.length());

    return decoded.trim();
  }

  /**
   * Gets a specific special token ID.
   *
   * <p>This method looks up special token IDs from Whisper's predefined vocabulary.
   * We hardcode these because encoding special token strings with BPE can result in
   * multiple tokens, giving incorrect IDs.
   *
   * @param token Special token string (e.g., "&lt;|startoftranscript|&gt;")
   * @return Token ID, or -1 if unknown
   */
  public long getSpecialTokenId(String token) {
    // Whisper special tokens (from tokenizer.json added_tokens section)
    switch (token) {
      case "<|endoftext|>": return 50256;
      case "<|startoftranscript|>": return 50257;
      case "<|en|>": return 50258;
      case "<|zh|>": return 50259;
      case "<|de|>": return 50260;
      case "<|es|>": return 50261;
      case "<|ru|>": return 50262;
      case "<|ko|>": return 50263;
      case "<|fr|>": return 50264;
      case "<|ja|>": return 50265;
      case "<|pt|>": return 50266;
      case "<|it|>": return 50267;
      case "<|translate|>": return 50357;
      case "<|transcribe|>": return 50358;
      case "<|notimestamps|>": return 50362;
      default:
        logger.warn("Unknown special token: {}", token);
        return -1;
    }
  }

  /**
   * Gets the end of text token ID.
   *
   * @return EOS token ID
   */
  public long getEosTokenId() {
    return getSpecialTokenId(TOKEN_END_OF_TEXT);
  }

  /**
   * Gets the start of transcript token ID.
   *
   * @return SOT token ID
   */
  public long getSotTokenId() {
    return getSpecialTokenId(TOKEN_START_OF_TRANSCRIPT);
  }

  /**
   * Gets the no timestamps token ID.
   *
   * @return No timestamps token ID
   */
  public long getNoTimestampsTokenId() {
    return getSpecialTokenId(TOKEN_NO_TIMESTAMPS);
  }

  /**
   * Gets the tokenizer instance (for advanced use).
   *
   * @return HuggingFaceTokenizer instance
   */
  public HuggingFaceTokenizer getTokenizer() {
    return tokenizer;
  }

  /**
   * Gets the tokenizer path.
   *
   * @return Path to tokenizer.json file
   */
  public Path getTokenizerPath() {
    return tokenizerPath;
  }

  /**
   * Closes the tokenizer and releases resources.
   */
  @Override
  public void close() {
    if (tokenizer != null) {
      tokenizer.close();
      logger.debug("WhisperTokenizer closed");
    }
  }
}
