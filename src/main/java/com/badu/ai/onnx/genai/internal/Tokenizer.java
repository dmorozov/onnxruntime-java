package com.badu.ai.onnx.genai.internal;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * The Tokenizer class is responsible for converting between text and token ids.
 * Uses HuggingFace tokenizers via DJL wrapper.
 */
public class Tokenizer implements AutoCloseable {

  private final HuggingFaceTokenizer tokenizer;

  /**
   * Creates a Tokenizer from the given model.
   *
   * @param model The model to use.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Tokenizer(Model model) throws GenAIException {
    try {
      // Look for tokenizer.json in model directory
      Path tokenizerPath = Paths.get(model.getModelDir(), "tokenizer.json");
      if (!Files.exists(tokenizerPath)) {
        throw new GenAIException("Tokenizer file not found: " + tokenizerPath);
      }
      this.tokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath);
    } catch (IOException e) {
      throw new GenAIException("Failed to load tokenizer from " + model.getModelDir(), e);
    }
  }

  /**
   * Encodes a string into a sequence of token ids.
   *
   * @param string Text to encode as token ids.
   * @return a Sequences object with a single sequence in it.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Sequences encode(String string) throws GenAIException {
    try {
      Encoding encoding = tokenizer.encode(string);
      long[] ids = encoding.getIds();

      // Convert long[] to int[]
      int[] tokenIds = new int[ids.length];
      for (int i = 0; i < ids.length; i++) {
        tokenIds[i] = (int) ids[i];
      }

      return new Sequences(tokenIds);
    } catch (Exception e) {
      throw new GenAIException("Failed to encode string: " + string, e);
    }
  }

  /**
   * Encodes an array of strings into a sequence of token ids for each input.
   *
   * @param strings Collection of strings to encode as token ids.
   * @return a Sequences object with one sequence per input string.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Sequences encodeBatch(String[] strings) throws GenAIException {
    try {
      Encoding[] encodings = tokenizer.batchEncode(List.of(strings));
      List<int[]> tokenSequences = new ArrayList<>();

      for (Encoding encoding : encodings) {
        long[] ids = encoding.getIds();
        int[] tokenIds = new int[ids.length];
        for (int i = 0; i < ids.length; i++) {
          tokenIds[i] = (int) ids[i];
        }
        tokenSequences.add(tokenIds);
      }

      return new Sequences(tokenSequences);
    } catch (Exception e) {
      throw new GenAIException("Failed to encode batch", e);
    }
  }

  /**
   * Decodes a sequence of token ids into text.
   *
   * @param sequence Collection of token ids to decode to text.
   * @return The text representation of the sequence.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public String decode(int[] sequence) throws GenAIException {
    try {
      // Convert int[] to long[]
      long[] ids = new long[sequence.length];
      for (int i = 0; i < sequence.length; i++) {
        ids[i] = sequence[i];
      }

      // Skip special tokens (like </s>, <pad>, etc.) for clean output
      return tokenizer.decode(ids, true);
    } catch (Exception e) {
      throw new GenAIException("Failed to decode sequence", e);
    }
  }

  /**
   * Decodes a batch of sequences of token ids into text.
   *
   * @param sequences A Sequences object with one or more sequences of token ids.
   * @return An array of strings with the text representation of each sequence.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public String[] decodeBatch(Sequences sequences) throws GenAIException {
    int numSequences = (int) sequences.numSequences();

    String[] result = new String[numSequences];
    for (int i = 0; i < numSequences; i++) {
      result[i] = decode(sequences.getSequence(i));
    }

    return result;
  }

  /**
   * Gets the beginning of sentence token ID.
   *
   * @return The BOS token ID.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int getBosTokenId() throws GenAIException {
    try {
      // Try to encode BOS token
      return (int) tokenizer.encode("<s>").getIds()[0];
    } catch (Exception e) {
      // Return common T5 BOS token ID (0)
      return 0;
    }
  }

  /**
   * Gets the padding token ID.
   *
   * @return The padding token ID.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int getPadTokenId() throws GenAIException {
    try {
      // Try to encode PAD token
      return (int) tokenizer.encode("<pad>").getIds()[0];
    } catch (Exception e) {
      // Return common T5 PAD token ID (0)
      return 0;
    }
  }

  /**
   * Gets the end of sentence token IDs.
   *
   * @return An array of EOS token IDs.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int[] getEosTokenIds() throws GenAIException {
    try {
      // Try to encode EOS token
      long[] eosIds = tokenizer.encode("</s>").getIds();
      int[] result = new int[eosIds.length];
      for (int i = 0; i < eosIds.length; i++) {
        result[i] = (int) eosIds[i];
      }
      return result;
    } catch (Exception e) {
      // Return common T5 EOS token ID (1)
      return new int[]{1};
    }
  }

  /**
   * Converts a string to a token ID.
   *
   * @param str The string to convert to a token ID.
   * @return The token ID for the given string.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public int toTokenId(String str) throws GenAIException {
    try {
      long[] ids = tokenizer.encode(str).getIds();
      return ids.length > 0 ? (int) ids[0] : -1;
    } catch (Exception e) {
      throw new GenAIException("Failed to convert string to token ID: " + str, e);
    }
  }

  /**
   * Applies a chat template to format messages.
   *
   * @param templateStr The template string to use.
   * @param messages The messages in JSON format.
   * @param tools The tools in JSON format (can be null).
   * @param addGenerationPrompt Whether to add generation prompt.
   * @return The formatted chat string.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public String applyChatTemplate(String templateStr, String messages, String tools,
      boolean addGenerationPrompt) throws GenAIException {
    // Basic implementation - would need Jinja2 template engine for full support
    throw new GenAIException("applyChatTemplate not yet implemented");
  }

  /**
   * Updates tokenizer options.
   *
   * @param options Map of option keys to values.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public void updateOptions(java.util.Map<String, String> options) throws GenAIException {
    // Options update not supported by HuggingFace tokenizers Java wrapper
    // This is typically done at tokenizer initialization
  }

  /**
   * Creates a TokenizerStream object for streaming tokenization. This is used with Generator class
   * to provide each token as it is generated.
   *
   * @return The new TokenizerStream instance.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public TokenizerStream createStream() throws GenAIException {
    return new TokenizerStream(this);
  }

  /**
   * Gets the underlying HuggingFace tokenizer.
   *
   * @return The HuggingFace tokenizer instance.
   */
  HuggingFaceTokenizer getHuggingFaceTokenizer() {
    return tokenizer;
  }

  @Override
  public void close() {
    if (tokenizer != null) {
      tokenizer.close();
    }
  }
}
