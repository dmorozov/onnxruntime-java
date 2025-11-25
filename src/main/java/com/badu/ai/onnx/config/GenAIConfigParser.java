package com.badu.ai.onnx.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Parses Microsoft ONNX Runtime GenAI config (genai_config.json) to extract model parameters.
 *
 * <p>This parser reads GenAI-specific configuration files and extracts:
 * <ul>
 *   <li>KV-cache parameters (num_key_value_heads, head_size)</li>
 *   <li>Decoder configuration (hidden_size, num_hidden_layers)</li>
 *   <li>Input/output tensor names</li>
 *   <li>Search/generation defaults</li>
 * </ul>
 *
 * <p><b>File Structure:</b>
 * <pre>{@code
 * {
 *   "model": {
 *     "decoder": {
 *       "head_size": 64,
 *       "num_key_value_heads": 8,
 *       "num_attention_heads": 32,
 *       "num_hidden_layers": 16,
 *       "hidden_size": 2048
 *     }
 *   }
 * }
 * }</pre>
 *
 * <p><b>Usage Example:</b>
 * <pre>{@code
 * Path genaiPath = Paths.get("models/llama-3.2/genai_config.json");
 * GenAIConfigParser parser = new GenAIConfigParser(genaiPath);
 * int numKVHeads = parser.getNumKeyValueHeads();
 * int headSize = parser.getHeadSize();
 * }</pre>
 *
 * @see ModelConfigParser
 */
public class GenAIConfigParser {
    private static final Logger logger = LoggerFactory.getLogger(GenAIConfigParser.class);

    private final JsonNode configNode;
    private final Path configPath;

    /**
     * Creates a GenAIConfigParser from a genai_config.json file.
     *
     * @param configPath path to genai_config.json file
     * @throws IOException if file cannot be read or JSON is invalid
     */
    public GenAIConfigParser(Path configPath) throws IOException {
        if (!Files.exists(configPath)) {
            throw new IOException("GenAI config file does not exist: " + configPath);
        }

        this.configPath = configPath;
        ObjectMapper mapper = new ObjectMapper();
        String jsonContent = Files.readString(configPath);
        this.configNode = mapper.readTree(jsonContent);

        logger.debug("Parsed genai_config.json from: {}", configPath);
    }

    /**
     * Get number of key-value heads for grouped query attention.
     *
     * <p>Path: model.decoder.num_key_value_heads
     *
     * <p>Common values:
     * <ul>
     *   <li>Llama 3.2 1B: 8 (with 32 attention heads)</li>
     *   <li>Phi-3 Mini: varies by model size</li>
     * </ul>
     *
     * @return number of key-value heads, or null if not specified
     */
    public Integer getNumKeyValueHeads() {
        JsonNode decoderNode = configNode.path("model").path("decoder");
        if (decoderNode.has("num_key_value_heads")) {
            return decoderNode.path("num_key_value_heads").asInt();
        }
        return null;
    }

    /**
     * Get dimension per attention head.
     *
     * <p>Path: model.decoder.head_size
     *
     * <p>Common values:
     * <ul>
     *   <li>Llama 3.2 1B: 64</li>
     *   <li>Phi-3 Mini: 96</li>
     * </ul>
     *
     * @return head dimension, or null if not specified
     */
    public Integer getHeadSize() {
        JsonNode decoderNode = configNode.path("model").path("decoder");
        if (decoderNode.has("head_size")) {
            return decoderNode.path("head_size").asInt();
        }
        return null;
    }

    /**
     * Get number of attention heads.
     *
     * <p>Path: model.decoder.num_attention_heads
     *
     * @return number of attention heads, or null if not specified
     */
    public Integer getNumAttentionHeads() {
        JsonNode decoderNode = configNode.path("model").path("decoder");
        if (decoderNode.has("num_attention_heads")) {
            return decoderNode.path("num_attention_heads").asInt();
        }
        return null;
    }

    /**
     * Get number of hidden layers.
     *
     * <p>Path: model.decoder.num_hidden_layers
     *
     * @return number of hidden layers, or null if not specified
     */
    public Integer getNumHiddenLayers() {
        JsonNode decoderNode = configNode.path("model").path("decoder");
        if (decoderNode.has("num_hidden_layers")) {
            return decoderNode.path("num_hidden_layers").asInt();
        }
        return null;
    }

    /**
     * Get hidden size (model dimension).
     *
     * <p>Path: model.decoder.hidden_size
     *
     * @return hidden size, or null if not specified
     */
    public Integer getHiddenSize() {
        JsonNode decoderNode = configNode.path("model").path("decoder");
        if (decoderNode.has("hidden_size")) {
            return decoderNode.path("hidden_size").asInt();
        }
        return null;
    }

    /**
     * Get BOS (beginning of sequence) token ID.
     *
     * <p>Path: model.bos_token_id
     *
     * @return BOS token ID, or null if not specified
     */
    public Integer getBosTokenId() {
        JsonNode modelNode = configNode.path("model");
        if (modelNode.has("bos_token_id")) {
            return modelNode.path("bos_token_id").asInt();
        }
        return null;
    }

    /**
     * Get EOS (end of sequence) token ID(s).
     *
     * <p>Path: model.eos_token_id
     *
     * <p>Note: Some models have multiple EOS tokens (array). This method returns the first one.
     *
     * @return EOS token ID, or null if not specified
     */
    public Integer getEosTokenId() {
        JsonNode modelNode = configNode.path("model");
        if (modelNode.has("eos_token_id")) {
            JsonNode eosNode = modelNode.path("eos_token_id");
            if (eosNode.isArray() && eosNode.size() > 0) {
                return eosNode.get(0).asInt();
            } else if (eosNode.isInt()) {
                return eosNode.asInt();
            }
        }
        return null;
    }

    /**
     * Get PAD token ID.
     *
     * <p>Path: model.pad_token_id
     *
     * @return PAD token ID, or null if not specified
     */
    public Integer getPadTokenId() {
        JsonNode modelNode = configNode.path("model");
        if (modelNode.has("pad_token_id")) {
            return modelNode.path("pad_token_id").asInt();
        }
        return null;
    }

    /**
     * Get vocabulary size.
     *
     * <p>Path: model.vocab_size
     *
     * @return vocabulary size, or null if not specified
     */
    public Integer getVocabSize() {
        JsonNode modelNode = configNode.path("model");
        if (modelNode.has("vocab_size")) {
            return modelNode.path("vocab_size").asInt();
        }
        return null;
    }

    /**
     * Get model type.
     *
     * <p>Path: model.type
     *
     * @return model type (e.g., "llama", "phi3"), or null if not specified
     */
    public String getModelType() {
        JsonNode modelNode = configNode.path("model");
        if (modelNode.has("type")) {
            return modelNode.path("type").asText();
        }
        return null;
    }

    /**
     * Check if model has KV-cache support.
     *
     * <p>Checks for presence of past_key_names and past_value_names in decoder inputs.
     *
     * @return true if KV-cache is supported
     */
    public boolean hasKVCacheSupport() {
        JsonNode inputsNode = configNode.path("model").path("decoder").path("inputs");
        return inputsNode.has("past_key_names") && inputsNode.has("past_value_names");
    }

    /**
     * Get the raw JSON configuration node.
     *
     * <p>Allows access to GenAI-specific fields not covered by this parser.
     *
     * @return JsonNode representing the entire genai_config.json
     */
    public JsonNode getConfigNode() {
        return configNode;
    }

    /**
     * Get the path to the config file.
     *
     * @return path to genai_config.json file
     */
    public Path getConfigPath() {
        return configPath;
    }

    @Override
    public String toString() {
        return String.format("GenAIConfigParser{modelType=%s, numKVHeads=%s, headSize=%s, hasKVCache=%s}",
                getModelType(), getNumKeyValueHeads(), getHeadSize(), hasKVCacheSupport());
    }
}
