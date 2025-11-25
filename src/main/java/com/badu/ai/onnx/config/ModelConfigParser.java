package com.badu.ai.onnx.config;

import com.badu.ai.onnx.engine.ModelArchitecture;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Parses model config.json to extract model type and parameters.
 *
 * <p>This parser reads HuggingFace-style config.json files and extracts:
 * <ul>
 *   <li>Model architecture type (phi3, llama, qwen2, qwen3, t5, bart)</li>
 *   <li>Special token IDs (BOS, EOS, PAD)</li>
 *   <li>Vocabulary size</li>
 *   <li>Architecture-specific parameters</li>
 *   <li>KV-cache parameters (num_key_value_heads, head_dim)</li>
 * </ul>
 *
 * <p><b>Supported Architectures:</b>
 * <ul>
 *   <li>"phi3" - Phi-3 Mini decoder-only</li>
 *   <li>"llama" - Llama decoder-only</li>
 *   <li>"qwen2", "qwen", "qwen3" - Qwen decoder-only</li>
 *   <li>"t5" - T5 encoder-decoder</li>
 *   <li>"bart" - BART encoder-decoder</li>
 * </ul>
 *
 * <p><b>KV-Cache Support:</b>
 * <p>For decoder-only models, this parser extracts KV-cache parameters used
 * in auto-regressive generation with past_key_values optimization:
 * <ul>
 *   <li>{@link #getNumKeyValueHeads()} - Number of KV-heads (GQA)</li>
 *   <li>{@link #getHeadDim()} - Dimension per attention head</li>
 * </ul>
 *
 * <p><b>Usage Example:</b>
 * <pre>{@code
 * Path configPath = Paths.get("models/Qwen3-1.7B-ONNX/config.json");
 * ModelConfigParser parser = new ModelConfigParser(configPath);
 * ModelArchitecture arch = parser.detectArchitecture();
 * Integer numKVHeads = parser.getNumKeyValueHeads();  // 8 for Qwen3
 * Integer headDim = parser.getHeadDim();              // 128 for Qwen3
 * }</pre>
 *
 * @see ModelArchitecture
 * @see com.badu.ai.onnx.config.GenAIConfigParser
 */
public class ModelConfigParser {
    private static final Logger logger = LoggerFactory.getLogger(ModelConfigParser.class);

    private final JsonNode configNode;
    private final Path configPath;

    /**
     * Creates a ModelConfigParser from a config.json file.
     *
     * @param configPath path to config.json file
     * @throws IOException if file cannot be read or JSON is invalid
     */
    public ModelConfigParser(Path configPath) throws IOException {
        if (!Files.exists(configPath)) {
            throw new IOException("Config file does not exist: " + configPath);
        }

        this.configPath = configPath;
        ObjectMapper mapper = new ObjectMapper();
        String jsonContent = Files.readString(configPath);
        this.configNode = mapper.readTree(jsonContent);

        logger.debug("Parsed config.json from: {}", configPath);
    }

    /**
     * Detect model architecture from config.json.
     *
     * <p>Detection strategy:
     * <ol>
     *   <li>Check "model_type" field (primary)</li>
     *   <li>Check "architectures" array (fallback)</li>
     *   <li>Check for encoder/decoder fields (fallback)</li>
     * </ol>
     *
     * @return ModelArchitecture enum value
     */
    public ModelArchitecture detectArchitecture() {
        String modelType = getModelType();

        // Primary detection: model_type field
        ModelArchitecture arch = detectFromModelType(modelType);
        if (arch != null) {
            logger.info("Detected architecture from model_type: {} -> {}", modelType, arch);
            return arch;
        }

        // Fallback: Check architectures array
        arch = detectFromArchitectures();
        if (arch != null) {
            logger.info("Detected architecture from architectures field: {}", arch);
            return arch;
        }

        // Fallback: Check for decoder-only structure
        if (isDecoderOnly()) {
            logger.info("Detected DECODER_ONLY architecture from structure");
            return ModelArchitecture.DECODER_ONLY;
        }

        throw new IllegalArgumentException("Unknown model type: " + modelType);
    }

    /**
     * Detect architecture from model_type field.
     */
    private ModelArchitecture detectFromModelType(String modelType) {
        String normalized = modelType.toLowerCase();

        switch (normalized) {
            case "phi3":
            case "phi":
                return ModelArchitecture.DECODER_ONLY;

            case "llama":
                return ModelArchitecture.DECODER_ONLY;

            case "qwen3":
            case "qwen2":
            case "qwen":
                return ModelArchitecture.DECODER_ONLY;

            case "t5":
                return ModelArchitecture.T5_ENCODER_DECODER;

            case "bart":
                return ModelArchitecture.T5_ENCODER_DECODER;

            default:
                return null; // Unknown, try fallback
        }
    }

    /**
     * Detect architecture from architectures array.
     */
    private ModelArchitecture detectFromArchitectures() {
        if (!configNode.has("architectures")) {
            return null;
        }

        String architectures = configNode.path("architectures").toString().toLowerCase();

        // Check for decoder-only patterns
        if (architectures.contains("forcausallm") ||
            architectures.contains("gpt") ||
            architectures.contains("llama") ||
            architectures.contains("phi") ||
            architectures.contains("qwen")) {
            return ModelArchitecture.DECODER_ONLY;
        }

        // Check for encoder-decoder patterns
        if (architectures.contains("t5") ||
            architectures.contains("bart") ||
            architectures.contains("encoder") && architectures.contains("decoder")) {
            return ModelArchitecture.T5_ENCODER_DECODER;
        }

        return null; // Unknown
    }

    /**
     * Get model type from config.json.
     *
     * @return model type string (e.g., "phi3", "llama", "t5")
     */
    public String getModelType() {
        return configNode.path("model_type").asText("unknown");
    }

    /**
     * Get BOS (beginning of sequence) token ID.
     *
     * <p>Common defaults:
     * <ul>
     *   <li>Llama: 1</li>
     *   <li>Phi-3: 1</li>
     *   <li>Qwen: 151643</li>
     * </ul>
     *
     * @return BOS token ID (default: 1)
     */
    public int getBosTokenId() {
        return configNode.path("bos_token_id").asInt(1);
    }

    /**
     * Get EOS (end of sequence) token ID.
     *
     * <p>Common defaults:
     * <ul>
     *   <li>Llama: 2</li>
     *   <li>Phi-3: 32000</li>
     *   <li>Qwen: 151643</li>
     * </ul>
     *
     * @return EOS token ID (default: 2)
     */
    public int getEosTokenId() {
        return configNode.path("eos_token_id").asInt(2);
    }

    /**
     * Get PAD token ID.
     *
     * <p>If not specified in config, defaults to 0 or BOS token ID.
     *
     * @return PAD token ID (default: 0)
     */
    public int getPadTokenId() {
        // Try explicit pad_token_id first
        if (configNode.has("pad_token_id")) {
            return configNode.path("pad_token_id").asInt();
        }

        // Fallback: use 0 or BOS token ID
        int bosId = getBosTokenId();
        return bosId == 1 ? 0 : bosId;
    }

    /**
     * Get vocabulary size.
     *
     * <p>Common sizes:
     * <ul>
     *   <li>Llama: 32000</li>
     *   <li>Phi-3: 32064</li>
     *   <li>Qwen: 151936</li>
     *   <li>T5: 32128</li>
     * </ul>
     *
     * @return vocabulary size (default: 32000)
     */
    public int getVocabSize() {
        return configNode.path("vocab_size").asInt(32000);
    }

    /**
     * Check if config.json indicates decoder-only architecture.
     *
     * <p>Detection heuristics:
     * <ul>
     *   <li>Has "decoder" field but no "encoder" field</li>
     *   <li>architectures array contains "ForCausalLM"</li>
     *   <li>Has "num_attention_heads" but no "encoder_layers"</li>
     * </ul>
     *
     * @return true if decoder-only architecture detected
     */
    public boolean isDecoderOnly() {
        // Check for "decoder" pipeline in config
        if (configNode.has("decoder") && !configNode.has("encoder")) {
            return true;
        }

        // Check architectures for CausalLM pattern
        if (configNode.has("architectures")) {
            String architectures = configNode.path("architectures").toString();
            if (architectures.contains("ForCausalLM")) {
                return true;
            }
        }

        // Check for decoder-only structure (has attention heads but no encoder layers)
        if (configNode.has("num_attention_heads") && !configNode.has("encoder_layers")) {
            return true;
        }

        return false;
    }

    /**
     * Get maximum sequence length supported by the model.
     *
     * <p>Checks multiple field names:
     * <ul>
     *   <li>max_position_embeddings (primary)</li>
     *   <li>n_positions (GPT-style)</li>
     *   <li>max_sequence_length (alternative)</li>
     * </ul>
     *
     * @return maximum sequence length (default: 2048)
     */
    public int getMaxSequenceLength() {
        if (configNode.has("max_position_embeddings")) {
            return configNode.path("max_position_embeddings").asInt(2048);
        }

        if (configNode.has("n_positions")) {
            return configNode.path("n_positions").asInt(2048);
        }

        if (configNode.has("max_sequence_length")) {
            return configNode.path("max_sequence_length").asInt(2048);
        }

        return 2048; // Default
    }

    /**
     * Get number of attention heads.
     *
     * @return number of attention heads (default: 12)
     */
    public int getNumAttentionHeads() {
        return configNode.path("num_attention_heads").asInt(12);
    }

    /**
     * Get hidden size (model dimension).
     *
     * <p>Checks multiple field names:
     * <ul>
     *   <li>hidden_size (primary)</li>
     *   <li>d_model (T5-style)</li>
     *   <li>n_embd (GPT-style)</li>
     * </ul>
     *
     * @return hidden size (default: 768)
     */
    public int getHiddenSize() {
        if (configNode.has("hidden_size")) {
            return configNode.path("hidden_size").asInt(768);
        }

        if (configNode.has("d_model")) {
            return configNode.path("d_model").asInt(768);
        }

        if (configNode.has("n_embd")) {
            return configNode.path("n_embd").asInt(768);
        }

        return 768; // Default
    }

    /**
     * Get number of hidden layers.
     *
     * <p>Checks multiple field names:
     * <ul>
     *   <li>num_hidden_layers (primary)</li>
     *   <li>n_layer (GPT-style)</li>
     *   <li>num_layers (alternative)</li>
     * </ul>
     *
     * @return number of hidden layers (default: 12)
     */
    public int getNumLayers() {
        if (configNode.has("num_hidden_layers")) {
            return configNode.path("num_hidden_layers").asInt(12);
        }

        if (configNode.has("n_layer")) {
            return configNode.path("n_layer").asInt(12);
        }

        if (configNode.has("num_layers")) {
            return configNode.path("num_layers").asInt(12);
        }

        return 12; // Default
    }

    /**
     * Get number of key-value heads for KV-cache.
     *
     * <p>Used in decoder-only models with Grouped Query Attention (GQA).
     * The KV-cache stores keys and values for these heads during auto-regressive generation.
     *
     * <p>Path: num_key_value_heads (top-level in HuggingFace config.json)
     *
     * <p>Common values:
     * <ul>
     *   <li>Llama 3.2 1B: 8</li>
     *   <li>Phi-3 Mini: 32</li>
     *   <li>Qwen3 1.7B: 8</li>
     * </ul>
     *
     * @return number of KV-heads, or null if not specified in config
     */
    public Integer getNumKeyValueHeads() {
        if (configNode.has("num_key_value_heads")) {
            return configNode.path("num_key_value_heads").asInt();
        }
        return null;
    }

    /**
     * Get head dimension for attention heads.
     *
     * <p>This is the dimension of each attention head, used for KV-cache tensor shapes.
     * The method tries multiple strategies to extract this value:
     * <ol>
     *   <li>Check "head_dim" field (Qwen, newer models)</li>
     *   <li>Check "head_size" field (some models)</li>
     *   <li>Compute from hidden_size / num_attention_heads (fallback)</li>
     * </ol>
     *
     * <p>Common values:
     * <ul>
     *   <li>Llama 3.2 1B: 64</li>
     *   <li>Phi-3 Mini: 96</li>
     *   <li>Qwen3 1.7B: 128</li>
     * </ul>
     *
     * <p><b>Note:</b> Some models use "head_size" while others use "head_dim".
     * This method handles both naming conventions.
     *
     * @return head dimension, or null if cannot be determined
     */
    public Integer getHeadDim() {
        // Strategy 1: Try head_dim first (Qwen, newer models)
        if (configNode.has("head_dim")) {
            return configNode.path("head_dim").asInt();
        }

        // Strategy 2: Fallback to head_size (some models)
        if (configNode.has("head_size")) {
            return configNode.path("head_size").asInt();
        }

        // Strategy 3: Compute from hidden_size and num_attention_heads
        if (configNode.has("hidden_size") && configNode.has("num_attention_heads")) {
            int hiddenSize = configNode.path("hidden_size").asInt();
            int numHeads = configNode.path("num_attention_heads").asInt();
            if (numHeads > 0) {
                return hiddenSize / numHeads;
            }
        }

        return null;
    }

    /**
     * Get the raw JSON configuration node.
     *
     * <p>Allows access to model-specific fields not covered by this parser.
     *
     * @return JsonNode representing the entire config.json
     */
    public JsonNode getConfigNode() {
        return configNode;
    }

    /**
     * Get the path to the config file.
     *
     * @return path to config.json file
     */
    public Path getConfigPath() {
        return configPath;
    }

    @Override
    public String toString() {
        return String.format("ModelConfigParser{modelType=%s, architecture=%s, vocabSize=%d, maxSeqLen=%d}",
                getModelType(), detectArchitecture(), getVocabSize(), getMaxSequenceLength());
    }
}
