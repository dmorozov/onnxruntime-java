package com.badu.ai.onnx.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Parses Whisper generation configuration from generation_config.json.
 *
 * <p>This parser reads Whisper-specific configuration files and extracts:
 * <ul>
 *   <li>Multilingual vs English-only detection</li>
 *   <li>Maximum sequence length</li>
 *   <li>Token suppression rules</li>
 *   <li>Special token IDs (EOS, BOS, PAD, etc.)</li>
 *   <li>Timestamp configuration</li>
 * </ul>
 *
 * <p><strong>File Structure:</strong>
 * <pre>{@code
 * {
 *   "is_multilingual": false,
 *   "max_length": 448,
 *   "eos_token_id": 50256,
 *   "bos_token_id": 50257,
 *   "suppress_tokens": [1, 2, 7, ...],
 *   "begin_suppress_tokens": [220, 50256],
 *   ...
 * }
 * }</pre>
 *
 * <p><strong>Usage:</strong>
 * <pre>{@code
 * Path configPath = Paths.get("models/whisper-tiny.en-ONNX/generation_config.json");
 * WhisperConfigParser parser = new WhisperConfigParser(configPath);
 * WhisperGenerationConfig config = parser.parse();
 * }</pre>
 *
 * @see WhisperGenerationConfig
 * @see GenAIConfigParser
 */
public class WhisperConfigParser {
    private static final Logger logger = LoggerFactory.getLogger(WhisperConfigParser.class);

    private final JsonNode configNode;
    private final Path configPath;

    /**
     * Creates a WhisperConfigParser from a generation_config.json file.
     *
     * @param configPath path to generation_config.json file
     * @throws IOException if file cannot be read or JSON is invalid
     */
    public WhisperConfigParser(Path configPath) throws IOException {
        if (!Files.exists(configPath)) {
            throw new IOException("Whisper config file does not exist: " + configPath);
        }

        this.configPath = configPath;
        ObjectMapper mapper = new ObjectMapper();
        String jsonContent = Files.readString(configPath);
        this.configNode = mapper.readTree(jsonContent);

        logger.debug("Parsed generation_config.json from: {}", configPath);
    }

    /**
     * Parses the configuration file and returns a WhisperGenerationConfig.
     *
     * @return Parsed Whisper generation configuration
     */
    public WhisperGenerationConfig parse() {
        WhisperGenerationConfig.WhisperGenerationConfigBuilder builder =
                WhisperGenerationConfig.builder();

        // Parse is_multilingual
        if (configNode.has("is_multilingual")) {
            builder.isMultilingual(configNode.get("is_multilingual").asBoolean());
        }

        // Parse max_length
        if (configNode.has("max_length")) {
            builder.maxLength(configNode.get("max_length").asInt());
        }

        // Parse token IDs
        if (configNode.has("eos_token_id")) {
            builder.eosTokenId(configNode.get("eos_token_id").asLong());
        }

        if (configNode.has("bos_token_id")) {
            builder.bosTokenId(configNode.get("bos_token_id").asLong());
        }

        if (configNode.has("pad_token_id")) {
            builder.padTokenId(configNode.get("pad_token_id").asLong());
        }

        if (configNode.has("decoder_start_token_id")) {
            builder.decoderStartTokenId(configNode.get("decoder_start_token_id").asLong());
        }

        if (configNode.has("no_timestamps_token_id")) {
            builder.noTimestampsTokenId(configNode.get("no_timestamps_token_id").asLong());
        }

        if (configNode.has("prev_sot_token_id")) {
            builder.prevSotTokenId(configNode.get("prev_sot_token_id").asLong());
        }

        // Parse timestamp configuration
        if (configNode.has("return_timestamps")) {
            builder.returnTimestamps(configNode.get("return_timestamps").asBoolean());
        }

        if (configNode.has("max_initial_timestamp_index")) {
            builder.maxInitialTimestampIndex(configNode.get("max_initial_timestamp_index").asInt());
        }

        // Parse suppress_tokens
        if (configNode.has("suppress_tokens")) {
            List<Long> suppressTokens = parseTokenList(configNode.get("suppress_tokens"));
            builder.suppressTokens(suppressTokens);
            logger.debug("Parsed {} suppress_tokens", suppressTokens.size());
        }

        // Parse begin_suppress_tokens
        if (configNode.has("begin_suppress_tokens")) {
            List<Long> beginSuppressTokens = parseTokenList(configNode.get("begin_suppress_tokens"));
            builder.beginSuppressTokens(beginSuppressTokens);
            logger.debug("Parsed {} begin_suppress_tokens", beginSuppressTokens.size());
        }

        // Parse forced_decoder_ids
        if (configNode.has("forced_decoder_ids")) {
            List<List<Integer>> forcedDecoderIds = parseForcedDecoderIds(configNode.get("forced_decoder_ids"));
            builder.forcedDecoderIds(forcedDecoderIds);
            logger.debug("Parsed {} forced_decoder_ids", forcedDecoderIds.size());
        }

        // Parse alignment_heads
        if (configNode.has("alignment_heads")) {
            List<List<Integer>> alignmentHeads = parseAlignmentHeads(configNode.get("alignment_heads"));
            builder.alignmentHeads(alignmentHeads);
            logger.debug("Parsed {} alignment_heads", alignmentHeads.size());
        }

        WhisperGenerationConfig config = builder.build();
        logger.info("Parsed Whisper config: {}", config);

        return config;
    }

    /**
     * Parses a JSON array of token IDs into a List<Long>.
     */
    private List<Long> parseTokenList(JsonNode arrayNode) {
        List<Long> tokens = new ArrayList<>();
        if (arrayNode.isArray()) {
            for (JsonNode tokenNode : arrayNode) {
                tokens.add(tokenNode.asLong());
            }
        }
        return tokens;
    }

    /**
     * Parses forced_decoder_ids array: [[position, token_id], ...]
     */
    private List<List<Integer>> parseForcedDecoderIds(JsonNode arrayNode) {
        List<List<Integer>> result = new ArrayList<>();
        if (arrayNode.isArray()) {
            for (JsonNode pairNode : arrayNode) {
                if (pairNode.isArray() && pairNode.size() == 2) {
                    List<Integer> pair = new ArrayList<>();
                    pair.add(pairNode.get(0).asInt());
                    pair.add(pairNode.get(1).asInt());
                    result.add(pair);
                }
            }
        }
        return result;
    }

    /**
     * Parses alignment_heads array: [[layer, head], ...]
     */
    private List<List<Integer>> parseAlignmentHeads(JsonNode arrayNode) {
        List<List<Integer>> result = new ArrayList<>();
        if (arrayNode.isArray()) {
            for (JsonNode pairNode : arrayNode) {
                if (pairNode.isArray() && pairNode.size() == 2) {
                    List<Integer> pair = new ArrayList<>();
                    pair.add(pairNode.get(0).asInt());
                    pair.add(pairNode.get(1).asInt());
                    result.add(pair);
                }
            }
        }
        return result;
    }

    /**
     * Get the raw JSON configuration node.
     *
     * @return JsonNode representing the entire generation_config.json
     */
    public JsonNode getConfigNode() {
        return configNode;
    }

    /**
     * Get the path to the config file.
     *
     * @return path to generation_config.json file
     */
    public Path getConfigPath() {
        return configPath;
    }

    /**
     * Parses a Whisper generation config from a model directory.
     *
     * <p>Looks for generation_config.json in the specified directory.
     *
     * @param modelPath Path to model directory
     * @return Parsed configuration, or default config if file not found
     */
    public static WhisperGenerationConfig parseFromModelPath(Path modelPath) {
        Path configPath = modelPath.resolve("generation_config.json");

        if (!Files.exists(configPath)) {
            logger.warn("generation_config.json not found at: {}. Using defaults.", configPath);
            return WhisperGenerationConfig.getDefault();
        }

        try {
            WhisperConfigParser parser = new WhisperConfigParser(configPath);
            return parser.parse();
        } catch (IOException e) {
            logger.error("Failed to parse generation_config.json: {}. Using defaults.", e.getMessage());
            return WhisperGenerationConfig.getDefault();
        }
    }

    @Override
    public String toString() {
        return String.format("WhisperConfigParser{configPath=%s}", configPath);
    }
}
