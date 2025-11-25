package com.badu.ai.onnx.utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.badu.ai.onnx.config.DeviceType;
import com.badu.ai.onnx.config.ModelConfig;
import com.badu.ai.onnx.gpu.GpuAdapter;
import com.badu.ai.onnx.gpu.GpuAdapterFactory;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;

/**
 * Utility class for model configuration and optimization
 */
public class ModelUtils {

  private static final Logger logger = LoggerFactory.getLogger(ModelUtils.class);

  private static List<OrtProvider> cachedGpuAvailable = null;

  /**
   * Create optimized session options for the ONNX model
   * Automatically detects and enables GPU if CUDA is available.
   *
   * @param config Model config
   * @return Configured session options
   * @throws OrtException If there's an error configuring options
   */
  public static OrtSession.SessionOptions createSessionOptions(ModelConfig config) throws OrtException {

    int numThreads = config.getThreadsCount();
    if (numThreads < 0) {
      numThreads = ModelUtils.getOptimalThreadCount();
    }

    boolean useGpu = (config.getDeviceType() == DeviceType.GPU);
    OrtSession.SessionOptions options = new OrtSession.SessionOptions();

    // Set optimization level
    options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

    // Set number of threads for CPU execution
    if (numThreads > 0) {
      /**
       * This method sets the number of threads used for executing a single operation
       * within a graph.
       * 
       * Recommendation:
       * - For optimal performance on a single inference, setting this to the number
       * of available CPU cores or a slightly lower value can be beneficial.
       * - If ONNX Runtime is built with OpenMP, consider using OpenMP environment
       * variables to control intra-op threads, as they might override or interact
       * with this setting.
       * - Setting it to 1 effectively disables intra-op parallelism for that
       * operation, forcing it to run on the main thread.
       * - Setting it to 0 allows ONNX Runtime to choose the number of threads
       * automatically.
       */
      options.setIntraOpNumThreads(numThreads);

      /**
       * This method sets the number of threads in the CPU thread pool used for
       * executing multiple operations concurrently or for handling multiple
       * concurrent inference requests.
       * 
       * Recommendation:
       * - This is particularly relevant when running multiple ONNX Runtime sessions
       * or handling a high volume of concurrent inference requests.
       * - The optimal value depends on the number of concurrent operations/requests
       * and the available CPU resources. A common strategy is to set it to a fraction
       * or a multiple of the available CPU cores, depending on the workload.
       * - Unlike intra-op threads, inter-op threads are not typically affected by
       * OpenMP settings and should be configured using this API.
       */
      options.setInterOpNumThreads(numThreads);

    }

    // numThreads == 0 is to Allow ONNX Runtime to decide
    if (numThreads > 1 || numThreads == 0) {
      /**
       * This method controls the level of graph optimizations applied to your ONNX
       * model. Higher optimization levels generally lead to better performance but
       * might increase session creation time.
       */
      options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
      /**
       * This method determines how operators within the graph are executed, either
       * sequentially or in parallel.
       */
      options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);
    } else {
      // Execution mode
      options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
    }

    // Enable ONNX Runtime profiling if requested (disabled by default for production)
    if (config.isEnableProfiling()) {
      String profileFileName = String.format("onnx_profile_%d.json", System.currentTimeMillis());
      options.enableProfiling(profileFileName);
      logger.info("ONNX Runtime profiling enabled: {}", profileFileName);
    }

    // Memory optimization
    options.setMemoryPatternOptimization(true);

    // Memory arena configuration (10-15% memory reduction)
    if (config.isEnableMemoryArena()) {
      configureMemoryArena(options, config);
    }

    if (useGpu) {
      configureGpu(config, options);
    }

    return options;
  }

  /**
   * Create optimized session options specifically for encoder models.
   *
   * <p>Encoders benefit from parallel execution because they process the entire input
   * sequence at once with independent attention heads that can be computed in parallel.
   *
   * <p><b>Optimization Strategy:</b>
   * <ul>
   *   <li>Parallel execution mode for inter-op parallelism</li>
   *   <li>More intra-op threads to parallelize attention heads</li>
   *   <li>Inter-op threads to parallelize independent operations</li>
   * </ul>
   *
   * <p><b>Expected Impact:</b> 10-15% speedup on multi-core CPUs
   *
   * @param config Model configuration
   * @return Session options optimized for encoder execution
   * @throws OrtException If configuration fails
   */
  public static OrtSession.SessionOptions createEncoderOptions(ModelConfig config) throws OrtException {
    int cpuCores = Runtime.getRuntime().availableProcessors();

    OrtSession.SessionOptions options = new OrtSession.SessionOptions();

    // Encoder: Maximize parallelism across attention heads
    // Use half of available cores for intra-op (within single operation)
    int intraThreads = Math.max(1, cpuCores / 2);
    // Use quarter of cores for inter-op (across operations)
    int interThreads = Math.max(1, cpuCores / 4);

    options.setIntraOpNumThreads(intraThreads);
    options.setInterOpNumThreads(interThreads);
    options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);

    // Graph optimization
    options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
    options.setMemoryPatternOptimization(true);

    // Memory arena configuration
    if (config.isEnableMemoryArena()) {
      configureMemoryArena(options, config);
    }

    // GPU configuration if enabled
    if (config.getDeviceType() == DeviceType.GPU) {
      configureGpu(config, options);
    }

    // Profiling if enabled
    if (config.isEnableProfiling()) {
      String profileFileName = String.format("onnx_profile_encoder_%d.json", System.currentTimeMillis());
      options.enableProfiling(profileFileName);
      logger.info("Encoder profiling enabled: {}", profileFileName);
    }

    logger.debug("Encoder session options: intra={}, inter={}, mode=PARALLEL",
        intraThreads, interThreads);

    return options;
  }

  /**
   * Create optimized session options specifically for decoder models.
   *
   * <p>Decoders perform auto-regressive generation (one token at a time) and benefit
   * from low-latency sequential execution rather than parallelism.
   *
   * <p><b>Optimization Strategy:</b>
   * <ul>
   *   <li>Sequential execution mode for minimal overhead</li>
   *   <li>More intra-op threads for within-operation parallelism</li>
   *   <li>Minimal inter-op threads (sequential processing)</li>
   * </ul>
   *
   * <p><b>Expected Impact:</b> Lower latency per token, no regression
   *
   * @param config Model configuration
   * @return Session options optimized for decoder execution
   * @throws OrtException If configuration fails
   */
  public static OrtSession.SessionOptions createDecoderOptions(ModelConfig config) throws OrtException {
    int cpuCores = Runtime.getRuntime().availableProcessors();

    OrtSession.SessionOptions options = new OrtSession.SessionOptions();

    // Decoder: Optimize for low-latency single-token processing
    // Use half of cores for intra-op parallelism
    int intraThreads = Math.max(1, cpuCores / 2);
    // Sequential inter-op (minimal parallelism overhead)
    int interThreads = 1;

    options.setIntraOpNumThreads(intraThreads);
    options.setInterOpNumThreads(interThreads);
    options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);

    // Graph optimization
    options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
    options.setMemoryPatternOptimization(true);

    // Memory arena configuration
    if (config.isEnableMemoryArena()) {
      configureMemoryArena(options, config);
    }

    // GPU configuration if enabled
    if (config.getDeviceType() == DeviceType.GPU) {
      configureGpu(config, options);
    }

    // Profiling if enabled
    if (config.isEnableProfiling()) {
      String profileFileName = String.format("onnx_profile_decoder_%d.json", System.currentTimeMillis());
      options.enableProfiling(profileFileName);
      logger.info("Decoder profiling enabled: {}", profileFileName);
    }

    logger.debug("Decoder session options: intra={}, inter={}, mode=SEQUENTIAL",
        intraThreads, interThreads);

    return options;
  }

  /**
   * Configure memory arena for reduced memory fragmentation.
   *
   * <p>Memory arena optimization reduces memory fragmentation by:
   * <ul>
   *   <li>Using device allocator for model initializers (weights)</li>
   *   <li>Enabling arena shrinkage to release unused memory</li>
   *   <li>Controlling arena growth strategy</li>
   * </ul>
   *
   * <p>Expected impact: 10-15% memory reduction
   *
   * @param options Session options to configure
   * @param config Model configuration with memory arena settings
   * @throws OrtException If configuration fails
   */
  private static void configureMemoryArena(OrtSession.SessionOptions options, ModelConfig config)
      throws OrtException {
    try {
      // Use device allocator for initializers (reduces memory fragmentation)
      options.addConfigEntry("session.use_device_allocator_for_initializers", "1");

      // Enable memory arena shrinkage (release unused memory)
      options.addConfigEntry("memory.enable_memory_arena_shrinkage", "cpu:0;gpu:0");

      // Set memory arena size limit if specified
      if (config.getMemoryArenaSize() > 0) {
        options.addConfigEntry("session.memory.arena_extend_strategy", "kSameAsRequested");
        logger.debug("Memory arena configured: size={}MB",
            config.getMemoryArenaSize() / 1024 / 1024);
      } else {
        logger.debug("Memory arena configured: size=auto");
      }

    } catch (OrtException e) {
      logger.warn("Failed to configure memory arena, continuing without optimization: {}",
          e.getMessage());
      // Don't throw - arena configuration is optional optimization
    }
  }

  /**
   * Configure GPU execution for the ONNX Runtime session.
   *
   * <p>Uses the GPU adapter factory to automatically detect and configure
   * the best available GPU provider (CUDA, ROCm, CoreML, etc.).
   *
   * @param config Model configuration containing GPU device preferences
   * @param options Session options to configure with GPU settings
   * @throws OrtException If GPU configuration fails
   */
  public static final void configureGpu(ModelConfig config, OrtSession.SessionOptions options) throws OrtException {
    // Use the factory to auto-detect the best available GPU adapter
    GpuAdapter adapter = GpuAdapterFactory.createAdapter();

    if (adapter == null) {
      logger.warn("GPU requested but not available, using CPU execution provider");
      return;
    }

    try {
      // Get device ID from config (default to 0)
      int deviceId = config.getGpuDeviceId() >= 0 ? config.getGpuDeviceId() : 0;

      // Validate device ID
      int deviceCount = adapter.getDeviceCount();
      if (deviceId >= deviceCount) {
        logger.warn("Requested GPU device {} but only {} device(s) available, using device 0",
            deviceId, deviceCount);
        deviceId = 0;
      }

      // Configure the GPU adapter
      adapter.configure(options, deviceId);

      // Log detailed GPU information
      logger.info("GPU configured successfully:");
      logger.info("  Provider: {} ({})", adapter.getName(), adapter.getProviderType());
      logger.info("  Device: {}", adapter.getDeviceInfo(deviceId));
      logger.info("  FP16 support: {}", adapter.supportsFp16());
      logger.info("  INT8 support: {}", adapter.supportsInt8());

    } catch (OrtException e) {
      logger.error("Failed to configure GPU execution provider: {}", e.getMessage());
      logger.warn("Falling back to CPU execution");
      logger.warn("Ensure GPU drivers and ONNX Runtime GPU libraries are installed.");
      throw e;
    }
  }

  /**
   * Load configuration from properties file
   * 
   * @param configPath Path to configuration file
   * @return Properties object with configuration
   */
  public static Properties loadConfiguration(String configPath) {
    Properties props = new Properties();

    // Set default values
    props.setProperty("model.max_sequence_length", "2048");
    props.setProperty("model.vocab_size", "151936"); // Qwen3 default
    props.setProperty("model.hidden_size", "2048");
    props.setProperty("model.num_attention_heads", "16");
    props.setProperty("model.num_hidden_layers", "28");

    props.setProperty("generation.max_tokens", "100");
    props.setProperty("generation.temperature", "0.7");
    props.setProperty("generation.top_k", "50");
    props.setProperty("generation.top_p", "0.9");
    props.setProperty("generation.repetition_penalty", "1.0");

    props.setProperty("runtime.use_gpu", "false");
    props.setProperty("runtime.gpu_device_id", "0");
    props.setProperty("runtime.num_threads", "4");

    // Try to load from file if it exists
    Path path = Paths.get(configPath);
    if (Files.exists(path)) {
      try {
        props.load(Files.newInputStream(path));
        logger.info("Loaded configuration from: {}", configPath);
      } catch (IOException e) {
        logger.warn("Failed to load configuration file, using defaults: {}", e.getMessage());
      }
    } else {
      logger.info("Configuration file not found, using default values");
    }

    return props;
  }

  /**
   * Validate model file
   * 
   * @param modelPath Path to the ONNX model file
   * @return true if valid, false otherwise
   */
  public static boolean validateModelFile(String modelPath) {
    Path path = Paths.get(modelPath);

    if (!Files.exists(path)) {
      logger.error("Model file does not exist: {}", modelPath);
      return false;
    }

    if (!Files.isReadable(path)) {
      logger.error("Model file is not readable: {}", modelPath);
      return false;
    }

    // Check file extension
    if (!modelPath.toLowerCase().endsWith(".onnx")) {
      logger.warn("Model file does not have .onnx extension: {}", modelPath);
    }

    // Check file size
    try {
      long size = Files.size(path);
      logger.info("Model file size: {} MB", size / (1024 * 1024));

      if (size == 0) {
        logger.error("Model file is empty");
        return false;
      }

      if (size > 10_000_000_000L) { // 10 GB
        logger.warn("Model file is very large (>10GB), may cause memory issues");
      }
    } catch (IOException e) {
      logger.error("Error checking model file size", e);
      return false;
    }

    return true;
  }

  /**
   * Check if any GPU is available for ONNX Runtime (CUDA, ROCm, CoreML, etc.).
   *
   * <p>Uses the GPU adapter factory to detect available GPU providers.
   *
   * @return true if any GPU provider is available, false otherwise
   */
  public static boolean isGpuAvailable() {
    return GpuAdapterFactory.isAnyGpuAvailable();
  }

  /**
   * Get the optimal number of threads for CPU inference Uses the number of
   * available processors,
   * capped at a reasonable maximum
   *
   * @return Optimal thread count
   */
  public static int getOptimalThreadCount() {
    int processors = Runtime.getRuntime().availableProcessors();
    // Cap at 8 threads to avoid overhead, but use at least 4
    return Math.max(4, Math.min(processors, 8));
  }

  /**
   * Get system information for debugging, including detailed GPU detection.
   *
   * @return System information as string
   */
  public static String getSystemInfo() {
    StringBuilder info = new StringBuilder();

    info.append("System Information:\n");
    info.append("  OS: ").append(System.getProperty("os.name")).append(" ")
        .append(System.getProperty("os.version")).append("\n");
    info.append("  Java Version: ").append(System.getProperty("java.version")).append("\n");
    info.append("  Java Vendor: ").append(System.getProperty("java.vendor")).append("\n");
    info.append("  Available Processors: ").append(Runtime.getRuntime().availableProcessors())
        .append("\n");

    // Memory information
    long maxMemory = Runtime.getRuntime().maxMemory();
    info.append("  Max Memory: ").append(maxMemory / (1024 * 1024)).append(" MB\n");

    long totalMemory = Runtime.getRuntime().totalMemory();
    info.append("  Total Memory: ").append(totalMemory / (1024 * 1024)).append(" MB\n");

    long freeMemory = Runtime.getRuntime().freeMemory();
    info.append("  Free Memory: ").append(freeMemory / (1024 * 1024)).append(" MB\n");

    // Detailed GPU information using the factory
    info.append("\n");
    info.append(GpuAdapterFactory.getSystemGpuInfo());

    return info.toString();
  }

  /**
   * Format tokens for display
   * 
   * @param tokens     Array of token IDs
   * @param maxDisplay Maximum number of tokens to display
   * @return Formatted string
   */
  public static String formatTokens(long[] tokens, int maxDisplay) {
    if (tokens == null || tokens.length == 0) {
      return "[]";
    }

    StringBuilder sb = new StringBuilder("[");
    int limit = Math.min(tokens.length, maxDisplay);

    for (int i = 0; i < limit; i++) {
      if (i > 0)
        sb.append(", ");
      sb.append(tokens[i]);
    }

    if (tokens.length > maxDisplay) {
      sb.append(", ... (").append(tokens.length - maxDisplay).append(" more)");
    }

    sb.append("]");
    return sb.toString();
  }

  public static final List<OrtProvider> availableGpus() {
    if (cachedGpuAvailable != null) {
      return cachedGpuAvailable;
    }

    try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
      var providers = OrtEnvironment.getAvailableProviders();
      logger.debug("Available ONNX Runtime providers: {}", providers);

      cachedGpuAvailable = providers != null ? new ArrayList<>(providers) : List.of();
    } catch (Exception e) {
      logger.warn("Error checking CUDA availability", e);
      cachedGpuAvailable = List.of();
    }

    return cachedGpuAvailable;
  }
}
