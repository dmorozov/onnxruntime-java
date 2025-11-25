package com.badu.ai.onnx.gpu;

import ai.onnxruntime.OrtProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Factory for creating appropriate GPU adapters based on available hardware.
 *
 * <p>Automatically detects available GPUs and selects the best execution provider:
 * <ol>
 *   <li>CUDA (NVIDIA GPUs) - Best performance for NVIDIA hardware</li>
 *   <li>ROCm (AMD GPUs) - Optimized for AMD Radeon/Instinct</li>
 *   <li>CoreML (Apple Silicon) - Optimized for M-series chips</li>
 *   <li>WebGPU (Cross-platform) - Experimental Vulkan/Metal/DX12 backend</li>
 * </ol>
 *
 * <p><b>Priority Order:</b>
 * <p>The factory tries adapters in this priority (first available wins):
 * <pre>
 * 1. CUDA    (NVIDIA) - Most mature, best performance
 * 2. ROCm    (AMD)    - Good performance, actively developed
 * 3. CoreML  (Apple)  - Best for Apple Silicon
 * 4. WebGPU  (Any)    - Experimental fallback
 * </pre>
 *
 * <p><b>Usage Example:</b>
 * <pre>{@code
 * // Auto-detect and create best adapter
 * GpuAdapter adapter = GpuAdapterFactory.createAdapter();
 * if (adapter != null && adapter.isAvailable()) {
 *     logger.info("Using GPU: {}", adapter.getName());
 *     adapter.configure(sessionOptions, 0);
 * }
 *
 * // Create specific adapter
 * GpuAdapter rocm = GpuAdapterFactory.createAdapter(OrtProvider.ROCM);
 * }</pre>
 */
public class GpuAdapterFactory {

  private static final Logger logger = LoggerFactory.getLogger(GpuAdapterFactory.class);

  /**
   * Creates the best available GPU adapter based on hardware detection.
   *
   * <p>Tries adapters in priority order: CUDA > ROCm > CoreML > WebGPU.
   *
   * @return Best available GPU adapter, or null if no GPU is available
   */
  public static GpuAdapter createAdapter() {
    // Try adapters in priority order
    List<GpuAdapter> candidates = new ArrayList<>();
    candidates.add(new CudaGpuAdapter());    // NVIDIA (best maturity)
    candidates.add(new RocmGpuAdapter());    // AMD (good support)
    candidates.add(new CoreMlGpuAdapter());  // Apple Silicon (optimized)
    // Skipping WebGPU as it's experimental and not yet supported in Java API

    for (GpuAdapter adapter : candidates) {
      if (adapter.isAvailable()) {
        logger.info("GPU Adapter selected: {} ({})",
            adapter.getName(), adapter.getProviderType());
        logger.info("  - Device count: {}", adapter.getDeviceCount());
        logger.info("  - FP16 support: {}", adapter.supportsFp16());
        logger.info("  - INT8 support: {}", adapter.supportsInt8());
        return adapter;
      }
    }

    logger.warn("No GPU adapter available, will use CPU execution");
    return null;
  }

  /**
   * Creates a specific GPU adapter by provider type.
   *
   * @param provider OrtProvider type (CUDA, ROCM, CORE_ML, etc.)
   * @return GPU adapter for the specified provider, or null if not supported
   */
  public static GpuAdapter createAdapter(OrtProvider provider) {
    if (provider == null) {
      throw new IllegalArgumentException("Provider cannot be null");
    }

    GpuAdapter adapter = switch (provider) {
      case CUDA -> new CudaGpuAdapter();
      case ROCM -> new RocmGpuAdapter();
      case CORE_ML -> new CoreMlGpuAdapter();
      case WEBGPU -> new WebGpuAdapter();
      default -> {
        logger.warn("Unsupported GPU provider: {}. Supported: CUDA, ROCM, CORE_ML, WEBGPU", provider);
        yield null;
      }
    };

    if (adapter != null && !adapter.isAvailable()) {
      logger.warn("GPU adapter {} created but not available on this system", adapter.getName());
    }

    return adapter;
  }

  /**
   * Gets all available GPU adapters on the current system.
   *
   * @return List of available GPU adapters (may be empty if no GPUs)
   */
  public static List<GpuAdapter> getAvailableAdapters() {
    List<GpuAdapter> available = new ArrayList<>();

    // Check all adapter types
    GpuAdapter[] allAdapters = {
        new CudaGpuAdapter(),
        new RocmGpuAdapter(),
        new CoreMlGpuAdapter(),
        new WebGpuAdapter()
    };

    for (GpuAdapter adapter : allAdapters) {
      if (adapter.isAvailable()) {
        available.add(adapter);
      }
    }

    return available;
  }

  /**
   * Checks if any GPU is available on the system.
   *
   * @return true if at least one GPU adapter is available
   */
  public static boolean isAnyGpuAvailable() {
    return createAdapter() != null;
  }

  /**
   * Gets detailed system GPU information for debugging.
   *
   * @return Multi-line string with GPU detection details
   */
  public static String getSystemGpuInfo() {
    StringBuilder info = new StringBuilder("GPU Detection Report:\n");

    List<GpuAdapter> available = getAvailableAdapters();

    if (available.isEmpty()) {
      info.append("  No GPUs detected\n");
      info.append("  Execution will use CPU only\n");
    } else {
      info.append("  Available GPUs: ").append(available.size()).append("\n");

      for (int i = 0; i < available.size(); i++) {
        GpuAdapter adapter = available.get(i);
        info.append("\n").append(i + 1).append(". ").append(adapter.getName()).append(":\n");
        info.append("     Provider: ").append(adapter.getProviderType()).append("\n");
        info.append("     Devices: ").append(adapter.getDeviceCount()).append("\n");

        for (int deviceId = 0; deviceId < adapter.getDeviceCount(); deviceId++) {
          info.append("       Device ").append(deviceId).append(": ")
              .append(adapter.getDeviceInfo(deviceId)).append("\n");
        }

        info.append("     FP16: ").append(adapter.supportsFp16() ? "Yes" : "No").append("\n");
        info.append("     INT8: ").append(adapter.supportsInt8() ? "Yes" : "No").append("\n");
        info.append("     Arena: ").append(adapter.getRecommendedArenaSize()).append(" MB\n");
      }
    }

    return info.toString();
  }
}
