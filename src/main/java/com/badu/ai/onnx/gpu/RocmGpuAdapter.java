package com.badu.ai.onnx.gpu;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 * GPU adapter for AMD ROCm execution provider.
 *
 * <p>Supports AMD GPUs with ROCm 5.0+ (gfx900 architecture and newer).
 *
 * <p><b>Supported GPUs:</b>
 * <ul>
 *   <li>RX 7000 series (RDNA 3) - Latest architecture</li>
 *   <li>RX 6000 series (RDNA 2) - Recommended for inference</li>
 *   <li>RX 5000 series (RDNA) - Good performance</li>
 *   <li>Radeon VII, Vega series (GCN 5) - Older but supported</li>
 *   <li>MI100/MI200 series (CDNA) - Data center GPUs</li>
 * </ul>
 *
 * <p><b>ROCm Requirements:</b>
 * <ul>
 *   <li>ROCm 5.0 or later installed</li>
 *   <li>MIOpen library for GPU kernels</li>
 *   <li>rocBLAS library for linear algebra</li>
 *   <li>Compatible AMD GPU driver</li>
 * </ul>
 *
 * <p><b>Environment Variables:</b>
 * <ul>
 *   <li>{@code HIP_VISIBLE_DEVICES} - Specify which GPU to use (0, 1, etc.)</li>
 *   <li>{@code MIOPEN_LOG_LEVEL} - Set MIOpen log level for debugging</li>
 *   <li>{@code HSA_OVERRIDE_GFX_VERSION} - Override GPU architecture detection</li>
 * </ul>
 *
 * <p><b>Troubleshooting:</b>
 * <ul>
 *   <li>Verify ROCm installation: {@code rocm-smi}</li>
 *   <li>Check GPU visibility: {@code rocm-smi --showproductname}</li>
 *   <li>Test ROCm: {@code rocminfo | grep "Name:"}</li>
 * </ul>
 */
public class RocmGpuAdapter implements GpuAdapter {

  private static final Logger logger = LoggerFactory.getLogger(RocmGpuAdapter.class);

  @Override
  public OrtProvider getProviderType() {
    return OrtProvider.ROCM;
  }

  @Override
  public boolean isAvailable() {
    try {
      boolean available = OrtEnvironment.getAvailableProviders().contains(OrtProvider.ROCM);

      if (available) {
        // Additional check: verify ROCm runtime is accessible
        if (isRocmRuntimeAvailable()) {
          logger.debug("ROCm execution provider is available");
          return true;
        } else {
          logger.warn("ROCm provider detected but runtime not accessible. " +
              "Ensure ROCm 5.0+ is installed and GPU drivers are loaded.");
          return false;
        }
      }

      logger.debug("ROCm execution provider not available");
      return false;

    } catch (Exception e) {
      logger.debug("ROCm availability check failed: {}", e.getMessage());
      return false;
    }
  }

  @Override
  public void configure(OrtSession.SessionOptions options, int deviceId) throws OrtException {
    if (deviceId < 0) {
      throw new IllegalArgumentException("Device ID must be >= 0, got: " + deviceId);
    }

    try {
      // Set GPU device via environment variable if multi-GPU system
      if (deviceId > 0) {
        logger.info("ROCm: Using GPU device {} (set HIP_VISIBLE_DEVICES={} for explicit control)",
            deviceId, deviceId);
        // Note: Java cannot set environment variables for the current process,
        // user must set HIP_VISIBLE_DEVICES before launching JVM
      }

      // Add ROCm execution provider
      options.addROCM();
      logger.info("ROCm execution provider added (device {})", deviceId);

      // Log ROCm configuration hints
      logRocmConfigurationHints();

    } catch (OrtException e) {
      logger.error("Failed to add ROCm execution provider", e);
      throw new OrtException("ROCm configuration failed: " + e.getMessage() +
          ". Ensure ROCm 5.0+ is installed and GPU drivers are loaded.");
    }
  }

  @Override
  public String getName() {
    return "AMD ROCm";
  }

  @Override
  public String getDeviceInfo(int deviceId) {
    try {
      // Try to get GPU info from rocm-smi
      Process process = Runtime.getRuntime().exec(
          new String[]{"rocm-smi", "--showproductname", "--showmeminfo", "vram", "--gpu", String.valueOf(deviceId)});

      BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
      StringBuilder info = new StringBuilder();
      String line;

      while ((line = reader.readLine()) != null) {
        if (line.contains("GPU") || line.contains("Card") || line.contains("Memory")) {
          info.append(line.trim()).append("; ");
        }
      }

      reader.close();
      process.waitFor();

      if (info.length() > 0) {
        return "ROCm Device " + deviceId + ": " + info.toString();
      }

    } catch (Exception e) {
      logger.debug("Could not get ROCm device info from rocm-smi: {}", e.getMessage());

      // Fallback: try rocminfo
      try {
        Process process = Runtime.getRuntime().exec(new String[]{"rocminfo"});
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line;
        int gpuIndex = 0;
        StringBuilder gpuInfo = new StringBuilder();

        while ((line = reader.readLine()) != null) {
          if (line.contains("Name:") && gpuIndex == deviceId) {
            gpuInfo.append(line.trim());
          }
          if (line.contains("Agent ") && line.contains("GPU")) {
            gpuIndex++;
          }
        }

        reader.close();
        process.waitFor();

        if (gpuInfo.length() > 0) {
          return "ROCm Device " + deviceId + ": " + gpuInfo.toString();
        }

      } catch (Exception e2) {
        logger.debug("Could not get ROCm device info from rocminfo: {}", e2.getMessage());
      }
    }

    return "ROCm Device " + deviceId + " (info unavailable - run 'rocm-smi' to verify)";
  }

  @Override
  public int getDeviceCount() {
    try {
      // Count GPUs using rocm-smi
      Process process = Runtime.getRuntime().exec(new String[]{"rocm-smi", "--showid"});
      BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

      int count = 0;
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.contains("GPU[") || line.matches(".*GPU.*\\d+.*")) {
          count++;
        }
      }

      reader.close();
      process.waitFor();

      if (count > 0) {
        return count;
      }

    } catch (Exception e) {
      logger.debug("Could not get ROCm device count: {}", e.getMessage());
    }

    // Fallback: check HIP_VISIBLE_DEVICES environment variable
    String visibleDevices = System.getenv("HIP_VISIBLE_DEVICES");
    if (visibleDevices != null && !visibleDevices.isEmpty()) {
      String[] devices = visibleDevices.split(",");
      return devices.length;
    }

    return 1; // Assume at least one GPU if ROCm is available
  }

  @Override
  public int getRecommendedArenaSize() {
    // 512MB is recommended for ROCm (AMD GPUs typically have more VRAM than NVIDIA)
    return 512;
  }

  @Override
  public boolean supportsFp16() {
    // ROCm supports FP16 on RDNA and newer architectures
    // RDNA (gfx1010+), RDNA 2 (gfx1030+), RDNA 3 (gfx1100+)
    // Also supported on CDNA (MI100/MI200)
    return true;
  }

  @Override
  public boolean supportsInt8() {
    return true;
  }

  /**
   * Checks if ROCm runtime is accessible on the system.
   *
   * @return true if rocm-smi or rocminfo command is available
   */
  private boolean isRocmRuntimeAvailable() {
    try {
      // Try rocm-smi first (most reliable)
      Process process = Runtime.getRuntime().exec(new String[]{"rocm-smi", "--version"});
      int exitCode = process.waitFor();
      return exitCode == 0;

    } catch (Exception e) {
      logger.debug("rocm-smi not found, trying rocminfo");

      try {
        // Fallback to rocminfo
        Process process = Runtime.getRuntime().exec(new String[]{"rocminfo"});
        int exitCode = process.waitFor();
        return exitCode == 0;

      } catch (Exception e2) {
        logger.debug("ROCm runtime commands not available");
        return false;
      }
    }
  }

  /**
   * Logs ROCm configuration hints for users.
   */
  private void logRocmConfigurationHints() {
    logger.debug("ROCm Configuration Hints:");
    logger.debug("  - Multi-GPU: Set HIP_VISIBLE_DEVICES=0,1,2 before launching JVM");
    logger.debug("  - Debugging: Set MIOPEN_LOG_LEVEL=3 for detailed logs");
    logger.debug("  - Verify setup: Run 'rocm-smi' to see GPU status");
    logger.debug("  - GPU info: Run 'rocminfo | grep Name' to see available GPUs");

    // Log environment variables if set
    String visibleDevices = System.getenv("HIP_VISIBLE_DEVICES");
    if (visibleDevices != null) {
      logger.info("HIP_VISIBLE_DEVICES={} (GPU visibility controlled)", visibleDevices);
    }

    String logLevel = System.getenv("MIOPEN_LOG_LEVEL");
    if (logLevel != null) {
      logger.info("MIOPEN_LOG_LEVEL={} (MIOpen logging enabled)", logLevel);
    }
  }
}
