package com.badu.ai.onnx.gpu;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Manages GPU device discovery, monitoring, and health checking.
 *
 * <p>Discovers available GPUs, queries their properties, and monitors
 * utilization and memory usage for load balancing decisions.
 *
 * <p><b>Supported GPU Types:</b>
 * <ul>
 *   <li>NVIDIA GPUs (via nvidia-smi)</li>
 *   <li>AMD GPUs (via rocm-smi) - future support</li>
 * </ul>
 *
 * <p><b>Usage Example:</b>
 * <pre>{@code
 * GpuDeviceManager manager = new GpuDeviceManager();
 * List<GpuDevice> gpus = manager.discoverGpus();
 *
 * for (GpuDevice gpu : gpus) {
 *     System.out.println(gpu.toDetailedString());
 * }
 * }</pre>
 */
public class GpuDeviceManager {

  private static final Logger logger = LoggerFactory.getLogger(GpuDeviceManager.class);

  private final GpuAdapter adapter;

  /**
   * Creates a GPU device manager with auto-detected GPU adapter.
   */
  public GpuDeviceManager() {
    this.adapter = GpuAdapterFactory.createAdapter();
  }

  /**
   * Creates a GPU device manager with specified adapter.
   *
   * @param adapter GPU adapter to use
   */
  public GpuDeviceManager(GpuAdapter adapter) {
    this.adapter = adapter;
  }

  /**
   * Discovers all available GPUs on the system.
   *
   * @return list of available GPUs
   */
  public List<GpuDevice> discoverGpus() {
    List<GpuDevice> gpus = new ArrayList<>();

    if (adapter == null || !adapter.isAvailable()) {
      logger.warn("GPU adapter {} not available", adapter != null ? adapter.getName() : "NONE");
      return gpus;
    }

    int deviceCount = adapter.getDeviceCount();
    logger.info("Discovered {} GPU(s) using {}", deviceCount, adapter.getName());

    for (int i = 0; i < deviceCount; i++) {
      try {
        GpuDevice gpu = queryGpuDevice(i);
        gpus.add(gpu);
        logger.debug("  {}", gpu.toDisplayString());
      } catch (Exception e) {
        logger.warn("Failed to query GPU {}: {}", i, e.getMessage());
      }
    }

    return gpus;
  }

  /**
   * Queries detailed information about a specific GPU.
   *
   * @param deviceId GPU device ID
   * @return GPU device information
   */
  public GpuDevice queryGpuDevice(int deviceId) {
    GpuDevice.GpuDeviceBuilder builder = GpuDevice.builder()
        .deviceId(deviceId)
        .adapterType(adapter.getName())
        .supportsFp16(adapter.supportsFp16())
        .supportsInt8(adapter.supportsInt8())
        .available(true);

    // Try to get detailed info from nvidia-smi
    if (adapter instanceof CudaGpuAdapter) {
      parseNvidiaSmi(deviceId, builder);
    }

    return builder.build();
  }

  /**
   * Updates GPU utilization and memory stats for an existing GPU device.
   *
   * @param gpu GPU device to update
   * @return updated GPU device
   */
  public GpuDevice updateGpuStats(GpuDevice gpu) {
    if (adapter instanceof CudaGpuAdapter) {
      return updateNvidiaGpuStats(gpu);
    }

    // Return unchanged if monitoring not supported
    return gpu;
  }

  /**
   * Parses nvidia-smi output to populate GPU device info.
   */
  private void parseNvidiaSmi(int deviceId, GpuDevice.GpuDeviceBuilder builder) {
    try {
      Process process = Runtime.getRuntime().exec(new String[]{
          "nvidia-smi",
          "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
          "--format=csv,noheader,nounits",
          "--id=" + deviceId
      });

      BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
      String line = reader.readLine();
      reader.close();
      process.waitFor();

      if (line != null && !line.isEmpty()) {
        // Parse CSV: name, driver_version, memory.total, memory.free, utilization.gpu
        String[] parts = line.split(",");
        if (parts.length >= 5) {
          builder.name(parts[0].trim());
          builder.driverVersion(parts[1].trim());
          builder.totalMemoryMB(parseLong(parts[2].trim()));
          builder.freeMemoryMB(parseLong(parts[3].trim()));
          builder.utilizationPercent(parseInt(parts[4].trim()));
        }
      }
    } catch (Exception e) {
      logger.debug("Could not parse nvidia-smi output for GPU {}: {}", deviceId, e.getMessage());
      // Use defaults
      builder.name("NVIDIA GPU " + deviceId);
      builder.driverVersion("Unknown");
      builder.totalMemoryMB(0);
      builder.freeMemoryMB(0);
      builder.utilizationPercent(0);
    }
  }

  /**
   * Updates NVIDIA GPU stats (utilization and memory).
   */
  private GpuDevice updateNvidiaGpuStats(GpuDevice gpu) {
    try {
      Process process = Runtime.getRuntime().exec(new String[]{
          "nvidia-smi",
          "--query-gpu=memory.free,utilization.gpu",
          "--format=csv,noheader,nounits",
          "--id=" + gpu.getDeviceId()
      });

      BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
      String line = reader.readLine();
      reader.close();
      process.waitFor();

      if (line != null && !line.isEmpty()) {
        String[] parts = line.split(",");
        if (parts.length >= 2) {
          long freeMemory = parseLong(parts[0].trim());
          int utilization = parseInt(parts[1].trim());

          return gpu.toBuilder()
              .freeMemoryMB(freeMemory)
              .utilizationPercent(utilization)
              .build();
        }
      }
    } catch (Exception e) {
      logger.debug("Could not update GPU {} stats: {}", gpu.getDeviceId(), e.getMessage());
    }

    return gpu;
  }

  /**
   * Finds the GPU with lowest utilization.
   *
   * @param gpus list of GPUs to search
   * @return GPU with lowest utilization, or null if list is empty
   */
  public GpuDevice findLeastLoadedGpu(List<GpuDevice> gpus) {
    return gpus.stream()
        .filter(GpuDevice::isAvailable)
        .min((a, b) -> Integer.compare(a.getUtilizationPercent(), b.getUtilizationPercent()))
        .orElse(null);
  }

  /**
   * Finds the GPU with most free memory.
   *
   * @param gpus list of GPUs to search
   * @return GPU with most free memory, or null if list is empty
   */
  public GpuDevice findGpuWithMostMemory(List<GpuDevice> gpus) {
    return gpus.stream()
        .filter(GpuDevice::isAvailable)
        .max((a, b) -> Long.compare(a.getFreeMemoryMB(), b.getFreeMemoryMB()))
        .orElse(null);
  }

  /**
   * Helper to parse long values safely.
   */
  private long parseLong(String value) {
    try {
      return Long.parseLong(value);
    } catch (NumberFormatException e) {
      return 0;
    }
  }

  /**
   * Helper to parse int values safely.
   */
  private int parseInt(String value) {
    try {
      return Integer.parseInt(value);
    } catch (NumberFormatException e) {
      return 0;
    }
  }

  /**
   * Gets the underlying GPU adapter.
   *
   * @return GPU adapter
   */
  public GpuAdapter getAdapter() {
    return adapter;
  }
}
