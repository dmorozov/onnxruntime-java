package com.badu.ai.onnx.gpu;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * GPU adapter for NVIDIA CUDA execution provider.
 *
 * <p>Supports NVIDIA GPUs with CUDA compute capability 6.0+ (Pascal and newer).
 *
 * <p><b>Supported GPUs:</b>
 * <ul>
 *   <li>RTX 40xx series (Ada Lovelace) - Recommended for FP16</li>
 *   <li>RTX 30xx series (Ampere) - Good FP16 support</li>
 *   <li>RTX 20xx series (Turing) - Tensor Cores available</li>
 *   <li>GTX 16xx series (Turing) - No Tensor Cores</li>
 *   <li>GTX 10xx series (Pascal) - Basic CUDA support</li>
 * </ul>
 */
public class CudaGpuAdapter implements GpuAdapter {

  private static final Logger logger = LoggerFactory.getLogger(CudaGpuAdapter.class);

  @Override
  public OrtProvider getProviderType() {
    return OrtProvider.CUDA;
  }

  @Override
  public boolean isAvailable() {
    try {
      // First check if CUDA provider is listed as available
      if (!OrtEnvironment.getAvailableProviders().contains(OrtProvider.CUDA)) {
        logger.debug("CUDA provider not available in ONNX Runtime");
        return false;
      }

      // Check if CUDA runtime libraries are actually available
      // This is a more thorough check that verifies the libraries can be loaded
      try {
        // Test if we can actually create a session with CUDA
        OrtSession.SessionOptions testOptions = new OrtSession.SessionOptions();
        testOptions.addCUDA();
        testOptions.close();
        logger.debug("CUDA runtime libraries verified");
        return true;
      } catch (OrtException e) {
        logger.warn("CUDA provider is listed but runtime libraries are missing: {}", e.getMessage());
        logger.info("Please install CUDA 12 runtime libraries (libcublasLt.so.12, libcudnn.so, etc.)");
        return false;
      }
    } catch (Exception e) {
      logger.debug("CUDA availability check failed: {}", e.getMessage());
      return false;
    }
  }

  @Override
  public void configure(OrtSession.SessionOptions options, int deviceId) throws OrtException {
    if (deviceId < 0) {
      throw new IllegalArgumentException("Device ID must be >= 0");
    }

    try {
      // ONNX Runtime Java API: addCUDA() for device 0, or addCUDA(deviceId) for specific device
      if (deviceId == 0) {
        options.addCUDA();
        logger.info("CUDA execution provider added (device 0)");
      } else {
        options.addCUDA(deviceId);
        logger.info("CUDA execution provider added (device {})", deviceId);
      }
    } catch (OrtException e) {
      logger.error("Failed to add CUDA execution provider", e);
      throw new OrtException("CUDA configuration failed: " + e.getMessage());
    }
  }

  @Override
  public String getName() {
    return "NVIDIA CUDA";
  }

  @Override
  public String getDeviceInfo(int deviceId) {
    try {
      // Try to get GPU info from nvidia-smi
      Process process = Runtime.getRuntime().exec(
          new String[]{"nvidia-smi", "--query-gpu=name,driver_version,memory.total",
              "--format=csv,noheader", "--id=" + deviceId});

      BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
      String info = reader.readLine();
      reader.close();
      process.waitFor();

      if (info != null && !info.isEmpty()) {
        return "CUDA Device " + deviceId + ": " + info;
      }
    } catch (Exception e) {
      logger.debug("Could not get CUDA device info: {}", e.getMessage());
    }

    return "CUDA Device " + deviceId + " (info unavailable)";
  }

  @Override
  public int getDeviceCount() {
    try {
      Process process = Runtime.getRuntime().exec(new String[]{"nvidia-smi", "--list-gpus"});
      BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

      int count = 0;
      while (reader.readLine() != null) {
        count++;
      }
      reader.close();
      process.waitFor();

      return count;
    } catch (Exception e) {
      logger.debug("Could not get CUDA device count: {}", e.getMessage());
      return 1; // Assume at least one GPU if CUDA is available
    }
  }

  @Override
  public int getRecommendedArenaSize() {
    // 256MB is a good starting point for CUDA
    return 256;
  }

  @Override
  public boolean supportsFp16() {
    // CUDA supports FP16 on compute capability 6.0+ (all modern GPUs)
    return true;
  }

  @Override
  public boolean supportsInt8() {
    return true;
  }

  /**
   * Check for required CUDA runtime libraries.
   * @return List of missing libraries, empty if all required libraries are found
   */
  public static List<String> checkRequiredLibraries() {
    List<String> missingLibraries = new ArrayList<>();

    // List of required CUDA 12 libraries for ONNX Runtime
    String[] requiredLibraries = {
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcudart.so.12",
        "libcufft.so.11",
        "libcurand.so.10"
    };

    // Common library paths
    String[] libraryPaths = {
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-12.0/lib64",
        "/usr/local/cuda-12.1/lib64",
        "/usr/local/cuda-12.2/lib64",
        "/usr/local/cuda-12.3/lib64",
        "/usr/local/cuda-12.4/lib64",
        "/opt/cuda/lib64"
    };

    for (String library : requiredLibraries) {
      boolean found = false;

      // Check LD_LIBRARY_PATH
      String ldPath = System.getenv("LD_LIBRARY_PATH");
      if (ldPath != null) {
        for (String path : ldPath.split(":")) {
          Path libPath = Paths.get(path, library);
          if (Files.exists(libPath)) {
            found = true;
            break;
          }
        }
      }

      // Check common paths
      if (!found) {
        for (String path : libraryPaths) {
          Path libPath = Paths.get(path, library);
          if (Files.exists(libPath)) {
            found = true;
            break;
          }
        }
      }

      if (!found) {
        // Check if base version exists (e.g., libcudnn.so.8 instead of libcudnn.so.9)
        String baseLib = library.substring(0, library.lastIndexOf('.'));
        boolean baseFound = false;
        for (String path : libraryPaths) {
          Path dir = Paths.get(path);
          if (Files.exists(dir) && Files.isDirectory(dir)) {
            try {
              baseFound = Files.list(dir)
                  .anyMatch(p -> p.getFileName().toString().startsWith(baseLib));
              if (baseFound) break;
            } catch (Exception e) {
              // Ignore
            }
          }
        }
        if (!baseFound) {
          missingLibraries.add(library);
        }
      }
    }

    return missingLibraries;
  }

  /**
   * Get diagnostic information about CUDA installation.
   */
  public static String getCudaDiagnostics() {
    StringBuilder sb = new StringBuilder();
    sb.append("CUDA Diagnostics:\n");

    // Check NVIDIA driver
    try {
      Process process = Runtime.getRuntime().exec(new String[]{"nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"});
      BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
      String driverVersion = reader.readLine();
      reader.close();
      process.waitFor();

      if (driverVersion != null && !driverVersion.isEmpty()) {
        sb.append("  NVIDIA Driver: ").append(driverVersion).append("\n");
      }
    } catch (Exception e) {
      sb.append("  NVIDIA Driver: Not detected\n");
    }

    // Check missing libraries
    List<String> missingLibs = checkRequiredLibraries();
    if (!missingLibs.isEmpty()) {
      sb.append("  Missing CUDA libraries:\n");
      for (String lib : missingLibs) {
        sb.append("    - ").append(lib).append("\n");
      }
      sb.append("\n  To fix: Install CUDA 12 toolkit or set LD_LIBRARY_PATH to CUDA libraries\n");
    } else {
      sb.append("  All required CUDA libraries found\n");
    }

    // Check LD_LIBRARY_PATH
    String ldPath = System.getenv("LD_LIBRARY_PATH");
    if (ldPath != null && !ldPath.isEmpty()) {
      sb.append("  LD_LIBRARY_PATH: ").append(ldPath).append("\n");
    } else {
      sb.append("  LD_LIBRARY_PATH: Not set\n");
    }

    return sb.toString();
  }
}
