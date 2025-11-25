## Project Panama FFM API Implementation for ONNX Runtime I/O Binding

**Context and Goal:**

The objective is to establish a secure, idiomatic, and high-performance native interoperability layer for the **ONNX Runtime (ORT)** within the JVM. This new, dedicated Java library will expose the critical **I/O Binding** functionality—currently only available in C++ and Python/C#—to eliminate high-latency PCIe data copies and enable zero-copy GPU inference for general ONNX models.[1] The solution must be built exclusively on the **Project Panama Foreign Function and Memory (FFM) API** (JEP 442, JDK 21+), bypassing the inherent risks and complexity of the legacy Java Native Interface (JNI).[2, 3]

**Mandate for Research and Planning:**

Provide a detailed, multi-phase implementation plan. The plan must cite the specific ONNX Runtime C API calls being mapped and leverage structural insights from existing ORT Java codebases for naming conventions and resource lifecycle management.

---

### Phase 1: FFM API Foundation and Native Interop Setup

**Task 1.1: `jextract` Tooling Analysis and C Header Identification**

1.  Identify the primary C header files within the core ONNX Runtime repository (`microsoft/onnxruntime` [4]) that contain the necessary function declarations for `OrtSession`, `OrtIoBinding`, `OrtMemoryInfo`, and `OrtValue`.
2.  Detail the process by which the `jextract` tool will be used to generate the initial Java `MethodHandle` bindings and `MemoryLayout` definitions from these headers.
3.  Explain how C/C++ pointers (`OrtSession*`, `OrtIoBinding*`, etc.) will be represented as managed `MemorySegment` objects in the Java layer, ensuring their lifetime is handled by a `ResourceScope` or `MemorySession`.[5]

**Task 1.2: OrtEnvironment and OrtSession Integration**

1.  Analyze the existing `ai.onnxruntime.OrtSession` and `ai.onnxruntime.OrtEnvironment` classes (from the main ORT Maven artifact [6]) to determine how they expose the native `OrtEnv*` and `OrtSession*` pointers (currently as `jlong` handles [6]).
2.  The FFM wrapper must create an adapter layer to safely retrieve the underlying native addresses and wrap them in a `MemorySegment`. Detail the safest conceptual Java code required to bridge the existing JNI-based `long` handle into a new, FFM-managed `MemorySegment` for use by the `OrtIoBinding` constructor.

---

### Phase 2: Core I/O Binding API Mapping

The core of the zero-copy solution involves mapping the C++ `Ort::IoBinding` structure.[1, 7] Define the Java FFM classes and corresponding native functions required:

**Task 2.1: `OrtMemoryInfo` Abstraction (Device Specification)**

1.  Create a Java class, `OrtDeviceMemorySpec`, that abstracts the native `Ort::MemoryInfo` structure.[1]
2.  Detail the FFM API call sequence required to instantiate the native memory specification for CUDA, specifically using the **`"Cuda"`** device type, **`OrtDeviceAllocator`**, and the required device ID (`deviceId=0`).[8, 9]
3.  Include a design for methods that support key GPU memory types: `createCuda()` (for VRAM residence) and `createCudaPinned()` (for host-accessible, page-locked memory transfer [8]).

**Task 2.2: `OrtDeviceTensor` Abstraction (GPU Data Structure)**

1.  Create a Java class, `OrtDeviceTensor`, to represent the native `Ort::Value` allocated directly on the device (GPU VRAM).[8]
2.  Detail the native ORT C API call (via FFM) required to use the `Ort::Allocator` associated with the `OrtDeviceMemorySpec` to allocate GPU memory for the tensor data structure, thus ensuring the data is born device-resident.[1, 8]
3.  Explain how FFM's `MemorySegment` will be used to facilitate fast, zero-copy data transfer from Java arrays/buffers into the pre-allocated GPU buffer address, leveraging the `CudaPinned` type if necessary for fast asynchronous transfers.[8, 10]

**Task 2.3: `OrtIoBinding` Implementation and Execution**

1.  Create the main Java class, `OrtIoBinding`, which must wrap the native `OrtIoBinding*` handle as a `MemorySegment`.
2.  Provide the exact FFM binding signatures for the two critical C++ functions [1, 7]:
    *   **Input Binding:** `bindInput(String name, OrtDeviceTensor inputTensor)`
    *   **Output Binding (Device Allocation):** `bindOutput(String name, OrtDeviceMemorySpec spec)` (Used for dynamic shapes, instructing ORT to allocate the output on the GPU [1]).
3.  Detail the FFM implementation of the final `run()` method, which calls the native `OrtSessionRun` using the I/O Binding handle.

---

### Phase 3: Structural Stability and Code Reference

**Task 3.1: Resource Lifecycle Management**

1.  Explain how the **Java `AutoCloseable` interface** will be implemented for `OrtDeviceMemorySpec`, `OrtDeviceTensor`, and `OrtIoBinding` to ensure the corresponding native ORT C API release functions (e.g., `OrtReleaseMemoryInfo`, `OrtReleaseIoBinding`) are called deterministically, preventing catastrophic GPU memory leaks.[11]

**Task 3.2: Structural Guidance from GenAI and Core ORT**

1.  Research the structure of the preview `onnxruntime-genai` Java API on GitHub.[12, 13] Specifically, analyze the purpose and structure of its utility class functions, such as `setCurrentGpuDeviceId` and `getCurrentGpuDeviceId`.[13]
2.  Use the presence and intent of these GenAI utility methods to guide the design of the device management functions in the new FFM library, ensuring that the new library aligns with existing high-performance ORT device selection paradigms, even if the underlying JNI is replaced with FFM.

---

**Deliverables:**

1.  A comprehensive, phase-by-phase **Technical Implementation Plan** clearly detailing the required FFM binding steps.
2.  Conceptual **Java FFM code structure** (similar to the previous sketch, but enriched with technical detail) for the `OrtDeviceMemorySpec`, `OrtDeviceTensor`, and `OrtIoBinding` classes.
3.  A **Rationale Section** explaining how this FFM approach guarantees higher stability and broader execution provider support (CUDA, ROCm, etc.) compared to a JNI solution.[8, 14]


References:
https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html
https://www.unlogged.io/post/foreign-function-and-memory-api---java-22
https://openjdk.org/jeps/442
https://github.com/microsoft/onnxruntime
https://saltmarch.com/insight/project-panamas-ffm-api-a-new-dawn-for-java-native-interfacing
