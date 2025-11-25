

# **Enabling Zero-Copy GPU Acceleration in Java: An ONNX Runtime I/O Binding Implementation Strategy**

## **I. Executive Summary: The Criticality of Device-Local Memory in Java LLM Inference**

Enterprise Java environments are increasingly adopting large language models (LLMs) and other deep learning applications, necessitating robust and highly performant GPU inference capabilities directly within the Java Virtual Machine (JVM).1 Achieving the required sub-millisecond latency and maximal throughput relies fundamentally on minimizing data transfer overheads between the CPU host and the GPU device. This is the core function of I/O Binding in the ONNX Runtime (ORT) ecosystem.

The standard execution paradigm in heterogeneous systems mandates two synchronous memory copies for every inference call when inputs are supplied from the CPU: one copy from CPU $\\rightarrow$ GPU, and another copy for the result, GPU $\\rightarrow$ CPU.2 This round trip across the high-latency Peripheral Component Interconnect Express (PCIe) bus often constitutes the majority of the inference time, leading users to mistakenly conclude that ORT itself is slow.2

To circumvent this bottleneck, the ORT native C++ and subsequent Python and C\# bindings expose the I/O Binding feature, which allows inputs and outputs to be pre-arranged on the target device.2 Critically, the current official ORT Java binding does not publicly expose this essential functionality.3 Consequently, achieving zero-copy inference in Java requires immediate development of a custom extension. The most expedient path involves creating a specialized Java Native Interface (JNI) bridge to expose the underlying C++ Ort::IoBinding and its associated memory structures. For long-term stability and maintainability, however, strategic plans must prioritize migration to Project Panama’s Foreign Function and Memory (FFM) API, which offers a safer, more idiomatic, and performance-enhanced alternative to the brittle JNI mechanism.4

## **II. Architectural Deep Dive: ONNX Runtime I/O Binding Mechanisms**

The implementation strategy for Java must be driven by a precise understanding of how ORT manages device memory allocation and binding in its native C++ core, as this C++ structure is the target for the JNI wrapper.

### **2.1 Understanding Host-Device Memory Bottlenecks**

High-performance AI pipelines rely on maintaining data locality on the GPU as much as possible.6 Modern dedicated GPUs often boast memory bandwidths up to 1 terabyte per second (TB/s). In stark contrast, the common interconnect, such as PCIe 4.0 x16, is limited to approximately 32 gigabytes per second (GB/s). This several-order-of-magnitude difference means that every unnecessary data copy across the PCIe bus introduces significant latency.6

For complex, iterative tasks, such as those found in Large Language Models (LLMs) or diffusion networks, intermediate tensors (like the Key-Value cache in transformers) must be updated and reused in subsequent inference steps.6 If I/O Binding is not utilized, the output of step $N$ is copied back to the CPU, processed, and then copied back to the GPU as the input for step $N+1$. This wasteful round-trip dramatically reduces the throughput of generative workloads. I/O Binding eliminates this latency by ensuring all necessary data, including intermediate states, remains resident in high-bandwidth GPU memory (VRAM).

The shift from standard inference to fully bound I/O inference is characterized by the following latency profile:

Table 1: Comparison of Data Transfer Latency for GPU Inference

| Inference Mode | Input Data Location | Output Data Location | Memory Operation | Performance Implication |
| :---- | :---- | :---- | :---- | :---- |
| Standard ORT (Java/CPU Tensor) | CPU (JVM Heap/Off-heap) | CPU (JVM Heap/Off-heap) | CPU $\\rightarrow$ PCIe $\\rightarrow$ GPU $\\rightarrow$ PCIe $\\rightarrow$ CPU | High latency due to two synchronous PCIe transfers per Run() call. |
| I/O Binding (Input Only) | GPU Device Memory | CPU (JVM Heap/Off-heap) | GPU $\\rightarrow$ PCIe $\\rightarrow$ CPU | Eliminates input copy; output copy remains. |
| I/O Binding (Input & Output) | GPU Device Memory | GPU Device Memory | Zero Host-Device Copy | Near-optimal latency; data remains local to the GPU for iterative LLM/Diffusion tasks.6 |

### **2.2 Analysis of the Native C++ API (The Target Specification)**

The Java JNI implementation must successfully mirror the core capabilities provided by the C++ Ort::IoBinding class.7 The workflow begins by instantiating the binding associated with a specific inference session: Ort::IoBinding io\_binding{session}.2

#### **Binding Inputs and Pre-Allocated Tensors**

For input binding, the C++ API requires an existing device-resident tensor, achieved using io\_binding.BindInput("input1", input\_tensor).2 This implies that the Java implementation must first provide a mechanism for the user to allocate an Ort::Value (representing the tensor data structure) directly within the GPU’s VRAM, often using the session's internal Ort::Allocator.2 The Java wrapper, therefore, needs to manage the opaque native pointer to this pre-allocated Ort::Value.

#### **Handling Dynamic Output Shapes via Ort::MemoryInfo**

A critical feature of I/O Binding is its ability to handle outputs where the shape is not known in advance, a common scenario in modern dynamic LLM pipelines.2 The native ORT API addresses this by binding a memory specification instead of a pre-allocated tensor. This is achieved by binding an Ort::MemoryInfo object: io\_binding.BindOutput("output1", output\_mem\_info).2

The Ort::MemoryInfo object, such as output\_mem\_info{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault}, serves as the blueprint, instructing the ORT session on where and how to allocate the output tensor when the shape is determined during execution.2 By using OrtDeviceAllocator, ORT can leverage optimized allocation strategies (such as arena allocation) on the GPU device, ensuring optimal memory utilization and reduced fragmentation.6 This mechanism ensures that the session allocates the tensor automatically on the specified device, and crucially, does not copy the result back to the CPU.

### **2.3 CUDA Memory Types and ORT Configuration**

The Ort::MemoryInfo structure must allow the Java user to specify the required CUDA memory type for optimal performance.6

1. **Cuda Memory:** This refers to pure device VRAM, which is essential for maximizing computational speed. Tensors bound to this memory type remain on the GPU for computation and iterative reuse, such as the LLM KV cache update.  
2. **CudaPinned Memory:** This type designates page-locked host memory. While it is resident on the CPU host, it is directly accessible by the GPU, enabling fast, fully asynchronous transfers (CUDA host-to-device).6 This is highly beneficial when the application absolutely requires the data on the CPU after processing but still wishes to utilize the fastest possible transfer method.

The Java API must provide methods that accurately map to the C++ constructor parameters for Ort::MemoryInfo, allowing precise control over the allocation strategy.

## **III. Implementation Roadmap: Creating the Custom Java JNI Bridge**

The most pragmatic approach for immediate functionality is to build a JNI extension that bridges the established C++ I/O Binding APIs to Java. This requires careful management of native pointers and rigorous adherence to safety protocols.

### **3.1 JNI Architecture and Native Pointer Management**

The existing onnxruntime Java API relies heavily on JNI, with its source structure reflecting a clear separation between Java wrappers and C++ native glue code.9 The fundamental technique for bridging complex C++ objects to the JVM involves representing the native C++ pointer (e.g., Ort::IoBinding\*) as an opaque long integer (jlong) within a corresponding Java class (e.g., OrtIoBinding).10

The Java object then acts as a safe, managed proxy. It must implement the java.lang.AutoCloseable interface to ensure deterministic cleanup. Failure to call close() on the Java object leaves the native resource (which could be a large chunk of GPU VRAM) unfreed, leading to memory exhaustion and system instability—a key risk of the JNI architecture.11

### **3.2 Design of the Java Public Interfaces**

The Java extension requires three main wrapper classes:

1. **OrtIoBinding:** This class wraps the native Ort::IoBinding\* handle. Its constructor will call a JNI function that takes the native handle of the OrtSession (jlong) and returns the newly allocated Ort::IoBinding\* address as a jlong. It will expose bindInput and bindOutput methods.  
2. **OrtMemoryInfo:** This class encapsulates the configuration parameters (device name, allocator type, device ID) required by ORT. Its constructor must call a native function to instantiate the Ort::MemoryInfo struct in C++ and return its address as a jlong for subsequent use in the OrtIoBinding.bindOutput method.  
3. **OrtDeviceTensor:** This class will represent an Ort::Value that is resident on the GPU. It must provide methods to retrieve its native pointer (jlong) for binding purposes and to manage its allocation and eventual destruction.

### **3.3 C++ Native Implementation (JNI Glue Code)**

The C++ glue layer must perform several critical operations to maintain stability and functionality:

* **Pointer Retrieval and Casting:** All native functions receive the Java jlong pointers and must safely cast them back to their native C++ types using reinterpret\_cast (e.g., Ort::Session\* p\_session \= reinterpret\_cast\<Ort::Session\*\>(j\_session\_handle)).  
* **Implementing Bindings:** The native method nativeBindOutputToDevice must retrieve the native Ort::IoBinding\* and the Ort::MemoryInfo\* handles, and then execute the core C++ ORT call: p\_io\_binding-\>BindOutput(name\_cstr, \*p\_memory\_info).  
* **JNI Error Handling:** Due to JNI’s inherent fragility, robust error handling is paramount. Any runtime error originating in the native ORT library (e.g., failed GPU allocation or invalid device configuration) must be caught in the C++ layer and translated into a proper Java exception using JNI functions like ThrowNew. Failure to intercept native exceptions can lead to unchecked resource leakage or immediate JVM crashes.12

### **3.4 The Crux of Device Tensor Management**

For true zero-copy input, the data must be allocated directly into VRAM. Standard JNI techniques utilizing ByteBuffer.allocateDirect() followed by GetDirectBufferAddress() 13 only guarantee access to off-heap CPU memory. While this is sufficient for CudaPinned memory transfers, it is insufficient for tensors required to be resident in GPU VRAM (Type Cuda) where the address is not directly accessible by the JVM.

Therefore, the Java extension requires a native allocation function that accepts the OrtMemoryInfo configuration and tensor dimensions, invokes the native ORT allocator (Ort::Allocator gpu\_allocator...), constructs the Ort::Value on the GPU, and returns the C++ pointer as a jlong to the Java OrtDeviceTensor object.2 This guarantees that the input tensor is GPU-resident from creation, making the subsequent I/O Binding operation a true zero-copy execution.

The implementation requires a distinct JNI function for each binding strategy, reflecting the underlying C++ complexity, as shown below:

Table 2: Proposed JNI Implementation Mapping for I/O Binding

| Native C++ API Component | Target JNI Wrapper (C++ Glue) | Proposed Java Public Signature | Purpose |
| :---- | :---- | :---- | :---- |
| Ort::IoBinding{session} | JNIEXPORT jlong JNICALL nativeCreateIoBinding | public native long createIoBinding(OrtSession session) | Instantiate native I/O Binding object handle. |
| Ort::Value (Device Tensor) | N/A (Pointer passed as jlong) | public class OrtDeviceTensor | Java proxy for native GPU tensor data structure. |
| Ort::MemoryInfo | N/A (Pointer passed as jlong) | public class OrtMemoryInfo | Java proxy holding Cuda/device allocation type.2 |
| io\_binding.BindInput(name, value) | JNIEXPORT void JNICALL nativeBindInput | public native void bindInput(String name, OrtDeviceTensor input) | Binds an existing GPU-resident tensor as input. |
| io\_binding.BindOutput(name, mem\_info) | JNIEXPORT void JNICALL nativeBindOutputToDevice | public native void bindOutput(String name, OrtMemoryInfo outputSpec) | Instructs ORT to allocate output directly on the device.2 |
| session.Run(run\_options, io\_binding) | JNIEXPORT jobject JNICALL nativeRunIoBinding | public native OrtOutput run(OrtIoBinding ioBinding) | Executes inference using device bindings. |

## **IV. Existing High-Level Solutions and Architectural Reference**

While developing a custom JNI binding provides generalized low-level control, the analysis must consider existing specialized libraries that may already incorporate this optimization internally.

### **4.1 ONNX Runtime GenAI: The LLM Specialist**

Microsoft's onnxruntime-genai library offers a highly focused, vertically-integrated solution specifically designed for running Large Language Models (LLMs) and multi-modal models.14 This library implements the entire generative AI loop, including pre- and post-processing, KV cache management, logits processing, and sampling.14

The Java API for GenAI is currently in preview, confirming ongoing investment in high-performance JVM machine learning acceleration.15 The API exposes device management capabilities through its Utils class, including methods like setCurrentGpuDeviceId() and getCurrentGpuDeviceId().15 This explicit device awareness serves as strong circumstantial evidence that the library implements necessary memory optimizations, such as I/O Binding or equivalent zero-copy techniques, internally.

### **4.2 LLM Pipeline Efficiency**

The efficiency requirements for LLMs are particularly demanding due to the iterative nature of token generation. The speed of the generation loop hinges on the ability to update the KV cache—a substantial intermediate tensor—with minimal latency. For onnxruntime-genai to deliver its promised performance, it is architecturally mandated to ensure that this KV cache tensor remains resident on the GPU across all decoding steps.6 This guarantees that the output of one step is immediately available as an input for the next, without incurring the performance penalty of PCIe data copies.

### **4.3 Strategic Conclusion on Existing Libraries**

If the primary application is confined to established LLM architectures (Llama, Mistral, Phi, etc.), onnxruntime-genai represents the preferred deployment path. It delivers the zero-copy performance benefits required for high-throughput generation while abstracting the complexities and inherent dangers of manual JNI resource management from the application developer.

However, for enterprise architects seeking to integrate ORT with existing external GPU frameworks (e.g., custom CUDA kernels for image processing or specialized proprietary pre-processing pipelines), the generic I/O Binding API remains indispensable. The custom JNI extension described in Section III is necessary for applications requiring this low-level memory control that falls outside the specialized domain of onnxruntime-genai.16

## **V. Strategic Forward Look: Migration to Project Panama (FFM API)**

While the custom JNI extension addresses the immediate performance requirement, its implementation introduces significant technical debt and stability risks.12 The long-term strategy for high-performance Java ML integration must focus on migrating to the modern native interoperation standard provided by Project Panama.

### **5.1 Addressing the JNI Deficiencies**

The long-standing Java Native Interface (JNI) is known to be complex, requiring substantial expertise in both Java and native programming to write stable "glue" code.4 JNI necessitates verbose boilerplate code, manual C/C++ resource management, and complex cross-platform deployment due to the need for platform-specific shared libraries (.so, .dll) for every deployment target.5 Furthermore, JNI’s approach to memory management is manual and error-prone, carrying a high risk of resource leaks (especially un-freed GPU VRAM allocated in C++) and system crashes caused by unsafe pointer access.11

### **5.2 Project Panama: The Future of Native Interoperability**

Project Panama, specifically the Foreign Function and Memory (FFM) API (JEP 442, delivered in JDK 21 17), is the intended successor to JNI.4 FFM dramatically improves native interoperability by providing a safer, more efficient, and more idiomatic Java interface to foreign code and data.12

For ORT and CUDA integration, the FFM API offers several transformative advantages over JNI:

* **Safe Memory Access:** FFM introduces the MemorySegment abstraction, which represents native memory (off-heap memory).12 Unlike raw jlong pointers, these segments are managed by the JVM, incorporating runtime bounds checking and integration with JVM resource management. This fundamentally mitigates the risk of memory corruption and native crashes, which is critical when handling sensitive GPU memory addresses.4  
* **Automated Bindings:** The jextract tool associated with Project Panama can mechanically derive Java interfaces directly from C/C++ headers.5 This eliminates the need for developers to manually write complex, fragile C++ JNI glue code for every ORT C API function, accelerating development and improving reliability.  
* **Resource Lifecycle Management:** FFM facilitates resource management using MemorySession and other modern Java practices, ensuring that native resources—including those allocated on the GPU device—are properly tracked and released without the manual risk associated with JNI’s jlong handles.12

### **5.3 CUDA Interoperability and Pipeline Unification**

The core architectural benefit of FFM for this use case is its capability to safely represent and manipulate native memory addresses.5 This provides a clean path for integrating ORT not just with host memory, but with external native GPU allocations. For instance, a sophisticated application pipeline could use the FFM API to invoke a custom CUDA kernel, which returns a native VRAM pointer. FFM can wrap this pointer as a safe MemorySegment. This same MemorySegment can then be passed directly to the FFM-wrapped ORT I/O Binding input function. This eliminates all manual, dangerous pointer handling and unifies the GPU acceleration pipeline entirely within the managed context of the JVM, maximizing throughput and reducing development overhead.

Table 3: Architectural Comparison: JNI vs. FFM API for Native Interoperability

| Feature | Java Native Interface (JNI) | Project Panama Foreign Function and Memory (FFM) API | Implication for ORT/CUDA |
| :---- | :---- | :---- | :---- |
| **Memory Access** | Manual (raw pointers via jlong), unsafe, requires manual bounds checking.12 | Safe, idiomatic MemorySegment/MemoryLayout, JVM-managed lifetime/bounds checks.4 | Eliminates memory corruption risks associated with raw C++ pointers. |
| **Development** | High complexity, verbose C/C++ "glue" code, error-prone.5 | Simplified bindings via jextract tool, native code looks more like pure Java calls.5 | Reduces development and maintenance overhead dramatically. |
| **Performance** | Overhead on frequent calls; reliance on data copy functions.12 | Near native-speed calls; highly optimized memory access and calling conventions. | Essential for high-frequency operations common in LLM generation loops. |
| **Resource Management** | Manual close() required; high risk of native memory leaks (JNI memory footprint not GC'd).10 | Automatic resource management via MemorySession and modern JVM practices.12 | Ensures proper deallocation of ORT/CUDA resources allocated outside the JVM. |

## **VI. Conclusions and Recommendations**

The absence of public I/O Binding API exposure in the standard ONNX Runtime Java bindings is a critical deficiency that currently forces performance-critical JVM applications to endure high-latency PCIe data copies, thus hindering LLM and deep learning throughput.

1. **Immediate Action: Custom JNI Extension:** The performance imperative requires the immediate development of a custom JNI extension. This extension must faithfully wrap the C++ Ort::IoBinding and Ort::MemoryInfo structures, utilizing the jlong mechanism to pass and manage opaque native pointers to device-resident data structures. Special attention must be paid to implementing robust JNI exception handling and deterministic resource cleanup (via AutoCloseable) to manage the inherent fragility of this architecture.  
2. **Specialized LLM Use Case:** For applications focused strictly on generative AI, the use of the preview onnxruntime-genai Java API is recommended. This library provides internal, high-performance optimizations (equivalent to I/O Binding for iterative loops) and abstracts the complex native memory management, offering a higher degree of stability for specific tasks.14  
3. **Long-Term Architectural Strategy:** Due to the complexity, safety hazards, and maintenance burden associated with JNI, the implementation of an FFM API wrapper for ORT should be prioritized as the long-term architectural standard. Project Panama offers the required safety, performance, and automation via MemorySegment and jextract to manage external GPU resources securely and efficiently within the modern JVM, ensuring future scalability and maintainability of Java-based machine learning pipelines.4

#### **Works cited**

1. Bringing AI Inference to Java with ONNX: a Practical Guide for Enterprise Architects \- InfoQ, accessed November 23, 2025, [https://www.infoq.com/articles/onnx-ai-inference-with-java/](https://www.infoq.com/articles/onnx-ai-inference-with-java/)  
2. I/O Binding | onnxruntime, accessed November 23, 2025, [https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)  
3. Java | onnxruntime, accessed November 23, 2025, [https://onnxruntime.ai/docs/get-started/with-java.html](https://onnxruntime.ai/docs/get-started/with-java.html)  
4. Project Panama's FFM API: A New Dawn for Java-Native Interfacing \- Saltmarch, accessed November 23, 2025, [https://saltmarch.com/insight/project-panamas-ffm-api-a-new-dawn-for-java-native-interfacing](https://saltmarch.com/insight/project-panamas-ffm-api-a-new-dawn-for-java-native-interfacing)  
5. Foreign Function and Memory API \- Java 22 \- Unlogged, accessed November 23, 2025, [https://www.unlogged.io/post/foreign-function-and-memory-api---java-22](https://www.unlogged.io/post/foreign-function-and-memory-api---java-22)  
6. Using device tensors in ONNX Runtime, accessed November 23, 2025, [https://onnxruntime.ai/docs/performance/device-tensor.html](https://onnxruntime.ai/docs/performance/device-tensor.html)  
7. Ort::IoBinding Struct Reference \- ONNX Runtime, accessed November 23, 2025, [https://onnxruntime.ai/docs/api/c/struct\_ort\_1\_1\_io\_binding.html](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_io_binding.html)  
8. CUDA Runtime API \- 6.10. Memory Management \- NVIDIA Docs Hub, accessed November 23, 2025, [https://docs.nvidia.com/cuda/cuda-runtime-api/group\_\_CUDART\_\_MEMORY.html](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)  
9. java · main · mvg / mvg-oss / Onnxruntime \- GitLab, accessed November 23, 2025, [https://gitlab.ispras.ru/mvg/mvg-oss/onnxruntime/-/tree/main/java?ref\_type=heads](https://gitlab.ispras.ru/mvg/mvg-oss/onnxruntime/-/tree/main/java?ref_type=heads)  
10. Return function pointer from C++ behavior to Java object with JNI \- Stack Overflow, accessed November 23, 2025, [https://stackoverflow.com/questions/53301963/return-function-pointer-from-c-behavior-to-java-object-with-jni](https://stackoverflow.com/questions/53301963/return-function-pointer-from-c-behavior-to-java-object-with-jni)  
11. Passing a pointer from JNI to Java using a long \- Stack Overflow, accessed November 23, 2025, [https://stackoverflow.com/questions/5802340/passing-a-pointer-from-jni-to-java-using-a-long](https://stackoverflow.com/questions/5802340/passing-a-pointer-from-jni-to-java-using-a-long)  
12. Foreign Function & Memory API \- Previous FOSDEM Editions, accessed November 23, 2025, [https://archive.fosdem.org/2024/events/attachments/fosdem-2024-1714-foreign-function-memory-api/slides/22193/fosdem\_2024\_FtLDvIv.pdf](https://archive.fosdem.org/2024/events/attachments/fosdem-2024-1714-foreign-function-memory-api/slides/22193/fosdem_2024_FtLDvIv.pdf)  
13. SharedBuffer between Java and C++ using JNI \- Stack Overflow, accessed November 23, 2025, [https://stackoverflow.com/questions/44246338/sharedbuffer-between-java-and-c-using-jni](https://stackoverflow.com/questions/44246338/sharedbuffer-between-java-and-c-using-jni)  
14. microsoft/onnxruntime-genai: Generative AI extensions for onnxruntime \- GitHub, accessed November 23, 2025, [https://github.com/microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai)  
15. Java API | onnxruntime, accessed November 23, 2025, [https://onnxruntime.ai/docs/genai/api/java.html](https://onnxruntime.ai/docs/genai/api/java.html)  
16. onnxruntime \- ONNX Runtime, accessed November 23, 2025, [https://onnxruntime.ai/docs/](https://onnxruntime.ai/docs/)  
17. JEP 442: Foreign Function & Memory API (Third Preview) \- OpenJDK, accessed November 23, 2025, [https://openjdk.org/jeps/442](https://openjdk.org/jeps/442)