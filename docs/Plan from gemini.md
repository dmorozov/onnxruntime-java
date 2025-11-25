

# **Architectural Plan for ONNX Runtime I/O Binding in Java via Foreign Function & Memory API**

This report details the comprehensive technical strategy for developing an expert-level Java binding library that exposes the critical I/O Binding functionality of the ONNX Runtime (ORT) C API. The plan leverages the modern, safe, and efficient Foreign Function & Memory (FFM) API (Project Panama), addressing a significant performance gap in the existing Java ORT ecosystem for high-performance inference, particularly when utilizing GPU Execution Providers.

## **1\. Introduction and Performance Rationale**

### **1.1 The Crucial Need for I/O Binding in Java ORT**

The default mechanisms for handling data inputs and outputs in many current language bindings of the ONNX Runtime often involve implicit memory management and data movement. When utilizing non-CPU execution providers, such as CUDA or ROCm, the conventional process dictates that if inputs are not explicitly located on the target device prior to execution, the ORT framework must copy them from the host CPU memory as part of the invocation of the Run() function.1 Similarly, if the intended output is not pre-allocated on the device, ORT will assume the output is requested on the CPU and execute an expensive device-to-host copy as the final step of the Run() call.1

These necessary data transfers, particularly the Host-to-Device (H2D) and Device-to-Host (D2H) copies, consume substantial execution time, disproportionately impacting overall latency. This consumption of time can mistakenly lead users to believe that the core ORT execution is slow, when in reality, the majority of the elapsed time is dedicated to these memory copy operations.1 For large-scale models, such as contemporary Large Language Models (LLMs) running on specialized hardware, this transfer overhead becomes the primary performance bottleneck.

The I/O Binding mechanism was specifically introduced to mitigate this issue. It provides the ability to arrange inputs on the target device and pre-allocate outputs on the device before invoking Run().1 This capability is indispensable for achieving zero-copy inference, a prerequisite for maximizing the efficiency of GPU Execution Providers. Without an exposed I/O Binding feature, the Java ORT binding severely limits the potential performance gains available from acceleration hardware, restricting high-throughput application development.

### **1.2 Architectural Justification: FFM over JNI**

Historically, Java's primary method for interacting with native code was the Java Native Interface (JNI). While functional, JNI is widely recognized for requiring significant boilerplate, manual type mapping, and demanding highly granular, manual management of native memory lifetimes.3 This complexity frequently leads to code brittleness, difficult debugging, and the persistent risk of native memory leaks or JVM crashes.4

The Foreign Function & Memory (FFM) API, introduced as Project Panama, is designed to supersede JNI by offering a safer, more straightforward, and more efficient mechanism for interaction with foreign functions and memory.4 FFM achieves safety by integrating the management of native resources directly into the JVM’s architecture. Key components like Arena enable the client code to control the allocation and deterministic deallocation of foreign (off-heap) memory.6

By utilizing FFM, this library can provide safe, managed pointers to native memory using MemorySegment objects, effectively wrapping opaque ORT C pointers with Java safety guarantees.8 This is crucial for handling external device memory pointers, such as those originating from CUDA allocations, where the raw address (void\* in C) must be safely represented and passed into the native ORT functions.9 The adoption of FFM inherently addresses the brittleness and danger inherent in JNI, ensuring the high-performance binding remains robust and stable.5

## **2\. Phase I: Defining the Native Interface Layer and Utility (OrtNativeAPI)**

This initial phase establishes the foundational layer necessary for Java to locate and invoke the required functions within the core onnxruntime shared library.

### **2.1 FFM Infrastructure Setup and Native Library Loading**

The core of the FFM binding resides in a static utility class, designated OrtNativeAPI, which is responsible for initializing the FFM linking infrastructure. This setup involves obtaining the system's Linker instance and a SymbolLookup mechanism configured to search for symbols within the dynamically loaded onnxruntime library.7 Proper dynamic loading of the native library must precede the symbol lookup to ensure that functions like OrtCreateIoBinding and OrtRun are resolvable.

Once the library is loaded, the OrtNativeAPI performs a lookup for each required function name. The resulting native address (MemorySegment) is then used to create a strongly typed MethodHandle via the Linker.downcallHandle() method.10 The correctness of this process hinges on accurately defining the signature of the native C function using a FunctionDescriptor, which specifies the return type followed by the argument types using constants from java.lang.foreign.ValueLayout.6 This strict contract ensures type safety across the Java-native boundary.

### **2.2 Critical ORT C API Functions Required for I/O Binding**

The I/O binding implementation requires mapping several core C API functions that handle environment setup, session execution, resource creation, and data binding. All opaque C pointers used by ORT, such as OrtSession\*, OrtIoBinding\*, and OrtValue\*, are represented in FFM by ValueLayout.ADDRESS.10

The critical functions, and their required FFM descriptor mappings, are summarized below:

Table 1: Critical ONNX Runtime C API Mapping to FFM Function Descriptors

| ORT C Function | Purpose | FFM Return Layout | FFM Argument Layouts (Ordered) |
| :---- | :---- | :---- | :---- |
| OrtCreateIoBinding | Creates the binding resource. | ADDRESS (OrtStatus\*) | ADDRESS (OrtSession\*), ADDRESS (OrtIoBinding\*\*) |
| OrtBindInput | Binds an existing OrtValue as an input. | ADDRESS (OrtStatus\*) | ADDRESS (OrtIoBinding\*), ADDRESS (Name char\*), ADDRESS (OrtValue\*) |
| OrtBindOutputToDevice | Binds an output based on device memory location. | ADDRESS (OrtStatus\*) | ADDRESS (OrtIoBinding\*), ADDRESS (Name char\*), ADDRESS (OrtMemoryInfo\*) |
| OrtRun | Executes the session using the bound resources. | ADDRESS (OrtStatus\*) | ADDRESS (OrtSession\*), ADDRESS (OrtRunOptions\*), ADDRESS (OrtIoBinding\*), ADDRESS (OrtValue\*\*), LONG (Num Outputs) |
| SynchronizeBoundInputs | Ensures memory visibility before execution. | ADDRESS (OrtStatus\*) | ADDRESS (OrtIoBinding\*) |
| OrtGetErrorMessage | Retrieves detailed error information. | ADDRESS (const char\*) | ADDRESS (OrtStatus\*) |
| OrtReleaseIoBinding | Releases the binding resource. | VOID | ADDRESS (OrtIoBinding\*) |

### **2.3 The Necessity of Pointer-to-Pointer Mapping for Object Creation**

A significant architectural consideration, common across the ORT C API, is the function signature pattern used for object creation. Instead of returning the created object pointer directly, ORT functions like OrtCreateIoBinding typically return an error status (OrtStatus\*) while passing the actual pointer (OrtIoBinding\*) out through a pointer-to-a-pointer argument (e.g., OrtIoBinding\*\* out).9

This pattern requires a deliberate sequence of actions in the Java FFM wrapper layer:

1. A temporary native memory segment must be explicitly allocated in Java, sized precisely to hold a single native address (ValueLayout.ADDRESS), using a scoped memory manager like an Arena.9  
2. The address of this temporary segment is then passed as the out parameter to the ORT downcall handle.  
3. Upon return, the Java code must check the OrtStatus\* for errors and, if successful, read the newly written native address (long) from the temporary segment using an appropriate access operation.  
4. This resulting native address is the valid opaque pointer (OrtIoBinding\*) needed for future calls and lifetime management. The initial temporary segment allocated in step 1 must also be released when its scope closes.

Failing to implement this two-step process—allocating the memory segment for the output pointer and then dereferencing it after the downcall—would result in an inability to correctly retrieve the native handles created by the ORT core.

### **2.4 Mandatory Robust Error Handling via OrtStatus Translation**

The consistency of the ORT C API relies on returning an OrtStatus\* for nearly every operation to signal success or failure.11 A high-quality, production-ready Java binding must not ignore this status pointer.

The implementation requires a standardized utility function that wraps every ORT downcall handle invocation. This wrapper must immediately inspect the returned OrtStatus\* address. If the pointer is non-null, it signifies a failure. In the event of an error, the library must:

1. Call the OrtGetErrorMessage function, passing the failed OrtStatus\* pointer, to retrieve a descriptive string detailing the error cause.11  
2. Translate this failure into a robust, descriptive Java exception (e.g., OrtException), which encapsulates the error message and status code.  
3. Crucially, the function must then call the necessary ORT API release function, such as OrtReleaseStatus, to free the native status object itself, thereby preventing native memory leaks associated with transient error reporting structures.11

The process of checking, translating, and immediately releasing the OrtStatus object must be universal across all ORT C function wrappers to maintain stability and prevent silent native resource depletion.

## **3\. Phase II: Architecting Safe Native Resource Management (The OrtHandle Pattern)**

Safety in FFM is predicated on deterministic resource cleanup, particularly for long-lived native objects like sessions and I/O bindings. This is achieved through the use of an encapsulated handle pattern tied to FFM's memory management features.

### **3.1 The Managed Handle Hierarchy and AutoCloseable**

All ORT opaque pointers that manage resources—including OrtSession, OrtMemoryInfo, and the target OrtIoBinding—will be encapsulated within a Java class hierarchy inheriting from an abstract OrtHandle. This base class must implement the java.lang.AutoCloseable interface.

Resource management is primarily delegated to the FFM Arena. By utilizing a confined arena (Arena.ofConfined()) within a try-with-resources construct, the native memory associated with the resource objects can be guaranteed to be released upon exiting the block, eliminating the risk of resource leaks common in manual resource management.12 A confined arena ensures that all segments allocated within it are released when the arena itself is closed.12

### **3.2 FFM Reinterpretation and Cleanup Callback Implementation**

When an ORT function returns an allocated native pointer (e.g., after successful OrtCreateIoBinding), the resulting MemorySegment is typically a zero-length segment because the Java runtime has no inherent knowledge of the size or lifetime of the externally allocated region.13 This zero-length segment cannot be directly accessed, but it can be passed to other native functions.13

To integrate this native pointer into the Java managed ecosystem, the FFM API's MemorySegment::reinterpret method is employed.13 This method allows the library to:

1. Assign a symbolic size (e.g., 1 byte) to the segment, allowing it to be treated as a valid, addressable pointer handle.  
2. Associate the segment with an existing Arena, linking its lifetime to the arena's scope.  
3. Crucially, register a custom cleanup action (Consumer\<MemorySegment\>) that is invoked when the associated Arena is closed.13

For an OrtIoBinding handle, this cleanup action will execute the FFM downcall handle for OrtReleaseIoBinding(OrtIoBinding\* ptr). This mechanism ensures that when the Java application closes the Arena (or the try-with-resources block completes), the JVM automatically triggers the invocation of the native release function, guaranteeing deterministic native resource deallocation.13 This pattern establishes robust native resource integrity, far exceeding the safety and stability capabilities of traditional JNI development.

## **4\. Phase III: Implementing Core I/O Binding Functionality**

The successful implementation of I/O Binding requires handling two core object types: OrtMemoryInfo (defining device context) and OrtValue (defining tensor data structured on device memory).

### **4.1 Implementing OrtMemoryInfo Creation and Mapping**

The OrtMemoryInfo structure is fundamental, acting as the descriptor that tells ORT where a tensor resides—on which device, using which allocator, and what memory type.1

The Java library must provide a wrapper around the OrtCreateMemoryInfo C API function. This function requires the device name (e.g., "Cuda"), memory allocation type, device ID, and memory type.1

1. The Java wrapper accepts parameters defining the device context.  
2. String parameters (like the device name) must be converted into null-terminated native char\* segments, which are managed by the same allocation Arena as the OrtMemoryInfo handle itself.  
3. The downcall to OrtCreateMemoryInfo is executed, producing an OrtMemoryInfo\* handle.15  
4. This resulting opaque pointer is encapsulated in a managed OrtMemoryInfoHandle class, ensuring its automatic release via OrtReleaseMemoryInfo when the corresponding Arena closes.16

This managed memory information object is the essential key for binding tensors to GPU or other accelerated devices.17

### **4.2 Zero-Copy Input Binding (OrtValue from Device Memory)**

The primary goal of I/O Binding is to enable zero-copy inputs. This requires creating an OrtValue structure directly referencing memory that already resides on the device, bypassing the need for ORT to perform an internal H2D copy.1

This mechanism relies on external device memory management (e.g., a dedicated CUDA memory library in Java providing a raw address).

1. **Raw Address Acquisition:** The Java application obtains the raw memory address of the pre-allocated device buffer as a Java primitive long.18  
2. **FFM Segment Creation:** This raw address (long) is translated into a native segment using MemorySegment.ofAddress(long address, long byteSize, SegmentScope scope).8 This segment represents the contiguous device memory region. The associated scope should reflect the external management of the memory (i.e., it is not managed by the ORT Arena).  
3. **OrtValue Construction:** The native C API function OrtCreateTensorWithDataAsOrtValue is utilized.15 The inputs to this function are:  
   * The managed OrtMemoryInfo\* handle (identifying the device).  
   * The base address of the FFM segment (obtained via MemorySegment.address()) which is the raw device pointer.  
   * The total size of the tensor data in bytes.  
   * The tensor dimensions (passed as an allocated native array of longs).  
   * The element data type.  
4. The result is an OrtValue\* handle that points directly to the external device buffer. This handle is then passed to the OrtIoBinding.BindInput wrapper, which executes the native OrtBindInput function.1

This sequence establishes the zero-copy channel: the OrtValue is effectively a metadata wrapper around the device pointer, allowing ORT to operate on the device buffer without host intervention.19

### **4.3 Output Binding and Inference Execution**

Once inputs are bound, the outputs must be prepared. The C++ and C\# bindings support two modes for output binding.1

The Java library must expose wrappers for:

1. **Binding a Pre-allocated Output (OrtValue):** The user can provide a pre-allocated OrtValue (created similarly to the input, using a pre-allocated device buffer) if the output shape is known beforehand.  
2. **Binding to Device Only (Unknown Shape):** If the shape of the output tensor is only determined during runtime (common in generative models), the library must call OrtBindOutputToDevice. This function accepts the output name and the managed OrtMemoryInfo\* handle, instructing ORT to allocate device memory for the output internally during OrtRun.1

Finally, the inference is executed via the OrtSession.run(binding) method. This method executes the FFM downcall to OrtRun, passing the OrtIoBinding\* handle as the source of all input and output tensor information. The use of I/O Binding overrides the default tensor input mechanism, ensuring computation leverages the pre-arranged device memory locations.15

## **5\. Phase IV: Advanced FFM Techniques and Production Readiness**

Implementing I/O Binding for GPU acceleration introduces low-level operational concerns related to device concurrency that must be exposed and managed within the Java binding for production reliability.

### **5.1 Explicit Device Stream Synchronization for GPU Reliability**

In high-performance GPU programming, memory allocation and kernel execution are often managed asynchronously via device streams. When external systems (like a dedicated CUDA memory manager) allocate the device memory used for the OrtValue, they may operate on a different stream than the one used internally by the ORT Execution Provider (e.g., CUDA EP).11

The binding must acknowledge and address this potential stream divergence, which can lead to data races or visibility issues. ORT provides explicit C API functions for stream synchronization: SynchronizeBoundInputs and SynchronizeBoundOutputs.11

The library must expose these functions on the OrtIoBinding object. For applications utilizing CUDA, it is essential that the user calls SynchronizeBoundInputs immediately before initiating the OrtRun operation. This explicit synchronization ensures that any pending asynchronous operations that placed the data into the bound input buffers are completed and visible to the ORT execution stream before computation begins. Similarly, SynchronizeBoundOutputs should be called after OrtRun completes if the results are intended to be read or transferred by an external system operating on a separate stream.11 This control over synchronization is not optional but a requirement for robust, high-throughput GPU deployment using external memory allocation.

### **5.2 Integration with External GPU Memory Managers**

A crucial design decision is the strict separation of concerns: the ORT Java binding should provide the interface to ORT, but it should *not* attempt to manage the specifics of CUDA or ROCm device memory allocation itself. Implementing memory management for every potential Execution Provider (EP) would unnecessarily complicate the core ORT binding.

The library’s public API, therefore, must define an interface contract that accepts the essential parameters for zero-copy operation:

1. A raw native memory address (long) representing the start of the device buffer.  
2. The size of the buffer in bytes.  
3. The necessary tensor shape and element type information.

By accepting a primitive long address, the FFM implementation can reliably bridge the external GPU allocation system (e.g., JCUDA or another FFM-based CUDA library) to the ORT core. The FFM API allows creation of a MemorySegment from this raw address, providing the necessary void\* pointer format required by OrtCreateTensorWithDataAsOrtValue.8 This decoupling strategy maximizes portability and minimizes the library's scope, making it highly focused on ORT interoperability.

### **5.3 Library Build, Module Requirements, and Distribution Considerations**

The deployment of an FFM-based library necessitates attention to modern JVM constraints:

1. **Module and Runtime Requirements:** The consuming Java application must target a platform supporting the FFM API (Java 21 or later).4 Compilation and execution will require enabling preview features if the target Java version has not finalized FFM (e.g., using \--enable-preview and potentially \--add-modules jdk.incubator.foreign).20  
2. **Native Library Resolution:** The distribution strategy must account for the required native onnxruntime shared library (e.g., onnxruntime.dll on Windows, libonnxruntime.so on Linux).21 The Java library must include robust logic to locate and load this shared object dynamically so that the SymbolLookup can successfully resolve the addresses of the ORT C API functions.10 This often involves packaging native binaries within the JAR or providing platform-specific loader utilities.  
3. **FFM Handle Management:** The design ensures that all string conversions required for native calls (e.g., input and output names in OrtBindInput) utilize short-lived native segments allocated within a confined Arena. This adherence to FFM's safe memory patterns ensures transient native data does not leak.

## **6\. Conclusions and Recommendations**

The detailed architectural plan presented guarantees the successful implementation of the high-performance I/O Binding feature for ONNX Runtime in a stable Java environment, fulfilling the project mandate without resorting to the complexities of JNI.

The resulting Java library will embody a robust, layered architecture:

1. **Safety and Determinism:** Implementation of the OrtHandle pattern paired with FFM Arena lifetime management provides deterministic native resource release, effectively eliminating the common memory leak and stability risks associated with traditional native bindings.  
2. **Performance Optimization:** The core mechanism relies on creating OrtValue objects directly from externally managed raw device memory addresses (long), which is a necessary step to bypass the implicit host-to-device data copies mandated by standard ORT Java usage, thereby achieving true zero-copy inference acceleration.1  
3. **Operational Maturity:** By explicitly exposing and documenting the need for stream synchronization via SynchronizeBoundInputs and SynchronizeBoundOutputs, the library provides the necessary low-level control for integrating safely with high-performance, asynchronous GPU Execution Providers like CUDA.11

The recommendations for proceeding focus on rigorous implementation of the FFM pointer handling logic, particularly the two-step process of allocating temporary segments for capturing native output pointers, and the consistent wrapping of all downcalls to translate OrtStatus\* returns into idiomatic Java exceptions, ensuring the resulting library is production-ready and highly reliable.

#### **Works cited**

1. I/O Binding | onnxruntime, accessed November 24, 2025, [https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)  
2. Class OrtIoBinding \- ONNX Runtime, accessed November 24, 2025, [https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.OrtIoBinding.html](https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.OrtIoBinding.html)  
3. From JNI to FFM: The future of Java‑native interoperability \- IBM Developer, accessed November 24, 2025, [https://developer.ibm.com/articles/j-ffm/](https://developer.ibm.com/articles/j-ffm/)  
4. A Quick Intro To Java 21 Foreign Function and Memory (FFM) API \- Payara Server, accessed November 24, 2025, [https://payara.fish/blog/java-21-foreign-function-and-memory-api/](https://payara.fish/blog/java-21-foreign-function-and-memory-api/)  
5. 12 Foreign Function and Memory API \- Java \- Oracle Help Center, accessed November 24, 2025, [https://docs.oracle.com/en/java/javase/24/core/foreign-function-and-memory-api.html](https://docs.oracle.com/en/java/javase/24/core/foreign-function-and-memory-api.html)  
6. JEP 454: Foreign Function & Memory API \- OpenJDK, accessed November 24, 2025, [https://openjdk.org/jeps/454](https://openjdk.org/jeps/454)  
7. Java's FFM API:Foreign Function and Memory Access API | by ahmet erdem | Medium, accessed November 24, 2025, [https://medium.com/@ahmet.erdem/javas-ffm-api-foreign-function-and-memory-access-api-53d2a4f32d29](https://medium.com/@ahmet.erdem/javas-ffm-api-foreign-function-and-memory-access-api-53d2a4f32d29)  
8. MemorySegment (Java SE 24 & JDK 24\) \- Oracle Help Center, accessed November 24, 2025, [https://docs.oracle.com/en/java/javase/24/docs/api/java.base/java/lang/foreign/MemorySegment.html](https://docs.oracle.com/en/java/javase/24/docs/api/java.base/java/lang/foreign/MemorySegment.html)  
9. How to pass over a value pointer via java foreign memory api \- Stack Overflow, accessed November 24, 2025, [https://stackoverflow.com/questions/70816275/how-to-pass-over-a-value-pointer-via-java-foreign-memory-api](https://stackoverflow.com/questions/70816275/how-to-pass-over-a-value-pointer-via-java-foreign-memory-api)  
10. Going Native \- Foreign Function & Memory API (FFM) \- Leading EDJE, accessed November 24, 2025, [https://blog.leadingedje.com/post/goingnative/foreignfunctionandmemory.html](https://blog.leadingedje.com/post/goingnative/foreignfunctionandmemory.html)  
11. OrtApi Struct Reference \- ONNX Runtime, accessed November 24, 2025, [https://onnxruntime.ai/docs/api/c/struct\_ort\_api.html](https://onnxruntime.ai/docs/api/c/struct_ort_api.html)  
12. Java Foreign Function & Memory API (FFM API) \- HappyCoders.eu, accessed November 24, 2025, [https://www.happycoders.eu/java/foreign-function-memory-api/](https://www.happycoders.eu/java/foreign-function-memory-api/)  
13. Foreign Functions That Return Pointers \- Java \- Oracle Help Center, accessed November 24, 2025, [https://docs.oracle.com/en/java/javase/21/core/foreign-functions-that-return-pointers.html](https://docs.oracle.com/en/java/javase/21/core/foreign-functions-that-return-pointers.html)  
14. Global \- ONNX Runtime, accessed November 24, 2025, [https://onnxruntime.ai/docs/api/c/group\_\_\_global.html](https://onnxruntime.ai/docs/api/c/group___global.html)  
15. C | onnxruntime, accessed November 24, 2025, [https://onnxruntime.ai/docs/get-started/with-c.html](https://onnxruntime.ai/docs/get-started/with-c.html)  
16. Ort::MemoryInfo Struct Reference \- ONNX Runtime, accessed November 24, 2025, [https://onnxruntime.ai/docs/api/c/struct\_ort\_1\_1\_memory\_info.html](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_memory_info.html)  
17. Class OrtMemoryInfo \- ONNX Runtime, accessed November 24, 2025, [https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.OrtMemoryInfo.html](https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.OrtMemoryInfo.html)  
18. Class GPULazyCudaFreeMemoryManager \- Apache SystemDS, accessed November 24, 2025, [https://systemds.staged.apache.org/docs/2.0.0/api/java/org/apache/sysds/runtime/instructions/gpu/context/GPULazyCudaFreeMemoryManager.html](https://systemds.staged.apache.org/docs/2.0.0/api/java/org/apache/sysds/runtime/instructions/gpu/context/GPULazyCudaFreeMemoryManager.html)  
19. Ort::Value Struct Reference \- ONNX Runtime, accessed November 24, 2025, [https://onnxruntime.ai/docs/api/c/struct\_ort\_1\_1\_value.html](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_value.html)  
20. Guide to Java Project Panama | Baeldung, accessed November 24, 2025, [https://www.baeldung.com/java-project-panama](https://www.baeldung.com/java-project-panama)  
21. How to generate C API for onnxruntime on Linux \- Stack Overflow, accessed November 24, 2025, [https://stackoverflow.com/questions/62617626/how-to-generate-c-api-for-onnxruntime-on-linux](https://stackoverflow.com/questions/62617626/how-to-generate-c-api-for-onnxruntime-on-linux)