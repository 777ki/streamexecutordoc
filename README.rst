.. Using backticks indicates inline code.
.. default-role:: code

At Google we're doing a lot of work on parallel programming models for CPUs, GPUs and other platforms. One place where we're investing a lot are parallel libraries, especially those closely tied to compiler technology like runtime and math libraries. We would like to develop these in the open, and the natural place seems to be as a subproject in LLVM if others in the community are interested.

Initially, we'd like to open source our StreamExecutor runtime library, which is used for simplifying the management of data-parallel workflows on accelerator devices and can also be extended to support other hardware platforms. We'd like to teach Clang to use StreamExecutor when targeting CUDA and work on other integrations, but that makes much more sense if it is part of the LLVM project.

However, we think the LLVM subproject should be organized as a set of several libraries with StreamExecutor as just the first instance. As just one example of how creating a unified parallelism subproject could help with code sharing, the StreamExecutor library contains some nice wrappers around the CUDA driver API and OpenCL API that create a unified API for managing all kinds of GPU devices. This unified GPU wrapper would be broadly applicable for libraries that need to communicate with GPU devices.

Of course, there is already an LLVM subproject for a parallel runtime library: OpenMP! So there is a question of how it would fit into this picture.  Eventually, it might make sense to pull in the OpenMP project as a library in this proposed new subproject. In particular, there is a good chance that OpenMP and StreamExecutor could share code for offloading to GPUs and managing workloads on those devices. This is discussed at the end of the StreamExecutor documentation below. However, if it turns out that the needs of OpenMP are too specialized to fit well in a generic parallelism project, then it may make sense to leave OpenMP as a separate LLVM subproject so it can focus on serving the particular needs of OpenMP.

Documentation for the StreamExecutor library that is being proposed for open-sourcing is included below to give a sense of what it is, in order to give context for how it might fit into a general parallelism LLVM subproject.

What do folks think? Is there general interest in something like this? If so, we can start working on getting a project in place and sketching out a skeleton for how it would be organized, as well as contributing StreamExecutor to it. We're happy to iterate on the particulars to figure out what works for the community.


=============================================
StreamExecutor Runtime Library Documentation
=============================================


What is StreamExecutor?
========================

**StreamExecutor** is a unified wrapper around the **CUDA** and **OpenCL** host-side programming models (runtimes). It lets host code target either CUDA or OpenCL devices with identically-functioning data-parallel kernels. StreamExecutor manages the execution of concurrent work targeting the accelerator similarly to how an Executor_ from the Google APIs client library manages the execution of concurrent work on the host.

.. _Executor: http://google.github.io/google-api-cpp-client/latest/doxygen/classgoogleapis_1_1thread_1_1Executor.html

StreamExecutor is currently used as the runtime for the vast majority of Google's internal GPGPU applications, and a snapshot of it is included in the open-source TensorFlow_ project, where it serves as the GPGPU runtime.

.. _TensorFlow: https://www.tensorflow.org

It is currently proposed that StreamExecutor itself be independently open-sourced. As part of that proposal, this document describes the basics of its design and explains why it would fit in well as an LLVM subproject.


-------------------
Key points
-------------------

StreamExecutor:

* abstracts the underlying accelerator platform (avoids locking you into a single vendor, and lets you write code without thinking about which platform you'll be running on).
* provides an open-source alternative to the CUDA runtime library.
* gives users a stream management model whose terminology matches that of the CUDA programming model.
* makes use of modern C++ to create a safe, efficient, easy-to-use programming interface.

StreamExecutor makes it easy to:

* move data between host and accelerator (and also between peer accelerators).
* execute data-parallel kernels written in the OpenCL or CUDA kernel languages.
* inspect the capabilities of a GPU-like device at runtime.
* manage multiple devices.


--------------------------------
Example code snippet
--------------------------------

The StreamExecutor API uses abstractions that will be familiar to those who have worked with other GPU APIs: **Streams**, **Timers**, and **Kernels**. Its API is *fluent*, meaning that it allows the user to chain together a sequence of related operations on a stream, as in the following code snippet:

.. code-block:: c++

  se::Stream stream(executor);
  se::Timer timer(executor);
  stream.InitWithTimer(&timer)
      .ThenStartTimer(&timer)
      .ThenLaunch(se::ThreadDim(dim_block_x, dim_block_y),
                  se::BlockDim(dim_grid_x, dim_grid_y),
                  my_kernel,
                  arg0, arg1, arg2)
      .ThenStopTimer(&timer)
      .BlockHostUntilDone();

The name of the kernel being launched in the snippet above is `my_kernel` and the arguments being passed to the kernel are `arg0`, `arg1`, and `arg2`. Kernels with any number of arguments of any types are supported, and the number and types of the arguments is checked at compile time.

How does it work?
=======================


--------------------------------
Detailed example
--------------------------------

The following example shows how we can use StreamExecutor to create a `TypedKernel` instance, associate device code with that instance, and then use that instance to schedule work on an accelerator device.

.. code-block:: c++

    #include <cassert>

    #include "stream_executor.h"

    namespace se = streamexecutor;

    // A PTX string defining a CUDA kernel.
    //
    // This PTX string represents a kernel that takes two arguments: an input value
    // and an output pointer. The input value is a floating point number. The output
    // value is a pointer to a floating point value in device memory. The output
    // pointer is where the output from the kernel will be written.
    //
    // The kernel adds a fixed floating point value to the input and writes the
    // result to the output location.
    static constexpr const char *KERNEL_PTX = R"(
        .version 3.1
        .target sm_20
        .address_size 64
        .visible .entry add_mystery_value(
            .param .f32 float_literal,
            .param .u64 result_loc
            ) {
          .reg .u64 %rl<2>;
          .reg .f32 %f<2>;
          ld.param.f32 %f1, [float_literal];
          ld.param.u64 %rl1, [result_loc];
          add.f32 %f1, %f1, 123.0;
          st.f32 [%rl1], %f1;
          ret;
        }
        )";

    // The number of arguments expected by the kernel described in
    // KERNEL_PTX_TEMPLATE.
    static constexpr int KERNEL_ARITY = 2;

    // The name of the kernel described in KERNEL_PTX.
    static constexpr const char *KERNEL_NAME = "add_mystery_value";

    // The value added to the input in the kernel described in KERNEL_PTX.
    static constexpr float MYSTERY_VALUE = 123.0f;

    int main(int argc, char *argv[]) {
      // Get a CUDA Platform object. (Other platforms such as OpenCL are also
      // supported.)
      se::Platform *platform =
          se::MultiPlatformManager::PlatformWithName("cuda").ValueOrDie();

      // Get a StreamExecutor for the chosen Platform. Multiple devices are
      // supported, we indicate here that we want to run on device 0.
      const int device_ordinal = 0;
      se::StreamExecutor *executor =
          platform->ExecutorForDevice(device_ordinal).ValueOrDie();

      // Create a MultiKernelLoaderSpec, which knows where to find the code for our
      // kernel. In this case, the code is stored in memory as a PTX string.
      //
      // Note that the "arity" and name specified here must match  "arity" and name
      // of the kernel defined in the PTX string.
      se::MultiKernelLoaderSpec kernel_loader_spec(KERNEL_ARITY);
      kernel_loader_spec.AddCudaPtxInMemory(KERNEL_PTX, KERNEL_NAME);

      // Next create a kernel handle, which we will associate with our kernel code
      // (i.e., the PTX string).  The type of this handle is a bit verbose, so we
      // create an alias for it.
      //
      // This specific type represents a kernel that takes two arguments: a floating
      // point value and a pointer to a floating point value in device memory.
      //
      // A type like this is nice to have because it enables static type checking of
      // kernel arguments when we enqueue work on a stream.
      using KernelType = se::TypedKernel<float, se::DeviceMemory<float> *>;

      // Now instantiate an object of the specific kernel type we declared above.
      // The kernel object is not yet connected with the device code that we want it
      // to run (that happens with the call to GetKernel below), so it cannot be
      // used to execute work on the device yet.
      //
      // However, the kernel object is not completely empty when it is created. From
      // the StreamExecutor passed into its constructor it knows which platform it
      // is targeted for, and it also knows which device it will run on.
      KernelType kernel(executor);

      // Use the MultiKernelLoaderSpec defined above to load the kernel code onto
      // the device pointed to by the kernel object and to make that kernel object a
      // handle to the kernel code loaded on that device.
      //
      // The MultiKernelLoaderSpec may contain code for several different platforms,
      // but the kernel object has an associated platform, so there is no confusion
      // about which code should be loaded.
      //
      // After this call the kernel object can be used to launch its kernel on its
      // device.
      executor->GetKernel(kernel_loader_spec, &kernel);

      // Allocate memory in the device memory space to hold the result of the kernel
      // call. This memory will be freed when this object goes out of scope.
      se::ScopedDeviceMemory<float> result = executor->AllocateOwnedScalar<float>();

      // Create a stream on which to schedule device operations.
      se::Stream stream(executor);

      // Schedule a kernel launch on the new stream and block until the kernel
      // completes. The kernel call executes asynchronously on the device, so we
      // could do more work on the host before calling BlockHostUntilDone.
      const float kernel_input_argument = 42.5f;
      stream.Init()
          .ThenLaunch(se::ThreadDim(), se::BlockDim(), kernel,
                      kernel_input_argument, result.ptr())
          .BlockHostUntilDone();

      // Copy the result of the kernel call from device back to the host.
      float host_result = 0.0f;
      executor->SynchronousMemcpyD2H(result.cref(), sizeof(host_result),
                                     &host_result);

      // Verify that the correct result was computed.
      assert((kernel_input_argument + MYSTERY_VALUE) == host_result);
    }


--------------------------------
Kernel Loader Specs
--------------------------------

An instance of the class `MultiKernelLoaderSpec` is used to encapsulate knowledge of where the device code for a kernel is stored and what format it is in.  Given a `MultiKernelLoaderSpec` and an uninitialized `TypedKernel`, calling the `StreamExecutor::GetKernel` method will load the code onto the device and associate the `TypedKernel` instance with that loaded code. So, in order to initialize a `TypedKernel` instance, it is first necessary to create a `MultiKernelLoaderSpec`.

A `MultiKernelLoaderSpec` supports a different method for adding device code
for each combination of platform, format, and storage location. The following
table shows some examples:

===========     =======         ===========     =========================
Platform        Format          Location        Setter
===========     =======         ===========     =========================
CUDA            PTX             disk            `AddCudaPtxOnDisk`
CUDA            PTX             memory          `AddCudaPtxInMemory`
CUDA            cubin           disk            `AddCudaCubinOnDisk`
CUDA            cubin           memory          `AddCudaCubinInMemory`
OpenCL          text            disk            `AddOpenCLTextOnDisk`
OpenCL          text            memory          `AddOpenCLTextInMemory`
OpenCL          binary          disk            `AddOpenCLBinaryOnDisk`
OpenCL          binary          memory          `AddOpenCLBinaryInMemory`
===========     =======         ===========     =========================

The specific method used in the example is `AddCudaPtxInMemory`, but all other methods are used similarly.


------------------------------------
Compiler Support for StreamExecutor
------------------------------------


General strategies
-------------------

For illustrative purposes, the PTX code in the example is written by hand and appears as a string literal in the source code file, but it is far more typical for the kernel code to be expressed in a high level language like CUDA C++ or OpenCL C and for the device machine code to be generated by a compiler.

There are several ways we can load compiled device code using StreamExecutor.

One possibility is that the build system could write the compiled device code to a file on disk. This can then be added to a `MultiKernelLoaderSpec` by using one of the `OnDisk` setters.

Another option is to add a feature to the compiler which embeds the compiled device code into the host executable and provides some symbol (probably with a name based on the name of the kernel) that allows the host code to refer to the embedded code data.

In fact, as discussed below, in the current use of StreamExecutor inside Google, the compiler goes even further and generates an instance of `MultiKernelLoaderSpec` for each kernel. This means the application author doesn't have to know anything about how or where the compiler decided to store the compiled device code, but instead gets a pre-made loader object that handles all those details.


Compiler-generated code makes things safe
--------------------------------------------

Two of the steps in the example above are dangerous because they lack static safety checks: instantiating the `MultiKernelLoaderSpec` and specializing the `TypedKernel` class template. This section discusses how compiler support for StreamExecutor can make these steps safe.

Instantiating a `MultiKernelLoaderSpec` requires specifying a three things:

1. the kernel *arity* (number of parameters),
2. the kernel name,
3. a string containing the device machine code for the kernel (either as assembly, or some sort of object file).

The problem with this is that the kernel name and the number of parameters is already fully determined by the kernel's machine code. In the best case scenario the *arity* and name arguments passed to the `MultiKernelLoaderSpec` methods match the information in the machine code and are simply redundant, but in the worst case these arguments contradict the information in the machine code and we get a runtime error when we try to load the kernel..

The second unsafe operation is specifying the kernel parameter types as type arguments to the `TypedKernel` class template. The specified types must match the types defined in the kernel machine code, but again there is no compile-time checking that these types match. Failure to match these types will result in a runtime error when the kernel is launched.

We would like the compiler to perform these checks for the application author, so as to eliminate this source of runtime errors. In particular, we want the compiler to create an appropriate `MultiKernelLoaderSpec` instance and `TypedKernel` specialization for each kernel definition.

One of the main goals of open-sourcing StreamExecutor is to let us add this code generation capability to Clang, when the user has chosen to use StreamExecutor as their runtime for accelerator operations.

Google has been using an internally developed CUDA compiler based on Clang called **gpucc** that generates code for StreamExecutor in this way.  The code below shows how the example above would be written using gpucc to generate the unsafe parts of the code.

The kernel is defined in a high-level language (CUDA C++ in this example) in its own file:

.. code-block:: c++

    // File: add_mystery_value.cu

    __global__ void add_mystery_value(float input, float *output) {
      *output = input + 42.0f;
    }

The host code is defined in another file:

.. code-block:: c++

    // File: example_host_code.cc

    #include <cassert>

    #include "stream_executor.h"

    // This header is generated by the gpucc compiler and it contains the
    // definitions of gpucc::kernel::AddMysteryValue and
    // gpucc::spec::add_mystery_value().
    //
    // The name of this header file is derived from the name of the file containing
    // the kernel code. The trailing ".cu" is replaced with ".gpu.h".
    #include "add_mystery_value.gpu.h"

    namespace se = streamexecutor;

    int main(int argc, char *argv[]) {
      se::Platform *platform =
          se::MultiPlatformManager::PlatformWithName("cuda").ValueOrDie();

      const int device_ordinal = 0;
      se::StreamExecutor *executor =
          platform->ExecutorForDevice(device_ordinal).ValueOrDie();

      // AddMysteryValue is an instance of TypedKernel generated by gpucc. The
      // template arguments are chosen by the compiler to match the parameters of
      // the add_mystery_value kernel.
      gpucc::kernel::AddMysteryValue kernel(executor);

      // gpucc::spec::add_mystery_value() is generated by gpucc. It returns a
      // MultiKernelLoaderSpec that knows how to find  the compiled code for the
      // add_mystery_value kernel.
      executor->GetKernel(gpucc::spec::add_mystery_value(), &kernel);

      se::ScopedDeviceMemory<float> result = executor->AllocateOwnedScalar<float>();
      se::Stream stream(executor);

      const float kernel_input_argument = 42.5f;

      stream.Init()
          .ThenLaunch(se::ThreadDim(), se::BlockDim(), kernel,
                      kernel_input_argument, result.ptr())
          .BlockHostUntilDone();

      float host_result = 0.0f;
      executor->SynchronousMemcpyD2H(result.cref(), sizeof(host_result),
                                     &host_result);

      assert((kernel_input_argument + 42.0f) == host_result);
    }

This support from the compiler makes the use of StreamExecutor safe and easy.


Compiler support for triple angle bracket kernel launches
----------------------------------------------------------

For even greater ease of use, Google's gpucc CUDA compiler also supports an integrated mode that looks like NVIDIA's `CUDA programming model`_,which uses triple angle brackets (`<<<>>>`) to launch kernels.

.. _CUDA programming model: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels

.. code-block:: c++
    :emphasize-lines: 22

    #include <cassert>

    #include "stream_executor.h"

    namespace se = streamexecutor;

    __global__ void add_mystery_value(float input, float *output) {
      *output = input + 42.0f;
    }

    int main(int argc, char *argv[]) {
      se::Platform *platform =
          se::MultiPlatformManager::PlatformWithName("cuda").ValueOrDie();

      const int device_ordinal = 0;
      se::StreamExecutor *executor =
          platform->ExecutorForDevice(device_ordinal).ValueOrDie();

      se::ScopedDeviceMemory<float> result = executor->AllocateOwnedScalar<float>();

      const float kernel_input_argument = 42.5f;
      add_mystery_value<<<1, 1>>>(kernel_input_argument, *result.ptr());

      float host_result = 0.0f;
      executor->SynchronousMemcpyD2H(result.cref(), sizeof(host_result),
                                     &host_result);

      assert((kernel_input_argument + 42.0f) == host_result);
    }

Under the hood, gpucc converts the triple angle bracket kernel call into a series of calls to the StreamExecutor library similar to the calls seen in the previous examples.

Clang currently supports the triple angle bracket kernel call syntax for CUDA compilation by replacing a triple angle bracket call with calls to the NVIDIA CUDA runtime library, but it would be easy to add a compiler flag to tell Clang to emit calls to the StreamExecutor library instead. There are several benefits to supporting this mode of compilation in Clang:

.. _benefits-of-streamexecutor:

* StreamExecutor is a high-level, modern C++ API, so is easier to use and less prone to error than the NVIDIA CUDA runtime and the OpenCL runtime.
* StreamExecutor will be open-source software, so GPU code will not have to depend on opaque binary blobs like the NVIDIA CUDA runtime library.
* Using StreamExecutor as the runtime would allow for easy extension of the triple angle bracket kernel launch syntax to support different accelerator programming models.


Supporting other platforms
===========================

StreamExecutor currently supports CUDA and OpenCL platforms out-of-the-box, but it uses a platform plugin architecture that makes it easy to add new platforms at any time. The CUDA and OpenCL platforms are both implemented as platform plugins in this way, so they serve as good examples for future platform developers of how to write these kinds of plugins.


Canned operations
==================

StreamExecutor provides several predefined kernels for common data-parallel operations. The supported classes of operations are:

* BLAS: basic linear algebra subprograms,
* DNN: deep neural networks,
* FFT: fast Fourier transforms, and
* RNG: random number generation.

Here is an example of using a canned operation to perform random number generation:

.. code-block:: c++
    :emphasize-lines: 12-13,17,34-35

    #include <array>

    #include "cuda/cuda_rng.h"
    #include "stream_executor.h"

    namespace se = streamexecutor;

    int main(int argc, char *argv[]) {
      se::Platform *platform =
          se::MultiPlatformManager::PlatformWithName("cuda").ValueOrDie();

      se::PluginConfig plugin_config;
      plugin_config.SetRng(se::cuda::kCuRandPlugin);

      const int device_ordinal = 0;
      se::StreamExecutor *executor =
          platform->ExecutorForDeviceWithPluginConfig(device_ordinal, plugin_config)
              .ValueOrDie();

      const uint8 seed[] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7,
                            0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf};
      constexpr uint64 random_element_count = 1024;

      using HostArray = std::array<float, random_element_count>;

      HostArray host_memory;
      const size_t data_size = host_memory.size() * sizeof(HostArray::value_type);

      se::ScopedDeviceMemory<float> device_memory =
          executor->AllocateOwnedArray<float>(random_element_count);

      se::Stream stream(executor);
      stream.Init()
          .ThenSetRngSeed(seed, sizeof(seed))
          .ThenPopulateRandUniform(device_memory.ptr())
          .BlockHostUntilDone();

      executor->SynchronousMemcpyD2H(*device_memory.ptr(), data_size,
                                     host_memory.data());
    }

Each platform plugin can define its own canned operation plugins for these operations or choose to leave any of them unimplemented.


Contrast with OpenMP
=====================

Recent versions of OpenMP also provide a high-level, easy-to-use interface for running data-parallel workloads on an accelerator device. One big difference between OpenMP's approach and that of StreamExecutor is that OpenMP generates both the kernel code that runs on the device and the host-side code needed to launch the kernel, whereas StreamExecutor only generates the host-side code. While the OpenMP model provides the convenience of allowing the author to write their kernel code in standard C/C++, the StreamExecutor model allows for the use of any kernel language (e.g. CUDA C++ or OpenCL C). This lets authors use  platform-specific features that are only present in platform-specific kernel definition languages.

The philosophy of StreamExecutor is that performance is critical on the device, but less so on the host.  As a result, no attempt is made to use a high-level device abstraction during device code generation. Instead, the high-level abstraction provided by StreamExecutor is used only for the host-side code that moves data and launches kernels.  This host-side work is tedious and is not performance critical, so it benefits from being wrapped in a high-level library that can support a wide range of platforms in an easily extensible manner.


Cooperation with OpenMP
========================

The Clang OpenMP community is currently in the process of `designing their implementation`_ of offloading support. They will want the compiler to convert the various standardized target-oriented OpenMP pragmas into device code to execute on an accelerator and host code to load and run that device code. StreamExecutor may provide a convenient API for OpenMP to use to generate their host-side code.

.. _designing their implementation: https://drive.google.com/a/google.com/file/d/0B-jX56_FbGKRM21sYlNYVnB4eFk/view

In addition to the :ref:`benefits<benefits-of-streamexecutor>` that all users of StreamExecutor enjoy over the alternative host-side runtime libraries, OpenMP and StreamExecutor may mutually benefit by sharing work to support new platforms. If OpenMP makes use of StreamExecutor, then it should be simple for OpenMP to add support for any new platforms that StreamExecutor supports in the future. Similarly, for any platforms OpenMP would like to target, they may add that support in StreamExecutor and take advantage of the knowledge of platform support in the StreamExecutor community. The resulting new platform support would then be available not just within OpenMP, but also to any user of StreamExecutor.

Although OpenMP and StreamExecutor support different programming models, some of the work they perform under the hood will likely be very similar. By sharing code and domain expertise, both projects will be improved and strengthened as their capabilities are expanded. The StreamExecutor community looks forward to much collaboration and discussion with OpenMP about the best places and ways to cooperate.
