.. default-role:: code

=======================================================
StreamExecutor Plugin Interfaces
=======================================================
This is a sketch of the platform plugin interface in StreamExecutor. A
developer who wants to support a new platform must implement the classes
defined in the **Interface Types** section.


Basic Types
===========
These types are defined here in order to create the vocabulary needed for
defining the interfaces below.

`Status`
    An object that signals whether an operation succeeded, and contains error
    information if the operation failed.

`ThreadDim`
    Dimensions of thread collection over which a parallel operation is run.
    Really just three integers specifying three dimension sizes.

`BlockDim`
    Same as `ThreadDim` but for blocks of threads.

`DeviceMemoryBase`
    Wrapper type for a raw device memory pointer. Has methods to get the number
    of bytes held at the address, and to get the raw pointer itself.

`KernelBase`
    Holds a pointer to a `KernelInterface` (defined below). See `GetKernel`
    below for details of how such an object is created.

`KernelArgsArrayBase`
    An object that holds all the arguments to be passed to a kernel. It has a
    method to get the number of arguments, and a method to get a pointer to the
    array of argument addresses.

`MultiKernelLoaderSpec`
    An object that knows where the compiled device code for a given kernel is
    stored, and in which format. Supports device code stored in a file by
    storing the name of the file. Supports device code stored in memory by
    storing a pointer to the memory. The *Multi* in the name expresses the fact
    that it can store different memory pointers and file names for the same
    kernel because it might store code for the same kernel in several different
    formats or compiled for different platforms (e.g. CUDA and OpenCL). See
    `GetKernel` below for details of how an object of this type is used to load
    a kernel.


Interface Types
=================


-----------------
StreamInterface
-----------------
Opaque handle for a single stream corresponding to a specific device. Each
platform-specific implementation will store its own host representation of a
stream. For example, the CUDA implementation stores a `CUstream` as defined in
the CUDA driver API.


-----------------
KernelInterface
-----------------
Opaque handle to device code for a single kernel loaded on a specific device.
Each platform-specific implementation will store its own host representation of
a kernel. For example, the CUDA implementation stores a `CUmodule` and a
`CUfunction` as defined in the CUDA driver API.


-------------------------
StreamExecutorInterface
-------------------------
An object that manages a single accelerator device. In all the methods below,
an implementation-specific `StreamExecutorInterface` can dig into the
implementation-specific details of the `StreamInterface` and `KernelInterface`
objects it deals with. So, for instance, when a CUDA `StreamInterface` is asked
to launch a kernel and is passed a `StreamInterface` and a `KernelInterface` it
can reach inside those objects to get the `CUstream`, `CUmodule`, and
`CUfunction` instances they contain.

Methods
--------
`int PlatformDeviceCount()`
    Gets the number of devices this StreamExecutor can manage.

`Status Init(int device_ordinal)`
    Takes an device ordinal integer and initializes the StreamExecutor to
    manage the device with that number. For CUDA this involves creating a
    context on the device.

`Status GetKernel(const MultiKernelLoaderSpec &spec, KernelBase *kernel)`
    Loads the device code specified by `spec` onto the device managed by this
    StreamExecutor and sets up the kernel object pointed to by `kernel` to be a
    handle for the loaded device code. The `MultiKernelLoaderSpec` basically
    provides a `void*` pointer to the compiled device code, so the
    implementation of this method has to handle the loading of a binary blob
    onto the device and storing a handled to that loaded blob in a `KernelBase`
    instance.

`StreamInterface *GetStreamImplementation()`
    Returns a new instance of a `StreamInterface` for this executor.

`void *Allocate(size_t size)`
    Allocates the given number of bytes on the device.

`void Deallocate(DeviceMemoryBase *mem)`
    Deallocates the memory on the device at this address.

`Status Launch(StreamInterface *s, const ThreadDim &t, const BlockDim &b, const KernelBase &k, KernelArgsArrayBase &args)`
    Launches the kernel pointed to by `k` on the stream `s` with thread a block
    dimensions given by `t` and `b`, respectively, and passing args specified
    by `args`.

`Status BlockHostUntilDone(StreamInterface *s)`
    Waits until all activity on the given stream is completed.

`Status SynchronizeAllActivity()`
    Waits until all activity on this device is completed.

`Status Memcpy(StreamInterface *s, void *host_dst, const DeviceMemoryBase &device_src, size_t size)`
    Copies data from device to host.

`Status Memcpy(StreamInterface *s, DeviceMemoryBase &device_dst, const void *host_src, size_t size)`
    Copies data from host to device.

`Status MemcpyDeviceToDevice(StreamInterface *s, DeviceMemoryBase *device_dst, const DeviceMemoryBase *device_src, size_t size)`
    Copies data from device to device.

`Status HostCallback(StreamInterface *s, std::function<void()> callback)`
    Executes a host function.
