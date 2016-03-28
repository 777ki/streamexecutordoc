.. Using backticks indicates inline code.
.. default-role:: code

================================
StreamExecutor and libomptarget
================================


------------
Introduction
------------

**StreamExecutor** and **libomptarget** are libraries that are both meant to
solve the problem of providing runtime support for offloading computational
work to an accelerator device. The libomptarget library is already hosted
within the OpenMP LLVM subproject, and there is currently a proposal to create
another LLVM subproject containing StreamExecutor. To avoid maintaining
duplicate functionality in LLVM, it has further been proposed that
StreamExecutor implement its platform plugins as thin wrappers around
libomptarget RTL instances. This document explains why that proposal does not
work given the current APIs of the two libraries, and talks about cases where
it might make sense.

Despite the similarities between the two libraries, the libomptarget RTL API
does not support the notion of streams of execution so it cannot be used to
implement general StreamExecutor platforms.

If the libomptarget RTL interface is extended to support streams in the future,
it may then become feasible to implement StreamExecutor on top of libomptarget,
but even then there would still be a question of whether the amount of
duplicate code saved by having StreamExecutor call into libomptarget would be
enough to balance out the extra code that would be needed in StreamExecutor to
adapt the libomptarget API to work with its own API.

To take the example of CUDA, both libomptarget and StreamExecutor have code for
very similar wrappers around the CUDA driver API, but in each case this wrapper
code is just meant to adapt the CUDA driver API to the API of the wrapper
library. It would not make sense to have StreamExecutor use libomptarget’s CUDA
wrapper because then StreamExecutor would just have to add code to adapt from
libomptarget’s API rather than CUDA’s driver API. An extra layer of wrapping
would be added and no reduction in code size or complexity would be achieved.

On the other hand, if there are cases where a runtime library that doesn’t
support streams is exposed only as a libomptarget RTL instance, then it would
make sense for StreamExecutor to wrap the libomptarget implementation in order
to provide support for that platform. For cases like those, the StreamExecutor
implementation might insist that a `nullptr` is always passed for the stream
argument, or StreamExecutor might introduce other methods that don’t require a
stream argument. The StreamExecutor project would be very open to changes like
this.

For these reasons, it would make more sense at this time for StreamExecutor to
keep its current implementations of the CUDA and OpenCL platforms (which
support streams) rather than attempting to implement those platforms in terms
of libomptarget.

The sections below describe the similarities and differences between the two
library interfaces in more detail.


----------------------------------------
Comparison of runtime library interfaces
----------------------------------------

This section describes the parallels between the StreamExecutor platform plugin
interface and the libomptarget RTL interface, and explains the significant
differences that prevent StreamExecutor from implementing its platforms as thin
wrappers around libomptarget RTL targets.


Storing handles to device code
==============================
StreamExecutor's `KernelBase` and libomptarget's `__tgt_offload_entry` are both
types designed to hold references to device code loaded on a device. Both types
store the name of the kernel they point to and a handle that can be used to
refer to the loaded code. Additionally, the `KernelBase` class also stores the
number of arguments expected by the kernel it points to.

While it would be possible to have `KernelBase` store some of its data as a
`__tgt_offload_entry` internally, it would just add an extra layer of
abstraction and wouldn't simplify any code.


Referencing device code blobs
=============================
StreamExecutor's `MultiKernelLoaderSpec` and libomptarget's
`__tgt_device_image` types are both designed to be wrappers for `void *`
pointers to compiled device code.

Whereas a `MultiKernelLoaderSpec` instance only manages the code for a single
kernel function, a `__tgt_device_image` instance manages the code for any
number of device functions and global variables. An instance of
`MultiKernelLoaderSpec` can store code for the same kernel in several different
forms. In particular, this allows a `MultiKernelLoaderSpec` to hold several
different PTX versions of the code for different comput capabilities. In
contrast, a single `__tgt_device_image` stores only one binary blob that must
be loaded onto the device as a unit.  A `MultiKernelLoaderSpec` can reference a
file name rather than a memory pointer for its device code, whereas a
`__tgt_device_image` is restricted to referencing memory pointers.

A `MultiKernelLoaderSpec` keeps track of the name of its kernel and the number
of arguments that kernel takes. A `__tgt_device_image` keeps track of the names
of its kernels, but not the number of arguments they take.

In StreamExecutor terms, a `__tgt_device_image` is like a combination of
several `MultiKernelLoaderSpec` instances which all store their data in the
same format, and a corresponding set of `KernelBase` objects.

Both `MultiKernelLoaderSpec` and `__tgt_device_image` work best when their
instances are created by the compiler. The compiler can make sure the names of
the kernels and the number of arguments (in the case of
`MultiKernelLoaderSpec`) are set correctly. The compiler can also handle the
creation of the device code and can set up the pointers in the wrapper class to
point to that data.

The implementation of `__tgt_device_image` is already fully specified, so it
cannot be implemented in terms of `MultiKernelLoaderSpec`. It is conceivable
that `MultiKernelLoaderSpec` could be implemented as a set of
`__tgt_device_image` instances with an additional field to keep track of the
number of kernel arguments, but this wouldn't support the case of kernel code
stored in a file. Even so, it doesn't seem like a good fit because
`__tgt_device_image` is just a handful of pointers and only a few of them would
be used by `MultiKernelLoaderSpec`.


Loading device code onto a device
=================================
StreamExecutor's `GetKernel` method and libomptarget's `__tgt_rtl_load_binary`
method are both used to load device code onto a device.

`GetKernel` takes a `MultiKernelLoaderSpec` and a `KernelBase` pointer, while
`__tgt_rtl_load_binary` takes an argument of the analogous type,
`__tgt_device_image`. The `GetKernel` method sets up its `KernelBase` argument
to be a proper handle to the loaded code, whereas the `__tgt_rtl_load_binary`
function returns a `tgt_target_table`, which is really just an array of
`__tgt_offload_entry`, so the return value is analogous to an array of
`KernelBase` objects. These two methods are very close analogs.

It may be possible to implement `GetKernel` in terms of
`__tgt_rtl_load_binary`.


Managing device memory
======================
StreamExecutor has `void *Allocate(size_t)` for allocating device memory and
`void Deallocate(DeviceMemoryBase *)` for deallocating device memory. The
analogous methods for libomptarget are `void *__tgt_rtl_data_alloc(int32_t
device_id, int64_t size)` and `int32_t __tgt_rtl_data_delete(int32_t device_id,
void *target_ptr)`. These functions are basically identical, and either set
could be implemented in terms of the others.

For copying data between the host and device, however, the functionality is not
so similar. StreamExecutor has `Memcpy(StreamInterface *, void *, const
DeviceMemoryBase &, size_t)` for copying from the host to the device, and
`Memcpy(StreamInterface *, DeviceMemoryBase &, const void *, size_t)` for
copying from the device to the host. On the other hand, libomptarget has
`int32_t __tgt_rtl_data_submit(int32_t device_id, void *target_ptr, void
*host_ptr, int64_t size)` and `int32_t __tgt_rtl_data_retrieve(int32_t
device_id, void *host_ptr, void *target_ptr, int64_t size)`.

The single difference is that the StreamExecutor methods take a stream argument
and the libomptarget methods do not. This is an extremely important difference
because asynchronous data movement is a very important aspect of the
StreamExecutor interface and has a very large effect on program performance.
Without support for streams, it doesn't seem possible to implement the
StreamExecutor memory copying functions in terms of their libomptarget
counterparts.


Launching kernels on the device
===============================
StreamExecutor has the method `Launch(StreamInterface *, const ThreadDim &,
const BlockDim &, const KernelBase &, KernelArgsArrayBase)` and libomptarget
has `__tgt_rtl_run_target_team_region` which takes the device ID, a handle for
the device code on the device, an array of pointers to the kernel arguments,
the number of teams, and the number of threads.

The arguments are basically the same except that the StreamExecutor method
again takes a stream parameter, which allows for overlapping compute and data
motion. Just as in the case of memory copy, this prevents the StreamExecutor
kernel launch function from being implemented in terms of its libomptarget
counterpart.
