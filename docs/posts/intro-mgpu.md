# An Introduction to Pallas:MGPU
This is the first of a series of blog posts covering Pallas:MGPU, which is a custom kernel DSL developed by the JAX and XLA teams at Google, and is available for both JAX and PyTorch users. This blog post serves as an introductory primer, and subsequent posts will cover tutorials for various kernels and dive into more of the technical details behind the design of Pallas:MGPU.

## What is Pallas:MGPU?
Pallas:MGPU is a custom kernel DSL for modern (H100 and newer) GPUs. It is one of the platform-specific backends supported by the broader Pallas project, which also includes TPU and Triton as other possible backends to target.

The main goal of Pallas:MGPU is to allow engineers to write highly performant GPU kernels with the comfort and productivity of Python and JAX syntax. In terms of design philosophy, performance is the main goal of Pallas:MGPU, and abstractions are only made where it would improve usability but not affect performance. This means that while kernel writers will have to think about low-level hardware-specific features, such as memory spaces and synchronization, they can still write concise and information-dense code in Python that is easy to read and understandable. For those who are familiar with other GPU kernel languages, Pallas:MGPU exposes more hardware details than alternative offerings such as Triton and Helion, and is more similar in abstraction level to Gluon or CuteDSL.

On a more technical level, Pallas:MGPU primarily provides the following functionality:
- It allows users to write vectorized code at the block/warpgroup level, rather than thread level as done in CUDA, and enforces the necessary synchronization to ensure thread-safety.
- It abstracts away layouts, which are the way logical tensors are physically stored in registers and memory. In the future, Pallas:MGPU will automatically infer and optimize the layouts for you, but still allow the user to override any decisions the compiler makes.
- It provides a library of helpers for commonly used patterns, such as pipelining and partitioning work across cores.

## Pallas:MGPU programming basics
Programming in Pallas:MGPU is similar to programming in JAX, and involves using a mix of standard JAX library functions as well as GPU-specific operations. This means for the most part, you can use JAX ops such as those from jax.numpy for arithmetic, lax.cond or lax.while for control flow, etc. Here we’ll go over a quick guide on the basics of getting started, and then present a starter matrix multiplication kernel in the next section.
Kernel entry point: plgpu.kernel
Embedding a Pallas:MGPU kernel within a JAX program is easy and involves a couple of lines of setup code. The main entry point into Pallas:MGPU is the plgpu.kernel function. 

```python
import jax
from jax import numpy as jnp
import jax.experimental.pallas.mosaic_gpu as plgpu

def kernel_body(input_ref, out_ref):
  out_ref[...] = input_ref[...] + 1

inputs = jnp.ones((128,), jnp.float32)
result = plgpu.kernel(
  kernel_body,
  out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
)(inputs)
```

In this tiny example, we implement our kernel in the kernel_body function which loads an input, increments it by 1, and stores it to the output. We then wrap kernel_body with plgpu.kernel, which creates a function that accepts JAX arrays as inputs and returns the kernel result as an output. That’s it!

For PyTorch users, you will additionally need to add the plgpu.as_torch_kernel decorator around the kernel. Note that you will still need to install and import JAX as Pallas:MGPU uses JAX functions and tracing machinery, but the as_torch_kernel decorator will allow the kernel to consume and return PyTorch arrays.

### Grids, Clusters and Threads
When a kernel is launched, the body of the kernel will be run in parallel across multiple streaming multiprocessors (SMs) on the GPU. The grid argument of plgpu.kernel specifies how many logical blocks of work to map over SMs, and the block indices can be queried using lax.axis_index(grid_axis_name). scratch_shapes allows you to specify block-local resources such as shared memory (SMEM). Newer GPUs also introduce the concept of clusters, which are groups of adjacent SMs that can collaborate on certain operations or share resources. Clusters can be enabled by passing in the corresponding cluster and cluster_name arguments.

Within each block, we can also have multiple threads of execution. Programming in Pallas:MGPU is by default done at the warpgroup level (one warpgroup is 4 warps), where one logical Pallas:MGPU thread corresponds to one warpgroup in hardware. This is a useful level to program at because a warpgroup is roughly the number of threads that can run concurrently in an SM since each SM is split into 4 quadrants/subpartitions. On Hopper GPUs, matrix multiplication (wgmma) is performed at the warpgroup level, and on Blackwell GPUs it takes an entire warpgroup to load or store from TMEM. The number of logical threads to launch per block can be specified by the num_threads argument to plgpu.kernel.

The following table maps the thread hierarchy between Pallas:MGPU and CUDA:

------
| Pallas:MGPU | CUDA |
---|---
| n/a | Thread |
| n/a^1 | Warp |
| Thread | Warpgroup |
| Block | Block |
| Cluster | Cluster |
------

1 There is some limited support for warp-level specialization.

### Refs vs Arrays
You’ll notice in the above example that inputs was passed into the kernel as a JAX array, but once inside the kernel it becomes a Ref (input_ref) argument. While arrays in JAX are immutable, Refs (JAX documentation) are a separate construct that allows users to read and write from mutable buffers. You can only directly manipulate Refs in 3 ways: dereferencing the Ref to an array (x = ref[idx]), slicing (ref.at[idx]), or storing a value (ref[idx] = x). Additionally, some operations that require arguments to be in a particular memory space such as SMEM (e.g. wgmma, tcgen05_mma) or GMEM (e.g. async TMA copies) will take in Refs directly as arguments.

In Pallas, Refs are used to represent locations in memory (e.g. GMEM or SMEM), and Arrays represent values held in registers. You can think of a Ref as a pointer with shape information as metadata. De-referencing (note: ref[idx] syntax, not ref.at[idx]) a Ref means to load the value from memory into registers, while storing a value into the Ref will store a value from registers to memory. The nice part of this abstraction is that you no longer need to do pointer arithmetic yourself! Whenever you slice into a Ref, Pallas will automatically compute the corresponding pointer offsets/strides while also handling the physical register layout.

### Platform-specific instructions

Pallas:MGPU contains a collection of platform-specific functions to expose hardware instructions that don’t have a JAX equivalent. These can be accessed by importing the plgpu module (import jax.experimental.pallas.mosaic_gpu as plgpu). Here are some examples:
- TMAs (plgpu.async_copy_smem_to_gmem, plgpu.async_copy_gmem_to_smem)
- Fences (plgpu.commit_smem)
- Mbarriers (plgpu.barrier_arrive, plgpu.barrier_wait)
- Platform-specific matrix multiplication (plgpu.wgmma, plgpu.tcgen05_mma)

## A Basic Matrix Multiplication Kernel
Now that we have a basic understanding of what Pallas:MGPU is, let’s write up the “Hello world” program of machine learning - a matrix multiplication!

Cover up to the matmul example in the mosaic GPU docs.

Single-block.

TODO

Pipelined, multi-block.

TODO

If you’re interested in learning how to optimize this kernel from this point to reach state-of-the-art performance, you can read our in-depth tutorial at: https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html 

## Additional Resources
Please check out these additional resources if you would like to learn more about JAX, GPUs, and Pallas:MGPU:
- Pallas:MGPU Reference Documentation 
- Pallas:MGPU instruction reference
- JAX repository and installation instructions
- A primer on GPUs: https://jax-ml.github.io/scaling-book/gpus 

For examples of existing kernels written in Pallas:MGPU, there are a couple options:
The Tokamax project contains a library of production-ready kernels written in many DSLs (including Pallas:MGPU).
The JAX team also maintains a small collection of performant reference kernels implemented in Pallas:MGPU, which can be found at https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/gpu. 
