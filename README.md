# stream-dsa
## Prerequisites: Intel DML library 

This repository shows techniques that complement the strengths of Intel DSA in combination with CPUs to accelerate memory-bound/STREAM kernels (i.e., operations where performance is primarily determined by how fast the memory subsystem can deliver the data operands for computations to the CPU cores).

Intel® Data Streaming Accelerator (Intel® DSA) is a high-performance data copy and transformation accelerator, available starting from 4th Gen Intel® Xeon® Scalable processors. It supports various data operations such as memory copy, compare, fill, compare, etc. Intel DSA employs work queues and descriptors to interact with the CPU software. The descriptor contains information about the desired operation such as operation type, source and destination address, data length, etc. 

Many scientific and commercial applications require efficient CPU’s and high memory bandwidth to achieve optimal performance. While CPU’s receive significant performance enhancements with each generation through additional cores, wider SIMD units, and new ISA extensions among other microarchitectural features, the gains in DRAM bandwidth have historically lagged the CPU improvements. This leads to a scenario where real-world applications do not observe expected performance gains, as the speed at which data is fetched from DRAM is not fast enough to prevent stalls of CPU execution units. This has been a well-known observation for many years and is popularly known as “memory wall.”

CPUs have architectural buffers in the form of load/store queues and super queues and hardware prefetchers that mitigate some of the memory wall effects. However, their impact on memory bandwidth is limited at low core counts because each CPU core can only have a limited number of entries in hardware queues. When the data needs to be fetched from DRAM, the queues become the bottleneck as each memory request occupies the buffer slots for a longer duration to cover the DRAM latency. Hence, it is often common to use more CPU cores to generate memory access requests simultaneously to fully saturate available DRAM bandwidth. However, it’s still a challenge to achieve higher memory bandwidth at lower core counts.

While Intel DSA acts as a powerful data copy/transformation engine, it does not support execution of arithmetic operations like multiplication, addition, fused multiply-add, etc. on data operands. So on its own, it cannot be used in execution of application kernels that contain a mix of memory and compute operations. However, Intel DSA provides the unique capability to control the location of destination data produced from its operations: DRAM or last-level-cache (LLC) on CPUs. For example, DSA can copy data from a source buffer residing in DRAM to a destination located either in DRAM (bypassing the CPU caches) or LLC.

## Hybrid CPU + DSA Software Pipelinig
We can exploit the cache writing capability of Intel DSA and use it as a proxy hardware prefetch engine from DRAM to LLC and rely on the CPU cores for computation operations. The core of the solution works by efficiently overlapping Intel DSA data transfers (from DRAM to LLC) with CPU computations happening from LLC to CPU registers through asynchronous copies. By default, Intel DSA can write to an approximately 14 MB portion of LLC (two out of 15-way LLC, effective size = 2/15 * size of LLC = 14 MB on Intel 4th Gen Xeon Scalable Processors). The figure below shows the high-level differences between the CPU-only approach compared to the hybrid CPU + Intel DSA workflow.

![f1-high-level-differences-cpu-only](https://github.com/user-attachments/assets/17345189-5840-43c4-b553-12ebae1b1c02)

In the CPU-only approach, steps 1 and 2 refer to the load/store requests originating from CPU cores for data residing in DRAM and going to the memory controllers. In step 1 of the CPU + Intel DSA approach, the CPU loads data from LLC instead of DRAM. In step 2, Intel DSA initiates the data transfer from DRAM (through the memory controller) to LLC. Steps 1 and 2 happen in a pipelined and asynchronous manner such that the data the CPU needs to read at time Tx is already copied to LLC by DSA in T(x-1). In other words, while Intel DSA will concurrently fetch the next iteration’s data from DRAM and write them to LLC, the CPUs will read the current iteration’s data from LLC to perform compute operations. Software pipelining will ensure that all DSA engines are efficiently used.

Below figure shows the workflow of the hybrid CPU + Intel DSA pipelining approach for an elementary CPU operation like reading a data array from DRAM. Here, for each CPU thread, we use a queue depth of four with each entry holding 1 MB of data of input buffers. So, basically, while the CPU is reading 4 MB from LLC, DSA will fetch the next chunk of 4MB from DRAM.

![f2-sample-pipeline-data-operations-hybrid-cpu](https://github.com/user-attachments/assets/07a6e6c7-c3d4-40ad-9f3f-5a7f906a9d42)

## Performance
Performance at various core counts on a 4th Gen Intel Xeon Scalable Processor (56-cores, 8-channel DDR5@4800MT/s, theoretical peak bandwidth is 8 channels x 8 bytes x 4.8 GT/s = 307 GB/s) is shown below. Intel DSA+CPU speedup over the CPU-only implementation is color-coded as a heatmap.

![f5-performance-comparison-cpu-hybrid](https://github.com/user-attachments/assets/c898cc3d-7845-4dae-b772-70b6dd496175)

For more details, refer to the technical article at https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-memory-bandwidth-bound-kernels-with-dsa.html
