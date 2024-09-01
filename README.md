# gpu-burn
Multi-GPU stress test
http://wili.cc/blog/gpu-burn.html

# Building
To build GPU Burn:

`make`

To remove artifacts built by GPU Burn:

`make clean`

# Usage
```bash
GPU Burn
Usage: gpu_burn [OPTIONS] [TIME]

-m X   Use X MB of memory
-m N%  Use N% of the available GPU memory
-d     Use doubles
-tc    Try to use Tensor cores (if available)
-l     List all GPUs in the system
-i N   Execute only on GPU N
-h     Show this help message

Example:
gpu_burn -d 3600
```

Example output
```bash
$ ./gpu_burn 
Run length not specified in the command line. Using compare file: compare.hsaco
Burning for 10 seconds.


========================================= ROCm System Management Interface =========================================
=================================================== Concise Info ===================================================
Device  Node  IDs              Temp    Power  Partitions          SCLK    MCLK     Fan  Perf  PwrCap  VRAM%  GPU%  
              (DID,     GUID)  (Edge)  (Avg)  (Mem, Compute, ID)                                                   
====================================================================================================================
0       1     0x740f,   7916   30.0°C  41.0W  N/A, N/A, 0         800Mhz  1600Mhz  0%   auto  300.0W  0%     0%    
1       2     0x740f,   28773  34.0°C  41.0W  N/A, N/A, 0         800Mhz  1600Mhz  0%   auto  300.0W  0%     0%    
====================================================================================================================
=============================================== End of ROCm SMI Log ================================================
Initialized device 0 with 65520 MB of memory (65452 MB available, using 58906 MB of it), using FLOATS
Results are 268435456 bytes each, thus performing 228 iterations
Initialized device 1 with 65520 MB of memory (65452 MB available, using 58906 MB of it), using FLOATS
Results are 268435456 bytes each, thus performing 228 iterations

Killing processes with SIGTERM (soft kill)
Freed memory for dev 0
Uninitted cublas
Freed memory for dev 1
Uninitted cublas
done

Tested 2 GPUs:
        GPU 0: OK
        GPU 1: OK
```