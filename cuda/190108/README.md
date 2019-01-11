## CUDA Implicit Warp Level Synchronization

https://stackoverflow.com/a/54139515/7724939

Compile with `$ make` and run tests with `$ bash run.sh`.

### Test results

On CUDA `10.0`, `9.2`, `9.1`, `9.0`:

```
test_host host    :    0    1    3    6   10   15   21   28   36   45   55   66   78   91  105  120  136  153  171  190  210  231  253  276  300  325  351  378  406  435  465  496 (host)
test_dev0 kernel0 :    0    1    3    6   10   15   21   28   36   45   55   66   78   91  105  120  136  153  171  190  210  231  253  276  300  325  351  378  406  435  465  496 (device, global memory)
test_dev1 kernel1 :    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31 (device, shared memory)
test_arch kernel1 :    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31 (device, shared memory)
test_dev2 kernel2 :    0    1    3    6   10   15   21   28   36   45   55   66   78   91  105  120  136  153  171  190  210  231  253  276  300  325  351  378  406  435  465  496 (device, shared memory with volatile qualifier)
```

On CUDA `8.0`:

```
test_host host    :    0    1    3    6   10   15   21   28   36   45   55   66   78   91  105  120  136  153  171  190  210  231  253  276  300  325  351  378  406  435  465  496 (host)
test_dev0 kernel0 :    0    1    3    6   10   15   21   28   36   45   55   66   78   91  105  120  136  153  171  190  210  231  253  276  300  325  351  378  406  435  465  496 (device, global memory)
test_dev1 kernel1 :    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31 (device, shared memory)
test_arch kernel1 :    0    1    3    6   10   15   21   28   36   45   55   66   78   91  105  120  136  153  171  190  210  231  253  276  300  325  351  378  406  435  465  496 (device, shared memory)
test_dev2 kernel2 :    0    1    3    6   10   15   21   28   36   45   55   66   78   91  105  120  136  153  171  190  210  231  253  276  300  325  351  378  406  435  465  496 (device, shared memory with volatile qualifier)
```

### Disscusion

In this case, implicit warp level synchronization still works with `volatile` keyword. However, these kind of implicit warp-synchronous programming should be avoided generally.

NVIDIA stated in the [**Kepler Tuning Guide**](https://docs.nvidia.com/cuda/kepler-tuning-guide/index.html#warp-synchronous) that:

> 1.4.9. Warp-synchronous Programming
> As a means of mitigating the cost of repeated block-level synchronizations, particularly in parallel primitives such as reduction and prefix sum, some programmers exploit the knowledge that threads in a warp execute in lock-step with each other to omit `__syncthreads()` in some places where it is semantically necessary for correctness in the CUDA programming model.

> The absence of an explicit synchronization in a program where different threads communicate via memory constitutes a data race condition or synchronization error. **Warp-synchronous programs are unsafe and easily broken** by evolutionary improvements to the optimization strategies used by the CUDA compiler toolchain, which generally has no visibility into cross-thread interactions of this variety in the absence of barriers, or by changes to the hardware memory subsystem's behavior. Such programs also tend to assume that the warp size is 32 threads, which may not necessarily be the case for all future CUDA-capable architectures.

> Therefore, **programmers should avoid warp-synchronous programming to ensure future-proof correctness** in CUDA applications. When threads in a block must communicate or synchronize with each other, regardless of whether those threads are expected to be in the same warp or not, the appropriate barrier primitives should be used. Note that the Warp Shuffle primitive presents a future-proof, supported mechanism for intra-warp communication that can safely be used as an alternative in many cases.

It is also stated by NVIDIA in the [**Volta Tuning Guide**](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html#sm-independent-thread-scheduling) that:

> 1.4.1.2. Independent Thread Scheduling
> The Volta architecture introduces Independent Thread Scheduling among threads in a warp. This feature enables intra-warp synchronization patterns previously unavailable and simplifies code changes when porting CPU code. However, Independent Thread Scheduling can also lead to a rather different set of threads participating in the executed code than intended **if the developer made assumptions about warp-synchronicity of previous hardware architectures.** (The term warp-synchronous refers to code that implicitly assumes that threads in the same warp are synchronized at every instruction.)

> When porting existing codes to Volta, the following three code patterns need careful attention. For more details see the CUDA C Programming Guide.

> To avoid data corruption, applications using warp intrinsics (`__shfl*`, `__any`, `__all`, and `__ballot`) should transition to the new, safe, synchronizing counterparts, with the `*_sync` suffix. The new warp intrinsics take in a mask of threads that explicitly define which lanes (threads of a warp) must participate in the warp intrinsic.

> **Applications that assume reads and writes are implicitly visible to other threads in the same warp need to insert the new `__syncwarp()` warp-wide barrier synchronization instruction between steps where data is exchanged between threads via global or shared memory. Assumptions that code is executed in lockstep or that reads/writes from separate threads are visible across a warp without synchronization are invalid.**

> Applications using `__syncthreads()` or the PTX `bar.sync` (and their derivatives) in such a way that a barrier will not be reached by some non-exited thread in the thread block must be modified to ensure that all non-exited threads reach the barrier.

> The `racecheck` and `synccheck` tools provided by `cuda-memcheck` can aid in locating violations of points 2 and 3.
