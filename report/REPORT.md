# Optimization report

Here is a clear view of the performance of cuSplat for the gpu baseline branch, before any optimization.
![img_2.png](img_2.png)
*NSight System screenshot for the GPU baseline with no optimization - 1 frame (`nsys_report.nsys-rep`)*

At first glance, there are two main concerning parts. First the memory transfert from host to device that comes
from `thrust::copy` so it relate to the first transfert of every gaussian to the GPU. The second part, and the one taking
most the time, is the alpha blending kernel.

`1 Frame : 0.137 second` \
`200 Frames : 0.135 second`

## Memory transfer

### Reducing memory transfer
By generating only one frame, we miss a lot of the memory transfert work that would be done in a realistic use of a 3DGS renderer.
Indeed when we loop on the rasterization process (as it would be done in a real-time renderer), we are going from a workload of 66%
for kernels and 33% for memory to 42% kernel and 58% memory.

![img_3.png](img_3.png)
*NSight System screenshot for GPU baseline using pinned memory - 200 frames (`nsys_report_multiFrame.nsys-rep`)*

It is explained by the transfers of the raw gaussians that are done at the rasterization step, thus once per frame (even though each frame use the exact same gaussians, as we handle the same scene).
Doing transfer in rasterization did not matter when we were doing it for only one frame, but it became relevant when we handled a lot of them.
That's why, relocating gaussians to the device once and for all before any rasterization will avoid a lot of useless memory workload.

![img_4.png](img_4.png)
*NSight System screenshot for GPU baseline using pinned memory and loading the gaussians only once - 200 frames (`nsys_report_multiFrame_opti.nsys-rep`)*

It can now be clearly seen that, with a big enough number of frame, most of the remaining optimization is at kernel level with more than 85% of the workload being at kernel execution time.

`1 Frame : 0.0923 second` x1.48 SpeedUp \
`200 Frames : 0.086 second` x1.57 SpeedUp

> Note : Host memory allocation for `gaussians` and `image` has also changed to `cudaHostAlloc()` to use pinned memory,
> it increased memory throughput (from 4.9GiB/s to 6.3GiB/s) but no significant overall speedup was made (as `gaussians` loading is done once for every frame) so no section has been written about it

