# cuSplat

### Hardware-accelerated 3D Gaussian Splatting rendering pipeline using CUDA.
 
Optimized on a 3050Ti laptop.

## 3DGS Architecture

### From `.ply` to `Rasterize` input

This project covers the rendering (or forward) phase of gaussian splatting.
That's why we do not generate our own `.ply` input file, but the pipeline most likely reads it and transfer the gaussian vector
to the GPU.

### `Rasterize()` structure

Its architecture has been made with respect to the [original 3DGS paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) 
which describes the high-level design within the `rasterize()` algorithm as : \
![img.png](img.png) \
Each step depends on the previous one, that's why we can't run two kernels at the same time. The input of one function is 
the output of the previous one, that's why everything has been ported to GPU to keep every data directly into GPU and avoid
as much as possible DtoH and HtoD transfer. \
The GPU version of `Rasterize` do not needs `for` loop and is structured as is : 

- Frustum culling : `thrust::copy_if` with a device lambda calling `worldToCamera` + `isVisible`
- Screen-space projection : `screenspaceGaussians<<<GS,BS>>>` kernel
- Key count : `GetSize<<<GS,BS>>>` counts tiles touched by each splat
- Key generation : `FillKeys<<<GS,BS>>>` emits one `uint64_t` key = (tile_id ‖ depth) per touch, plus an index into the `GaussianSplated` array
- Sort : `thrust::sort_by_key` on the 64-bit keys    
- Tile ranges : `IdentifyTileRanges<<<GS,BS>>>` finds `[start,end)` per tile in the sorted key list          
- Alpha blend : `AlphaBlend<<<GSize,BSize>>>` 1 block = 1 tile, 1 thread = 1 pixel, splats cooperatively staged in shared memory (double-buffered)
- Async DtoH copy : `cudaMemcpyAsync(image, d_image, ..., DeviceToHost)`     

### From `Rasterize` output to `.ppm`

After `rasterize()` returns, the host buffer `image` holds `width * height * 3` floats in `[0, 1]`. `savePPM()` in `src/snapshot.cu` writes an ASCII PPM (`P3`)

## Branches

- `main` : The primary branch containing the latest stable GPU code, incorporating all optimizations documented in the project reports.
- `cpu-baseline` : First implementation of 3DGS in pure C++ (CPU) following the original Inria paper architecture.
- `gpu-baseline` : GPU implementation of 3DGS, porting the CPU pipeline in CUDA with no optimization yet.
- `gpu-opti-1` : Optimization of the gpu-baseline, include every optimization found in the report of its branch (including memory
transfer optimizations, profiling & optimization of the alphaBlend kernel)
- `gpu-opti-x` : Incremental optimization branches (where x is a number). Higher numbers include all improvements from 
previous branches plus new optimizations detailed in that branch's report (e.g., gpu-opti-2 builds directly on top of gpu-opti-1).

## Usage

To compile the project you should at least have installed cmake and nvidia developer toolkit.

You should compile at the root of the repo : \
Build the project \
```$> cmake -B build``` \
Compile it \
```$> cmake --build build``` \
\
In the build file you have 2 binary you can execute :
- `snapshot` : execute the pipeline on a single frame. Used to have a quick preview of the output.
- `bench` : Benchmarks the entire pipeline using CUDA events to collect accurate GPU timing metrics. The test runs first on a single frame, then across 200 consecutive frames from the same camera perspective.
