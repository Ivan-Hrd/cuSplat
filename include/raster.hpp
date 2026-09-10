#include "tiling.hpp"
#include "raft/core/device_span.hpp"
#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_uvector.hpp"

class Raster {
private:
    rmm::device_uvector<float> d_imageMemory;
    rmm::device_uvector<TileRange> tileRangesMemory;
public:
    raft::device_span<Gaussian> gaussians;
    raft::device_span<float> d_image;
    Camera camera;
    int nTilesX;
    int nTilesY;
    raft::device_span<TileRange> tileRanges;
    Raster(raft::device_span<Gaussian> gaussians, Camera &camera) : camera(camera), gaussians(gaussians),
                                                                    d_imageMemory(
                                                                        camera.width * camera.height * 3,
                                                                        rmm::cuda_stream_default),
                                                                    tileRangesMemory(
                                                                        ((camera.width + TILE_SIZE - 1) / TILE_SIZE)*((camera.height + TILE_SIZE - 1) / TILE_SIZE),
                                                                        rmm::cuda_stream_default) {
        nTilesX = (camera.width + TILE_SIZE - 1) / TILE_SIZE;
        nTilesY = (camera.height + TILE_SIZE - 1) / TILE_SIZE;
        tileRanges = raft::device_span<TileRange>(tileRangesMemory.data(), nTilesX * nTilesY);
        d_image = raft::device_span<float>(d_imageMemory.data(), d_imageMemory.size());
    }

    void rasterize(float *image);

};
