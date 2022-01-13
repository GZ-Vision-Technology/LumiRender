

__device__ inline float catmull_rom(
        float p[4],
        float t) {
    return p[1] + 0.5f * t * (p[2] - p[0] + t * (2.f * p[0] - 5.f * p[1] + 4.f * p[2] - p[3] + t * (3.f * (p[1] - p[2]) + p[3] - p[0])));
}

// apply flow to image at given pixel position (using bilinear interpolation), write back RGB result.
__device__ void denoise_add_temporal_flow(
        float4 *result,
        const float4 *image,
        const float4 *flow,
        unsigned int width,
        unsigned int height,
        unsigned int x,
        unsigned int y) {

    float dst_x = float(x) - flow[x + y * width].x;
    float dst_y = float(y) - flow[x + y * width].y;

    float x0 = dst_x - 1.f;
    float y0 = dst_y - 1.f;

    float r[4][4], g[4][4], b[4][4];
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
            int tx = static_cast<int>(x0) + k;
            if (tx < 0)
                tx = 0;
            else if (tx >= (int) width)
                tx = width - 1;

            int ty = static_cast<int>(y0) + j;
            if (ty < 0)
                ty = 0;
            else if (ty >= (int) height)
                ty = height - 1;

            r[j][k] = image[tx + ty * width].x;
            g[j][k] = image[tx + ty * width].y;
            b[j][k] = image[tx + ty * width].z;
        }
    }
    float tx = dst_x <= 0.f ? 0.f : dst_x - floorf(dst_x);

    r[0][0] = catmull_rom(r[0], tx);
    r[0][1] = catmull_rom(r[1], tx);
    r[0][2] = catmull_rom(r[2], tx);
    r[0][3] = catmull_rom(r[3], tx);

    g[0][0] = catmull_rom(g[0], tx);
    g[0][1] = catmull_rom(g[1], tx);
    g[0][2] = catmull_rom(g[2], tx);
    g[0][3] = catmull_rom(g[3], tx);

    b[0][0] = catmull_rom(b[0], tx);
    b[0][1] = catmull_rom(b[1], tx);
    b[0][2] = catmull_rom(b[2], tx);
    b[0][3] = catmull_rom(b[3], tx);

    float ty = dst_y <= 0.f ? 0.f : dst_y - floorf(dst_y);

    result[y * width + x].x = catmull_rom(r[0], ty);
    result[y * width + x].y = catmull_rom(g[0], ty);
    result[y * width + x].z = catmull_rom(b[0], ty);
}

extern "C" __global__ void denoise_add_temporal_flow_kernel(
        float4 *result,
        const float4 *image,
        const float4 *flow,
        unsigned int width,
        unsigned int height) {
    uint2 gid = threadIdx + blockDim * blockIdx;

    denoise_add_temporal_flow(
        result,
        image,
        flow,
        width,
        height,
        gid.x,
        gid.y
    );
}