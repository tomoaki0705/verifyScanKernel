#include <iostream>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstring>
#include <stdint.h>
#include <cmath>

#define CUDA_SAFE_CALL(func) \
do { \
        cudaError_t err = (func); \
        if (err != cudaSuccess) { \
                fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
                exit(err); \
        } \
} while (0)

float inputSrc[] = {
-0.352116,  0.52175,    -0.398729,  0.78888,    -0.254948,  0.663178,   0.249477,   -0.0624508, 0.0330817,  0.326156,   0.995528,   0.797423,   -0.782847,  -0.924304,  0.236109,   -0.295402,  -0.958085,  -0.647206,  0.890096,   0.645139,   -0.0992178, -0.678596,  0.440066,0.916935,   0.440062,   -0.980969,  0.267242,   -0.50554,   -0.760408,  -0.376026,      -0.923987,      -0.112524,      -0.854276,      -0.322716,      -0.323644,      -0.109225,      -0.659538,      0.925833,   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
0.120409,   -0.71457,   -0.63629,   -0.754743,  0.651874,   0.960259,   -0.272762,  -0.278134,  0.924093,   -0.708786,  0.409392,   -0.955534,  -0.119105,  0.166885,   -0.281341,  -0.788876,  -0.761118,  0.207553,   0.832673,   0.851211,   -0.398824,  -0.785917,  -0.76204,-0.666359,  -0.202591,  -0.563694,  -0.444551,  -0.128299,  -0.15822,   0.862886,       -0.925886,      0.962189,       -0.851684,      -0.562176,      -0.792554,      0.800189,       -0.601918,      -0.0653162, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,};
uint32_t srcStride = 128;
uint32_t srcWidth  = 38;
uint32_t srcHeight = 2;
uint32_t devStride = 128;
uint32_t devHeight = 128;


struct _mySize
{
    uint32_t width;
    uint32_t height;
};

typedef struct _mySize mySize;

uint32_t scanRowsWrapperDevice(float *d_src, uint32_t srcStride, float *d_dst, uint32_t dstStride, mySize roi);

TEST(testAccuracy, Integral) {
    // allocate memory (GPU)
    float *d_src, *d_dst;
    const uint32_t size = sizeof(float)*devStride*devHeight;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_src,  size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_dst,  size));
    CUDA_SAFE_CALL(cudaMemset((void*)d_src, 0, size));
    CUDA_SAFE_CALL(cudaMemset((void*)d_dst, 0, size));
    mySize imageSize;
    imageSize.width  = srcWidth;
    imageSize.height = srcHeight;

    // allocate memory (CPU)
    float *dstCPU = new float[(srcWidth+1) * srcHeight];
    float *host_dst = new float[devStride * devHeight];
    std::memset(dstCPU, 0, sizeof(float)*(srcWidth+1)*srcHeight);

    // transfer to gpu
    const uint32_t srcSize = sizeof(float) * devStride * srcHeight;
    CUDA_SAFE_CALL(cudaMemcpy(d_src, inputSrc, srcSize, cudaMemcpyHostToDevice));

    // scan row kernel
    uint32_t status = scanRowsWrapperDevice(d_src, srcStride, d_dst, devStride, imageSize);       

    // transfer back to cpu
    CUDA_SAFE_CALL(cudaMemcpy(d_dst, host_dst, srcSize, cudaMemcpyDeviceToHost));

    // create reference on CPU
    for(uint32_t y = 0;y < srcHeight;y++)
    {
        float sum = 0.f;
        for(uint32_t x = 0;x < srcWidth ;x++)
        {
            float v = inputSrc[srcStride * y + x];
            sum += v;
            dstCPU[srcStride * y + x + 1] = sum;
        }
    }

    // verify the results
    bool pass = true;
    for(uint32_t y = 0;y < srcHeight;y++)
    {
        for(uint32_t x = 0;x <= srcWidth;x++)
        {
            if(std::abs(dstCPU[(srcWidth+1) * y + x] - host_dst[srcStride * y + x]) > 0.01f)
            {
                pass = false;
            }
        }
    }

    // release the array
    delete [] dstCPU;
    delete [] host_dst;
    CUDA_SAFE_CALL(cudaFree((void*)d_src));
    CUDA_SAFE_CALL(cudaFree((void*)d_dst));

    EXPECT_TRUE(pass);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
