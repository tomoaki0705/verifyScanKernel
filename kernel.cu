#include <vector>
#include <cuda_runtime.h>
#include "types.h"

#define WORKAROUND 1
#define DUMP_RESULTS 0
#if DUMP_RESULTS
#define DUMP(a,b,c,d,e) dump(a,b,c,d,e)
#else
#define DUMP(a,b,c,d,e)
#endif


//==============================================================================
//
// IntegralImage.cu
//
//==============================================================================


const uint32_t NUM_SCAN_THREADS = 256;
const uint32_t LOG2_NUM_SCAN_THREADS = 8;


template<class T_in, class T_out>
struct _scanElemOp
{
    template<bool tbDoSqr>
    static inline __host__ __device__ T_out scanElemOp(T_in elem)
    {
        return scanElemOp( elem, Int2Type<(int)tbDoSqr>() );
    }

private:

    template <int v> struct Int2Type { enum { value = v }; };

    static inline __host__ __device__ T_out scanElemOp(T_in elem, Int2Type<0>)
    {
        return (T_out)elem;
    }

    static inline __host__ __device__ T_out scanElemOp(T_in elem, Int2Type<1>)
    {
        return (T_out)(elem*elem);
    }
};


template<class T>
inline __device__ T readElem(T *d_src, uint32_t texOffs, uint32_t srcStride, uint32_t curElemOffs);



template<>
inline __device__ uint32_t readElem<uint32_t>(uint32_t *d_src, uint32_t texOffs, uint32_t srcStride, uint32_t curElemOffs)
{
    return d_src[curElemOffs];
}


template<>
inline __device__ float readElem<float>(float *d_src, uint32_t texOffs, uint32_t srcStride, uint32_t curElemOffs)
{
    return d_src[curElemOffs];
}


/**
* \brief Segmented scan kernel
*
* Calculates per-row prefix scans of the input image.
* Out-of-bounds safe: reads 'size' elements, writes 'size+1' elements
*
* \tparam T_in      Type of input image elements
* \tparam T_out     Type of output image elements
* \tparam T_op      Defines an operation to be performed on the input image pixels
*
* \param d_src      [IN] Source image pointer
* \param srcWidth   [IN] Source image width
* \param srcStride  [IN] Source image stride
* \param d_II       [OUT] Output image pointer
* \param IIstride   [IN] Output image stride
*
* \return None
*/
template <class T_in, class T_out, bool tbDoSqr>
__global__ void scanRows(T_in *d_src, uint32_t texOffs, uint32_t srcWidth, uint32_t srcStride,
                         T_out *d_II, uint32_t IIstride)
{
    //advance pointers to the current line
    if (sizeof(T_in) != 1)
    {
        d_src += srcStride * blockIdx.x;
    }
    //for initial image 8bit source we use texref tex8u
    d_II += IIstride * blockIdx.x;
    d_II[0] = 0;

    uint32_t numBuckets = (srcWidth + NUM_SCAN_THREADS - 1) >> LOG2_NUM_SCAN_THREADS;
    uint32_t offsetX = 0;

    __shared__ T_out shmem[NUM_SCAN_THREADS];
    __shared__ T_out carryElem;
    carryElem = 0;
    __syncthreads();

#if (WORKAROUND == 1)
    T_out sum = 0;
    for(int x = 0;x < srcWidth;x++)
    {
        T_out v = d_src[x];
        sum += tbDoSqr ? v * v : v;
        d_II[x+1] = sum;
    }
#else
    while (numBuckets--)
    {
        uint32_t curElemOffs = offsetX + threadIdx.x;
        T_out curScanElem;

        T_in curElem;
        T_out curElemMod;

        if (curElemOffs < srcWidth)
        {
            //load elements
            curElem = readElem<T_in>(d_src, texOffs, srcStride, curElemOffs);
        }
        curElemMod = _scanElemOp<T_in, T_out>::scanElemOp<tbDoSqr>(curElem);

        //inclusive scan
        curScanElem = cv::cudev::blockScanInclusive<NUM_SCAN_THREADS>(curElemMod, shmem, threadIdx.x);

        if (curElemOffs <= srcWidth)
        {
            //make scan exclusive and write the bucket to the output buffer
            d_II[curElemOffs] = carryElem + curScanElem - curElemMod;
            offsetX += NUM_SCAN_THREADS;
        }

        //remember last element for subsequent buckets adjustment
        __syncthreads();
        if (threadIdx.x == NUM_SCAN_THREADS-1)
        {
            carryElem += curScanElem;
        }
        __syncthreads();
    }

    if (offsetX == srcWidth && !threadIdx.x)
    {
        d_II[offsetX] = carryElem;
    }
#endif
}

uint32_t scanRowsWrapperDevice(float *d_src, uint32_t srcStride, float *d_dst, uint32_t dstStride, mySize roi)
{
    cudaChannelFormatDesc cfdTex;
    size_t alignmentOffset = 0;
    scanRows
        <float, float, false>
        <<<roi.height, NUM_SCAN_THREADS, 0, 0>>>
        (d_src, (uint32_t)alignmentOffset, roi.width, srcStride, d_dst, dstStride);

    return 0;
}

