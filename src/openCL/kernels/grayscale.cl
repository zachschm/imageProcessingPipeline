// src/opencl/kernels/grayscale.cl
__kernel void grayscale(__global const uchar* inputImage,
                        __global uchar* outputImage, int width, int height,
                        int row_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1) + row_offset;  // Adjust for GPU chunk offset

    if (x < width && y < height)
    {
        int idx = (y * width + x) * 3;  // RGB input
        uchar r = inputImage[idx];
        uchar g = inputImage[idx + 1];
        uchar b = inputImage[idx + 2];
        uchar gray = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);
        outputImage[(y - row_offset) * width + x] = gray;
    }
}
