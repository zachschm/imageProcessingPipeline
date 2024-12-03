// src/opencl/kernels/sharpening.cl
__kernel void sharpening(__read_only image2d_t inputImage,
                         __write_only image2d_t outputImage,
                         __global const float* kernel, int kernelSize)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int width = get_image_width(inputImage);
    int height = get_image_height(inputImage);

    if (x < width && y < height)
    {
        float sum = 0.0f;
        int halfKernel = kernelSize / 2;

        for (int ky = -halfKernel; ky <= halfKernel; ky++)
        {
            for (int kx = -halfKernel; kx <= halfKernel; kx++)
            {
                int ix = clamp(x + kx, 0, width - 1);
                int iy = clamp(y + ky, 0, height - 1);

                float4 pixel = read_imagef(inputImage,
                                           CLK_NORMALIZED_COORDS_FALSE |
                                               CLK_ADDRESS_CLAMP_TO_EDGE,
                                           (int2)(ix, iy));
                float weight =
                    kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                sum += weight * pixel.x;  // Use the red channel for sharpening
            }
        }

        sum = clamp(sum, 0.0f, 1.0f);
        float4 result = (float4)(sum, sum, sum, 1.0f);
        write_imagef(outputImage, (int2)(x, y), result);
    }
}
