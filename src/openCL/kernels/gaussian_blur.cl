// src/opencl/kernels/gaussian_blur.cl
__kernel void gaussian_blur(__read_only image2d_t inputImage,
                            __write_only image2d_t outputImage,
                            __global const float* kernel, int kernelSize)
{
    const sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    int x = get_global_id(0);
    int y = get_global_id(1);

    int halfKernel = kernelSize / 2;
    float sum = 0.0f;

    for (int ky = -halfKernel; ky <= halfKernel; ++ky)
    {
        for (int kx = -halfKernel; kx <= halfKernel; ++kx)
        {
            int2 coord = (int2)(x + kx, y + ky);
            float pixel = read_imagef(inputImage, sampler, coord).x;
            float weight =
                kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
            sum += pixel * weight;
        }
    }

    write_imagef(outputImage, (int2)(x, y), (float4)(sum, sum, sum, 1.0f));
}
