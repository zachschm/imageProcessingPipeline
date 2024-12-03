// src/opencl/kernels/sobel_edge.cl
__kernel void sobelEdge(__read_only image2d_t inputImage,
                        __write_only image2d_t outputImage)
{
    const sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    int x = get_global_id(0);
    int y = get_global_id(1);

    int2 pos = (int2)(x, y);

    // Sobel kernels
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    float gradientX = 0.0f;
    float gradientY = 0.0f;

    // Apply Sobel operator
    for (int i = -1; i <= 1; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            int2 neighborPos = (int2)(x + i, y + j);
            float pixel = read_imagef(inputImage, sampler, neighborPos)
                              .x;  // Read grayscale value

            gradientX += Gx[i + 1][j + 1] * pixel;
            gradientY += Gy[i + 1][j + 1] * pixel;
        }
    }

    // Compute edge magnitude
    float magnitude = sqrt(gradientX * gradientX + gradientY * gradientY);
    magnitude = fmin(fmax(magnitude, 0.0f), 1.0f);  // Clamp to [0, 1]

    write_imagef(outputImage, pos,
                 (float4)(magnitude, magnitude, magnitude, 1.0f));
}
