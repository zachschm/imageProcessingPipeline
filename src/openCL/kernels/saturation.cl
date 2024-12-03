// src/opencl/kernels/saturation.cl
__kernel void adjustSaturation(__read_only image2d_t inputImage,
                               __write_only image2d_t outputImage, float scale)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Read pixel
    uint4 pixel = read_imageui(
        inputImage, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE,
        (int2)(x, y));
    float h = pixel.x / 255.0f;  // Hue
    float s = pixel.y / 255.0f;  // Saturation
    float v = pixel.z / 255.0f;  // Value

    // Adjust saturation
    s *= scale;
    s = clamp(s, 0.0f, 1.0f);

    // Write back to the output image
    write_imageui(outputImage, (int2)(x, y),
                  (uint4)(h * 255, s * 255, v * 255, pixel.w));
}
