__kernel void resize(__read_only image2d_t srcImage,
                     __write_only image2d_t dstImage, float scaleX,
                     float scaleY)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int dstWidth = get_image_width(dstImage);
    int dstHeight = get_image_height(dstImage);

    if (x < dstWidth && y < dstHeight)
    {
        float srcX = x / scaleX;
        float srcY = y / scaleY;

        // Bilinear interpolation
        float2 coord = (float2)(srcX, srcY);
        float4 pixel = read_imagef(srcImage,
                                   CLK_NORMALIZED_COORDS_FALSE |
                                       CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR,
                                   coord);

        // Write the interpolated pixel to the destination image
        write_imagef(dstImage, (int2)(x, y), pixel);
    }
}
