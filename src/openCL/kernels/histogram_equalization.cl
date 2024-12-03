__kernel void
histogram_equalization(__global const uchar* input, __global uchar* output,
                       __global int* localHist, __global int* localCdf,
                       int width, int height, int row_offset, int totalPixels)
{
    int x = get_global_id(0);
    int y = get_global_id(1) + row_offset;

    // Ensure work is within bounds
    if (x < width && y < height)
    {
        int idx = y * width + x;

        // Update histogram
        uchar intensity = input[idx];
        atomic_inc(&localHist[intensity]);

        // Barrier to ensure all threads complete histogram calculation
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Compute the CDF once per workgroup
        if (get_global_id(0) == 0 && get_global_id(1) == 0)
        {
            localCdf[0] = localHist[0];
            for (int i = 1; i < 256; ++i)
            {
                localCdf[i] = localCdf[i - 1] + localHist[i];
            }
        }

        // Barrier to ensure the CDF is ready
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Map input to output
        output[idx - (row_offset * width)] =
            (uchar)(((float)localCdf[intensity] - localCdf[0]) /
                    (totalPixels - localCdf[0]) * 255);
    }
}
