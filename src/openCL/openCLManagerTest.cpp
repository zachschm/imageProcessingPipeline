#include "OpenCLManager.h"
#include <iostream>

int main()
{
    try
    {
        OpenCLManager manager;

        // Test context, device, and queue
        cl::Context context = manager.getContext();
        cl::Device device = manager.getDevice();
        cl::CommandQueue queue = manager.getQueue();

        std::cout
            << "OpenCL context, device, and queue initialized successfully."
            << std::endl;

        // Test loading a kernel
        cl::Program program =
            manager.loadProgram("src/opencl/kernels/grayscale.cl");
        std::cout << "Kernel program loaded successfully." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
