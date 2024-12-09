#include "BatchManager.h"
#include "OpenCLManager.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_directory>" << std::endl;
        return 1;
    }

    std::string imageDirectory = argv[1];
    const char* mode = std::getenv("PROCESSING_MODE");
    std::string processingMode = mode ? std::string(mode) : "gpu";

    try
    {
        // Validate processing mode
        if (processingMode != "cpu" && processingMode != "gpu")
        {
            throw std::invalid_argument(
                "Invalid PROCESSING_MODE. Use 'cpu' or 'gpu'.");
        }

        // Declare OpenCLManager pointer
        OpenCLManager* openclManager = nullptr;

        if (processingMode == "gpu")
        {
            std::cout << "Initializing OpenCLManager for GPU processing..."
                      << std::endl;
            openclManager = new OpenCLManager();
            if (openclManager->getDeviceCount() < 4)
            {
                throw std::runtime_error(
                    "Insufficient GPUs available for processing. Ensure at "
                    "least 4 GPUs are accessible.");
            }
        }

        // Initialize the batch manager
        OpenCLManager dummyManager;  // Placeholder for CPU mode
        BatchManager batchManager(imageDirectory, (processingMode == "gpu")
                                                      ? *openclManager
                                                      : dummyManager);

        // Process the batch
        batchManager.processBatch();

        // Clean up OpenCLManager if it was initialized
        if (openclManager)
        {
            delete openclManager;
        }

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
