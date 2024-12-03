#include "Image.h"
#include "OpenCLManager.h"
#include "OpenCLResizeStep.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

TEST(OpenCLResizeStepTest, ResizeOperation)
{
    OpenCLManager oclManager;
    int gpuIndex = 0;
    int newWidth = 256;
    int newHeight = 256;

    OpenCLResizeStep resizeStep(oclManager, gpuIndex, newWidth, newHeight);

    // Create a synthetic RGB image (512x512)
    cv::Mat inputImage(512, 512, CV_8UC3, cv::Scalar(100, 150, 200));
    Image img(inputImage);

    resizeStep.process(img);

    cv::Mat resizedImage = img.getImage();

    // Check that the dimensions are correct
    ASSERT_EQ(resizedImage.cols, newWidth);
    ASSERT_EQ(resizedImage.rows, newHeight);

    // Validate pixel data (rough check for consistency)
    cv::Scalar meanOriginal = cv::mean(inputImage);
    cv::Scalar meanResized = cv::mean(resizedImage);

    ASSERT_NEAR(meanOriginal[0], meanResized[0], 1.0);
    ASSERT_NEAR(meanOriginal[1], meanResized[1], 1.0);
    ASSERT_NEAR(meanOriginal[2], meanResized[2], 1.0);
}
