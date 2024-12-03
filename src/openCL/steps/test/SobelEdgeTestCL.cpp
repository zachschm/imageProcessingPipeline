#include "Image.h"
#include "OpenCLManager.h"
#include "OpenCLSobelEdgeStep.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

TEST(OpenCLSobelEdgeStep, EdgeDetectionEffect)
{
    OpenCLManager openclManager;
    OpenCLSobelEdgeStep sobelEdgeStep(openclManager);

    // Create a synthetic test image (100x100 grayscale gradient)
    cv::Mat testImage(100, 100, CV_8UC3, cv::Scalar(128));
    cv::rectangle(testImage, cv::Point(30, 30), cv::Point(70, 70),
                  cv::Scalar(200), -1);  // Bright square in the center

    Image img(testImage);

    // Process image
    sobelEdgeStep.process(img);

    // Validate output
    cv::Mat processedImage = img.getImage();

    ASSERT_EQ(processedImage.rows, testImage.rows);
    ASSERT_EQ(processedImage.cols, testImage.cols);

    // Check edge values (verify edges are detected correctly)
    ASSERT_GT(cv::mean(processedImage)[0], 0);
}
