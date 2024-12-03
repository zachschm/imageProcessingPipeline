#include "Image.h"
#include "OpenCLManager.h"
#include "OpenCLSaturationStep.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

TEST(OpenCLSaturationStep, AdjustSaturationEffect)
{
    OpenCLManager openclManager;
    OpenCLSaturationStep saturationStep(openclManager,
                                        1.5f);  // Increase saturation by 50%

    // Create a synthetic test image (100x100 RGB gradient)
    cv::Mat testImage(100, 100, CV_8UC3);
    for (int i = 0; i < testImage.rows; ++i)
    {
        for (int j = 0; j < testImage.cols; ++j)
        {
            testImage.at<cv::Vec3b>(i, j) = cv::Vec3b(j, 100, 255 - j);
        }
    }

    Image img(testImage);

    // Process image
    saturationStep.process(img);

    // Validate output
    cv::Mat processedImage = img.getImage();

    ASSERT_EQ(processedImage.rows, testImage.rows);
    ASSERT_EQ(processedImage.cols, testImage.cols);

    // Check that saturation adjustment is applied
    cv::Mat hsvOriginal, hsvProcessed;
    cv::cvtColor(testImage, hsvOriginal, cv::COLOR_BGR2HSV);
    cv::cvtColor(processedImage, hsvProcessed, cv::COLOR_BGR2HSV);

    for (int i = 0; i < hsvOriginal.rows; ++i)
    {
        for (int j = 0; j < hsvOriginal.cols; ++j)
        {
            float originalS = hsvOriginal.at<cv::Vec3b>(i, j)[1] / 255.0f;
            float processedS = hsvProcessed.at<cv::Vec3b>(i, j)[1] / 255.0f;
            ASSERT_NEAR(processedS, std::min(originalS * 1.5f, 1.0f), 0.05f);
        }
    }
}
