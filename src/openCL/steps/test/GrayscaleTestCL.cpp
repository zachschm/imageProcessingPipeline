#include "Image.h"
#include "OpenCLGrayscaleStep.h"
#include "OpenCLManager.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

TEST(OpenCLGrayscaleStep, MultiGPURun)
{
    OpenCLManager manager;
    OpenCLGrayscaleStep grayscaleStep(manager);

    cv::Mat colorImage(400, 400, CV_8UC3, cv::Scalar(128, 64, 32));
    Image img(colorImage);

    grayscaleStep.process(img);

    cv::Mat processedImage = img.getImage();

    ASSERT_EQ(processedImage.rows, colorImage.rows);
    ASSERT_EQ(processedImage.cols, colorImage.cols);

    cv::Scalar meanOriginal = cv::mean(colorImage);
    cv::Scalar meanProcessed = cv::mean(processedImage);

    ASSERT_NE(meanOriginal[0], meanProcessed[0]);
}
