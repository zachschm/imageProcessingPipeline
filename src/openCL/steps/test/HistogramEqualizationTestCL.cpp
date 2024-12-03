#include "Image.h"
#include "OpenCLHistogramEqualizationStep.h"
#include "OpenCLManager.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

TEST(OpenCLHistogramEqualizationStep, MultiGPURun)
{
    OpenCLManager manager;
    OpenCLHistogramEqualizationStep step(manager);

    cv::Mat grayscaleImage(400, 400, CV_8UC1, cv::Scalar(128));
    Image img(grayscaleImage);

    step.process(img);

    cv::Mat processedImage = img.getImage();

    ASSERT_EQ(processedImage.rows, grayscaleImage.rows);
    ASSERT_EQ(processedImage.cols, grayscaleImage.cols);

    ASSERT_NE(cv::mean(processedImage)[0],
              128);  // Ensure histogram adjustment occurred
}
