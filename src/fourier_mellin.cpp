#include "fourier_mellin.hpp"

#include <iostream>

FourierMellin::FourierMellin(int cols, int rows):
    cols_(cols), rows_(rows),
    highPassFilter_(getHighPassFilter(rows_, cols_)),
    apodizationWindow_(getApodizationWindow(cols_, rows_, std::min(rows, cols))),
    logPolarMap_(createLogPolarMap(cols_, rows_))
{
}

FourierMellin::~FourierMellin() {
}

cv::Mat FourierMellin::GetProcessImage(const cv::Mat &img) const {
    return getProcessedImage(img, highPassFilter_, apodizationWindow_, logPolarMap_);
}

cv::Mat convertToGrayscale(const cv::Mat& img){
    // TODO: Don't return another mat, modify 'img' instead
    if(img.channels() == 1){
        return img;
    }
    else if(img.channels() == 3){
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    else{
        throw std::runtime_error("Cannot convert to grayscale with " + std::to_string(img.channels()) + " channels.");
        return cv::Mat();
    }
}

std::tuple<cv::Mat, Transform> FourierMellin::GetRegisteredImage(const cv::Mat &img0, const cv::Mat &img1) const {
    cv::Mat gray0 = convertToGrayscale(img0);
    cv::Mat gray1 = convertToGrayscale(img1);

    auto logPolar0 = GetProcessImage(gray0);
    auto logPolar1 = GetProcessImage(gray1);

    auto transform = registerGrayImage(gray0, gray1, logPolar0, logPolar1, logPolarMap_);
    auto transformed = getTransformed(img0, transform);

    return std::make_tuple(transformed, transform);
}
