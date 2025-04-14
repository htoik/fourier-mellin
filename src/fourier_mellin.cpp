#include "fourier_mellin.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

FourierMellin::FourierMellin(const cv::Mat& reference) : width_(reference.size().width),
                                                         height_(reference.size().height),
                                                         logPolarMap_(width_, height_),
                                                         imageFilter_(width_, height_),
                                                         referenceImg_(reference.clone()),
                                                         referenceImgLogPolar_(ConvertImageToLogPolar(referenceImg_)) {
    // TODO: assumes one channel, and normalized float, like other constructor
    // throw error if not float, and one channel
}

FourierMellin::FourierMellin(std::string_view reference_fp) : FourierMellin(ReadGrayscaleImageFromFile(reference_fp)) {}

Transform FourierMellin::RegisterImage(const cv::Mat& target) const {
    const auto& logPolarTarget = ConvertImageToLogPolar(target);

    int logPolarSize = logPolarMap_.GetLogPolarSize();
    double logBase = logPolarMap_.GetLogBase();

    double responseScaleRotation;
    auto [logScale, logRotation] = cv::phaseCorrelate(logPolarTarget, referenceImgLogPolar_, cv::noArray(), &responseScaleRotation);
    double rotation = -logRotation / logPolarSize * 180.0;
    double scale = std::pow(logBase, logScale);

    const auto center = cv::Point(width_, height_) / 2.0;
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, rotation, scale);
    cv::Mat rotatedReference;
    cv::warpAffine(referenceImg_, rotatedReference, rotationMatrix, referenceImg_.size());

    double response;
    auto [xOffset, yOffset] = cv::phaseCorrelate(target, rotatedReference, cv::noArray(), &response);

    return Transform{-xOffset, yOffset, scale, rotation, response};
}

Transform FourierMellin::RegisterImage(std::string_view target_fp) const {
    auto target = ReadGrayscaleImageFromFile(target_fp);
    return RegisterImage(target);
}

std::tuple<cv::Mat, Transform> FourierMellin::GetRegisteredImage(const cv::Mat& target) const {
    const auto& transform = RegisterImage(target);
    const auto& aligned = getTransformed(target, transform);
    return {aligned, transform};
}

std::tuple<cv::Mat, Transform> FourierMellin::GetRegisteredImage(std::string_view target_fp) const {
    auto target = ReadGrayscaleImageFromFile(target_fp);
    return GetRegisteredImage(target);
}

cv::Mat FourierMellin::ConvertImageToLogPolar(const cv::Mat& img) const {
    auto filtered = imageFilter_.GetFilteredImage(img);
    return logPolarMap_.ConvertToLogPolar(filtered);
}

cv::Mat FourierMellin::ReadGrayscaleImageFromFile(std::string_view img_fp) const {
    cv::Mat img = cv::imread(std::string(img_fp), cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    return img;
}
