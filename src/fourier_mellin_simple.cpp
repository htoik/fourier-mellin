#include "fourier_mellin_simple.hpp"

FourierMellinSimple::FourierMellinSimple(std::string_view reference_fp) {
    referenceImg_ = cv::imread(std::string(reference_fp), cv::IMREAD_GRAYSCALE);
    referenceImg_.convertTo(referenceImg_, CV_32F, 1.0 / 255.0);

    width_ = referenceImg_.size().width;
    height_ = referenceImg_.size().height;
    logPolarMap_ = createLogPolarMap(width_, height_);

    referenceImgLogPolar_ = ConvertImageToLogPolar(referenceImg_);
}

Transform FourierMellinSimple::RegisterImage(const cv::Mat& target) const {
    const auto& logPolar1 = ConvertImageToLogPolar(target);

    double responseScaleRotation;
    auto [logScale, logRotation] = cv::phaseCorrelate(logPolar1, referenceImgLogPolar_, cv::noArray(), &responseScaleRotation);
    double rotation = -logRotation / logPolarMap_.logPolarSize * 180.0;
    double scale = std::pow(logPolarMap_.logBase, -logScale);

    const auto center = cv::Point(width_, height_) / 2.0;
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, rotation, scale);
    cv::Mat rotated0;
    cv::warpAffine(referenceImg_, rotated0, rotationMatrix, referenceImg_.size());

    double response;
    auto [xOffset, yOffset] = cv::phaseCorrelate(target, rotated0, cv::noArray(), &response);

    return Transform{-xOffset, yOffset, scale, rotation, response};
}

Transform FourierMellinSimple::RegisterImage(std::string_view target_fp) const {
    cv::Mat img1 = cv::imread(std::string(target_fp), cv::IMREAD_GRAYSCALE);
    img1.convertTo(img1, CV_32F, 1.0 / 255.0);
    return RegisterImage(img1);
}

std::tuple<cv::Mat, Transform> FourierMellinSimple::GetRegisteredImage(std::string_view target_fp) const {
}

cv::Mat FourierMellinSimple::ConvertImageToLogPolar(const cv::Mat& img) const {
    cv::Mat log_polar;
    // cv::remap(img, log_polar, logPolarMap_.xMap, logPolarMap_.yMap, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
    cv::remap(img, log_polar, logPolarMap_.xMap, logPolarMap_.yMap, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, cv::Scalar());
    return log_polar;
}
