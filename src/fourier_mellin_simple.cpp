#include "fourier_mellin_simple.hpp"

FourierMellinSimple::FourierMellinSimple(const cv::Mat& reference) : referenceImg_(reference.clone()) {
    // TODO: assumes one channel, and normalized float, like other constructor
    // if (referenceImg_.depth() != CV_32F) {
    //     referenceImg_.convertTo(referenceImg_, CV_32F, 1.0 / 255.0);
    // }

    width_ = referenceImg_.size().width;
    height_ = referenceImg_.size().height;
    logPolarMap_ = createLogPolarMap(width_, height_);

    referenceImgLogPolar_ = ConvertImageToLogPolar(referenceImg_);
}

FourierMellinSimple::FourierMellinSimple(std::string_view reference_fp) : FourierMellinSimple(ReadGrayscaleImageFromFile(reference_fp)) {}

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
    auto target = ReadGrayscaleImageFromFile(target_fp);
    return RegisterImage(target);
}

std::tuple<cv::Mat, Transform> FourierMellinSimple::GetRegisteredImage(const cv::Mat& target) const {
    const auto& transform = RegisterImage(target);
    const auto& aligned = getTransformed(target, transform);
    return {aligned, transform};
}

std::tuple<cv::Mat, Transform> FourierMellinSimple::GetRegisteredImage(std::string_view target_fp) const {
    auto target = ReadGrayscaleImageFromFile(target_fp);
    return GetRegisteredImage(target);
}

cv::Mat FourierMellinSimple::ConvertImageToLogPolar(const cv::Mat& img) const {
    cv::Mat log_polar;
    cv::remap(img, log_polar, logPolarMap_.xMap, logPolarMap_.yMap, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, cv::Scalar());
    return log_polar;
}

cv::Mat FourierMellinSimple::PreprocessImage(const cv::Mat& img) const {
    return img;
}

cv::Mat FourierMellinSimple::ReadGrayscaleImageFromFile(std::string_view img_fp) const {
    cv::Mat img = cv::imread(std::string(img_fp), cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    return img;
}
