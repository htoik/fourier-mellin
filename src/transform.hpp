#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

#include <opencv2/core/mat.hpp>
#include <ostream>

class Transform {
   public:
    Transform(double xOffset = 0.0, double yOffset = 0.0, double scale = 1.0, double rotationDeg = 0.0, double response = 1.0);

    // TODO: This assumes `matrix` is properly defined 2D transformation matrix
    Transform(const cv::Mat& matrix, double response);

    Transform GetInverse() const;

    cv::Mat GetMatrixInverse() const;
    cv::Mat GetMatrix() const;

    void SetOffsetX(double x);
    void SetOffsetY(double y);
    void SetScale(double scale);
    void SetRotation(double rotationDeg);
    void SetResponse(double response);

    double GetOffsetX() const;
    double GetOffsetY() const;
    double GetScale() const;
    double GetRotation() const;
    double GetResponse() const;

    Transform operator*(const Transform& rhs) const;
    Transform& operator*=(const Transform& rhs);

   private:
    double xOffset_;
    double yOffset_;
    double scale_;
    double rotation_;
    double response_;
};

std::ostream& operator<<(std::ostream& os, const Transform& t);

#endif  // __TRANSFORM_H__