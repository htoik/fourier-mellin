#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

#include <opencv2/core/mat.hpp>
#include <ostream>

struct Transform {
  double x;
  double y;
  double scale;
  double rotation;
  double response;

  Transform GetInverse() const;
  cv::Mat GetMatrixInverse() const;
  cv::Mat GetMatrix() const;

  Transform operator*(const Transform &rhs) const;
  Transform &operator*=(const Transform &rhs);
};

std::ostream &operator<<(std::ostream &os, const Transform &t);

#endif // __TRANSFORM_H__