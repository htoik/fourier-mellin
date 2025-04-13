# Fourier-Mellin Python Library using OpenCV

This repository implements the Fourier-Mellin transform for image registration and video stabilization using semilog polar coordinates with OpenCV. The implementation is written in C++, but python bindings are provided. For information about the pipeline, see [this article](https://ieeexplore.ieee.org/document/506761). Tested on Ubuntu 20.04.

### Disclaimer

This repository is an unstable release and not intended for production. Many other repositories have implemented the Fourier-Mellin transform, for example [imreg_fmt by sthoduka](https://github.com/sthoduka/imreg_fmt) or [fourier-mellin by polakluk](https://github.com/polakluk/fourier-mellin).

Note that the image registration only works effectively for Eucledian/similar transformations without affine or perspective distortions.

## Video Stabilization

https://github.com/brht0/fourier-mellin/assets/90235713/c5524042-6bc6-46b5-837e-adb62ddcf9b1

Original video can be found [on Youtube](https://www.youtube.com/watch?v=mQxnB2X26CI) (No affiliation)

## Image Registration

![lena_transform_demo](https://github.com/brht0/fourier-mellin/assets/90235713/9db0cc26-581f-40db-b34c-3516e759f960)

Image registration with a transformed image, overlayed.

## Installation with `git clone`

It is recommended to use a python virtual environment. The repository will be added to PyPI in time.

```
# inside your own project
python3 -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/brht0/fourier-mellin
```

## Examples

Many examples, such as video stabilization, are included inside the `examples/` subdirectory. The following example registers two images.

```python
import cv2
import numpy as np
import fourier_mellin

reference = cv2.imread('lenna.png')
transformed = cv2.imread('lenna_transformed.png')

rows, cols = reference.shape[:2]
fm = fourier_mellin.FourierMellin(cols, rows)

transformed_reference, transform = fm.register_image(reference, transformed)
overlay = cv2.addWeighted(transformed, 0.5, transformed_reference, 0.5, 0.0, dtype=cv2.CV_32F)

cv2.imwrite("overlay.jpg", overlay)
cv2.imwrite("transformed.jpg", transformed)
```

## Building without pip

Building without pip is not required for use with python. Building without pip requires installing additional dependencies, such as pybind11. This step may be skipped, in case only python bindings are used.

```
mkdir -p build/release
cd build/release
cmake ../.. -DCMAKE_BUILD_TYPE=Release
cd ../..
cmake --build build/release -j 4
```

## Todo

- Register images directly from files
- Windows/MacOS support
- Benchmarking
- Optimization
- CUDA with OpenCV
- cv::phaseCorrelate already applies Hanning Window
- Documentation
- Proper threading support
- angles beyond -90 to 90 deg
