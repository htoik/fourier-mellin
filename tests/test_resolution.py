import cv2
import fourier_mellin


def test_resolution_1():
    class ImageRegistrationPreprocessingConfig:
        def __init__(self, imgSize, *, cropScale, maxDimension, scaleDownFactor):
            self.cropScale = cropScale
            self.maxDimension = maxDimension
            self.scaleDownFactor = scaleDownFactor
            self.imgSize = imgSize
            self.frameSize = self._getFrameSize()
            self.cropSize = (maxDimension * cropScale, maxDimension * cropScale)
            self.cropSize = tuple(int(x) for x in self.cropSize)

        def getProcessedResolution(self):
            return tuple(x // self.scaleDownFactor for x in self.cropSize)

        def _getFrameSize(self):
            if self.imgSize[0] < self.imgSize[1]:
                return (
                    self.maxDimension,
                    int(self.imgSize[1] / self.imgSize[0] * self.maxDimension),
                )
            else:
                return (
                    int(self.imgSize[0] / self.imgSize[1] * self.maxDimension),
                    self.maxDimension,
                )

        def preprocessImages(self, imgs):
            imgs = [self._preprocessImage(img) for img in imgs]
            return imgs

        def _preprocessImage(self, img):
            img = cv2.resize(img, self.frameSize, cv2.INTER_CUBIC)
            w, h = self.frameSize
            wi, hi = self.cropSize
            x = (w - wi) // 2
            y = (h - hi) // 2
            return img[y : y + hi : self.scaleDownFactor, x : x + wi : self.scaleDownFactor, :].copy()

    def convertTransformResolution(transform, input_shape, output_shape):
        pxScaler = crop_scale * output_shape[0] / (input_shape[0])
        x = transform.x * pxScaler
        y = transform.y * pxScaler
        scale = transform.scale
        rotation = transform.rotation
        return fourier_mellin.Transform(x, y, scale, rotation, transform.response)

    img1 = cv2.imread("./resources/dog_reference.png")
    img2 = cv2.imread("./resources/dog_t01.png")

    crop_scale = 0.85
    scale_down_factor = 2
    max_dimension = 640
    imreg_config = ImageRegistrationPreprocessingConfig(
        img1.shape[1::-1],
        cropScale=crop_scale,
        maxDimension=max_dimension,
        scaleDownFactor=scale_down_factor,
    )

    img1_low, img2_low = imreg_config.preprocessImages([img1, img2])
    cv2.imwrite("output/testres1_img1_processed.jpg", img1_low)
    cv2.imwrite("output/testres1_img2_processed.jpg", img2_low)

    resolution = img1.shape[:2]
    resolution_low = imreg_config.getProcessedResolution()

    fm = fourier_mellin.FourierMellin(img1_low)
    transform_low = fm.register_image(cv2.cvtColor(img2_low, cv2.COLOR_RGB2GRAY))
    img1_low_transformed = fourier_mellin.get_transformed(img1_low, transform_low)
    overlap_low = img1_low_transformed // 2 + img2_low // 2
    cv2.imwrite("output/testres1_overlap_low.jpg", overlap_low)

    transform = convertTransformResolution(transform_low, resolution_low, resolution)
    img1_transformed = fourier_mellin.get_transformed(img1, transform)
    overlap = img1_transformed // 2 + img2 // 2
    cv2.imwrite("output/testres1_overlap.jpg", overlap)
    # TODO: check MSE, but also works visually
