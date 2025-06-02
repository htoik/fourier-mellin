import cv2
import matplotlib.pyplot as plt
import numpy as np
from fourier_mellin import FourierMellin

img0 = cv2.imread("resources/dog_reference.png", cv2.IMREAD_COLOR)
img1 = cv2.imread("resources/dog_t01.png", cv2.IMREAD_COLOR)

fm = FourierMellin(*img0.shape[1::-1])
img_transformed, transform = fm.register_image(img1, img0)

print(f"Offset x,y: {transform.x():0.1f}, {transform.y():0.1f}")
print(f"Rotation in degrees: {transform.rotation():0.1f}")
print(f"Scale factor: {transform.scale():0.1f}")
print(f"Response/Quality: {transform.response():0.2f}")

img0 = img0.astype(np.float32)
img1 = img1.astype(np.float32)

img_overlay = cv2.cvtColor(img0 * 0.5 + img_transformed * 0.5, cv2.COLOR_RGB2BGR) * (1 / 255)
img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR) * (1 / 255)
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR) * (1 / 255)

cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR)

plt.figure()
plt.subplot(2, 2, 1)
plt.title("Reference image")
plt.imshow(img0)
plt.subplot(2, 2, 2)
plt.title("Image")
plt.imshow(img1)
plt.subplot(2, 1, 2)
plt.title("Registered & Overlayed")
plt.imshow(img_overlay)
plt.show()
