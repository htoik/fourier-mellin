import cv2
from fourier_mellin import FourierMellin

img0_path = "resources/road2_clear.png"
img1_path = "resources/road2_dark.png"
img2_path = "resources/road2_snow.png"

img_ref = cv2.imread(img0_path, cv2.IMREAD_COLOR)
img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

fm = FourierMellin(*img_ref.shape[1::-1])
_, t1 = fm.register_image(img_ref, img1)
_, t2 = fm.register_image(img_ref, img2)

print(f"Transform: {t1.x():0.1f}, {t1.y():0.1f}, {t1.rotation():0.1f}, {t1.scale():0.1f}, {t1.response():0.1f}")
print(f"Transform: {t2.x():0.1f}, {t2.y():0.1f}, {t2.rotation():0.1f}, {t2.scale():0.1f}, {t2.response():0.1f}")

print(t1.to_dict())
print(t2.to_dict())
