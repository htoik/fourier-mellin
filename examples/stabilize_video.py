import cv2
import numpy as np
from fourier_mellin import FourierMellin


def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    flag, firstFrame = cap.read()
    if not flag:
        print(f"Error opening video {input_path}")
        exit(1)
    frameShape = firstFrame.shape[1::-1]
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), frameShape)
    fm = FourierMellin(*frameShape)
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        img, t = fm.register_image(frame, firstFrame)
        # print(img.shape, t)
        out.write(img.astype(np.uint8))
    out.release()


if __name__ == "__main__":
    input_path = "resources/shaky_drone.mp4"
    output_path = "output/stabilized_shaky_drone.mp4"
    stabilize_video(input_path, output_path)
