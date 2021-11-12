import os.path
import sys

from face_align import FaceAligner
from eye_detection import eye_detect
import numpy as np
import cv2 as cv


def norm_to_gray(x: np.ndarray):
    xx = 255 * (x - np.min(x)) / (np.max(x) - np.min(x))
    return xx.astype('uint8')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("you may want to run `myreconstruct.py <input_img> <model_file>")
        exit(0)
    args = sys.argv[1:]
    input_img, model_file = args[0], args[1]
    if not os.path.exists(input_img):
        print("input image not exist")
        exit(0)
    if not os.path.exists(model_file):
        print("model file not exist")
        exit(0)


    model_info = np.load(model_file)
    A = model_info["eigenvector"]  # (4800, new_dim)
    mu = model_info["mu"]
    print(f"loaded model: {model_info.files}")

    img = cv.imread(input_img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    eye_left, eye_right = eye_detect(img_gray)
    aligner = FaceAligner()
    img_gray = aligner.align(img_gray,
                              (int(eye_left[0]), int(eye_left[1])),
                              (int(eye_right[0]), int(eye_right[1])))
    img_gray = cv.equalizeHist(img_gray)

    pcs = [10, 25, 50, 100, 160]
    images = []
    for pc in pcs:
        # reconstruct
        A_select = A[:, :pc]  #(4800, pc)
        M = np.dot(A_select, A_select.T) #(4800, 4800)
        y_hat = np.dot(M, img_gray.flatten()-mu) + mu
        # map reconstructed image to gray scale
        y_hat = norm_to_gray(y_hat.reshape((80, 60)))
        y_hat = cv.resize(y_hat, (120, 160), interpolation=cv.INTER_LINEAR)
        images.append(y_hat)
    images.append(cv.resize(img_gray, (120, 160), interpolation=cv.INTER_LINEAR))
    cv.imshow("reconstruct", np.hstack(images))
    cv.waitKey(0)