import csv
import pathlib

import numpy as np
import dlib
import cv2 as cv

from learn_eigenface.eye_detection import eye_detect
from learn_eigenface.face_align import FaceAligner

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    img_dir = "att-face/s1"
    aligner = FaceAligner()

    for p in pathlib.Path(img_dir).iterdir():
        img = cv.imread(str(p))
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        eye_left, eye_right = eye_detect(img_gray)
        img_gray = aligner.align(img_gray,
                                 (int(eye_left[0]), int(eye_left[1])),
                                 (int(eye_right[0]), int(eye_right[1])),
                                 corp=False)
        img_gray = cv.equalizeHist(img_gray)
        # img = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
        cv.imwrite(str(p).removesuffix(".jpg")+".pgm", img_gray)