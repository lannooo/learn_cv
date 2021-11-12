import numpy as np
import cv2 as cv


class FaceAligner:
    def __init__(self, desiredLeftEye=(1.0/3, 0.375), desiredFaceWidth=92, desiredFaceHeight=112):
        # eye position, and desired output face width + height
        self.left_eye = desiredLeftEye
        self.width = desiredFaceWidth
        self.height = desiredFaceHeight
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.height is None:
            self.height = self.width

    def align(self, image, left_eye_center: tuple, right_eye_center: tuple, corp=True):
        # calculate angle
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        # calculate scale
        right_eye = 1.0 - self.left_eye[0]
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        to_dist = (right_eye - self.left_eye[0]) * self.width
        scale = to_dist / dist
        # get affine transform matrix
        center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
        M = cv.getRotationMatrix2D(center, angle, scale)
        tx = self.width * 0.5
        ty = self.height * self.left_eye[1]
        M[0, 2] += (tx - center[0])
        M[1, 2] += (ty - center[1])
        # transform and corp image
        (w, h) = (self.width, self.height)
        output = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC)
        if corp:
            return output[16:-16, 16:-16]  # corp to shape (80, 60)
        else:
            return output
