import csv
import pathlib

import numpy as np
import dlib
import cv2 as cv


def eye_detect(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    rects = detector(img, 2)
    for rect in rects:
        shape = predictor(img, rect)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np
        eye_left = np.mean(shape[36:42], axis=0).astype("int")
        eye_right = np.mean(shape[42:48], axis=0).astype("int")
        return eye_left, eye_right
    return None, None


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    img_dir = "att-face"
    imgpath = pathlib.Path(img_dir)
    images = []
    points = []
    header = ["file", "id", "eye1_x", "eye1_y", "eye2_x", "eye2_y"]
    lines = []
    for p in imgpath.iterdir():
        if not p.is_dir(): continue
        images_h = []
        for img_p in p.iterdir():
            img = cv.imread(str(img_p))
            if len(img.shape) == 3:
                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            else:
                img_gray = img
            assert len(img_gray.shape) == 2

            rects = detector(img_gray, 2)
            if len(rects) == 0:
                # mark eye position by hand, click on image and save them

                # define mouse click callback
                def onMouseCallback(event, x, y, flags, param):
                    if event == cv.EVENT_LBUTTONDOWN:
                        cv.circle(img, (x, y), 2, (0, 0, 255), -1)
                        points.append((x, y))
                        cv.imshow("detect", img)
                # register into window
                cv.namedWindow("detect")
                cv.setMouseCallback("detect", onMouseCallback)

                print("manually setting: " + str(img_p))
                points.clear()
                cv.imshow("detect", img)
                while cv.waitKey(0) != ord('s'):
                    pass
                print("catched: " + str(points))
                points.sort()  # from left eye to right order, make sure there are only 2 points here
                lines.append([str(img_p), str(p.name), points[0][0], points[0][1], points[1][0], points[1][1]])
            else:
                for rect in rects:
                    shape = predictor(img_gray, rect)
                    # convert to numpy position
                    shape_np = np.zeros((68, 2), dtype="int")
                    for i in range(0, 68):
                        shape_np[i] = (shape.part(i).x, shape.part(i).y)
                    shape = shape_np
                    # calculate eye position
                    eye_left = np.mean(shape[36:42], axis=0).astype("int")
                    eye_right = np.mean(shape[42:48], axis=0).astype("int")
                    # draw red spots on eyes
                    cv.circle(img, eye_left, 2, (0, 0, 255), -1)
                    cv.circle(img, eye_right, 2, (0, 0, 255), -1)
                    # new line written into csv
                    lines.append([str(img_p), str(p.name), eye_left[0], eye_left[1], eye_right[0], eye_right[1]])
                    # detect only once
                    break
            images_h.append(img)
        images.append(np.hstack(images_h))
    # write to data_marker.csv
    with open("data_marker.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(lines)
    cv.imshow("images", np.vstack(images))
    cv.waitKey(0)

