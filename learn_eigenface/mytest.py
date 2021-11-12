import os.path
import sys

from face_align import FaceAligner
from eye_detection import eye_detect
import csv
import numpy as np
import cv2 as cv


def desc_y(y:np.ndarray):
    n = y.shape[0]
    result = []
    result.append("y=[")
    line = ''
    for i in range(n):
        v = y.item(i)
        line += "{:>10.3f},".format(v)
        if (i+1) % 5 == 0:
            result.append(line)
            line = ''
    if len(line) != 0:
        result.append(line)
    result.append("]")
    return result

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("you may want to run `mytest.py <input_img> <model_file> <train_data_csv>")
        exit(0)
    args = sys.argv[1:]
    input_img, model_file, data_list = args[0], args[1], args[2]
    if not os.path.exists(input_img):
        print("input image not exist")
        exit(0)
    if not os.path.exists(model_file):
        print("model file not exist")
        exit(0)
    if not os.path.exists(data_list):
        print("data list not exist")
        exit(0)

    images = []
    img_tables = []
    aligner = FaceAligner()
    with open(data_list, "r", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = cv.imread(row["file"])
            left_eye = (int(row["eye1_x"]), int(row["eye1_y"]))
            right_eye = (int(row["eye2_x"]), int(row["eye2_y"]))
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            aligned = aligner.align(img, left_eye, right_eye)
            aligned = cv.equalizeHist(aligned)  # histogram equalization
            images.append(aligned.flatten())
            img_tables.append(row["file"])
    data = np.array(images)  # (k, m*n)

    model_info = np.load(model_file)
    A = model_info["eigenvector"]  # (4800, new_dim)
    mu = model_info["mu"]
    print(f"loaded model: {model_info.files}")

    img = cv.imread(input_img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    eye_left, eye_right = eye_detect(img_gray)
    img_gray = aligner.align(img_gray,
                              (int(eye_left[0]), int(eye_left[1])),
                              (int(eye_right[0]), int(eye_right[1])))
    img_gray = cv.equalizeHist(img_gray)

    # images database in feature space
    y_ref = np.dot(A.T, (data-mu).T)  # (new_dim, 4800) * (4800, k)
    # input image in feature space
    y = np.dot(A.T, img_gray.flatten()-mu)
    # calculate distance
    distance = np.sqrt(np.sum((y_ref - y.reshape((-1, 1)))**2, axis=0))
    # fetch the matched image with min distance
    index = np.argmin(distance).item()
    matched_img = cv.imread(img_tables[index])

    # prepare images to show
    (h, w) = img.shape[:2]
    h, w = h*2, w*2
    img = cv.resize(img, (w, h), interpolation=cv.INTER_LINEAR)
    matched_img = cv.resize(matched_img, (w, h), interpolation=cv.INTER_LINEAR)
    # draw result image
    back = np.empty((800, 800, 3), np.uint8)
    back[:, :] = (255, 255, 255)
    font = cv.FONT_HERSHEY_SIMPLEX
    color = (0,0,0)
    back[50: h+50, 0:w] = img
    back[h+100:2*h+100, 0:w] = matched_img
    cv.putText(back, 'input', (10, 30), font, 1.0, color, 2)
    cv.putText(back, 'matched', (10, 310), font, 1.0, color, 2)
    y_list = desc_y(y)
    for i, line in enumerate(y_list):
        cv.putText(back, line, (300, 30+20*i), font, 0.5, (10,10,250), 2)
    cv.imshow("test result", back)
    cv.waitKey(0)



