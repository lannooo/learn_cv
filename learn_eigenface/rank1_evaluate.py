import csv
import time

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from face_align import FaceAligner
from learn_eigenface.mytrain import pca

if __name__ == '__main__':
    aligner = FaceAligner()
    images_map = {}
    train_images = []
    test_images = []
    test_labels = []
    train_labels = []
    with open("data_marker.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = cv.imread(row["file"])
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            left_eye = (int(row["eye1_x"]), int(row["eye1_y"]))
            right_eye = (int(row["eye2_x"]), int(row["eye2_y"]))
            img = cv.equalizeHist(aligner.align(img, left_eye, right_eye))  # histogram equalization

            if row["id"] not in images_map.keys():
                images_map[row["id"]] = []

            if len(images_map[row["id"]]) < 5:
                # train list
                images_map[row["id"]].append(row['file'])
                train_images.append(img.flatten())
                train_labels.append(int(row["id"].removeprefix("s")))
            else:
                # test list
                images_map[row["id"]].append(row['file'])
                test_images.append(img.flatten())
                test_labels.append(int(row["id"].removeprefix("s")))
    train_x = np.array(train_images)
    test_x = np.array(test_images)
    eigvalue, A, mu = pca(train_x, 10, threshold=0.95)
    pcs = eigvalue.shape[0]
    accs = []
    for pc in range(1, pcs + 1):
        B = A[:, :pc]
        y_ref = np.dot(B.T, (train_x - mu).T)
        y_test = np.dot(B.T, (test_x - mu).T)
        right = 0
        for i in range(y_test.shape[1]):
            y = y_test[:, i].reshape((-1, 1))
            distance = np.sum((y_ref - y) ** 2, axis=0)
            index = np.argmin(distance).item()
            if test_labels[i] == train_labels[index]:
                right += 1
        accs.append(right / y_test.shape[1])
    pcs_x = np.arange(1, pcs+1)
    acc_y = np.array(accs) * 100
    plt.plot(pcs_x, acc_y)
    plt.xlabel("PCs")
    plt.ylabel("rank-1 rate (%)")
    plt.savefig("rank1_pc.png")
    plt.show()
