import sys

from face_align import FaceAligner
import csv
import numpy as np
from numpy import linalg
import cv2 as cv


def pca(x, pc=10, threshold=0.95):
    # be sure x shape of (k, m*n)
    mu = np.mean(x, axis=0)
    # subtract the average face
    x = x - mu
    # calculate eigenvalues and eigenvectors
    sigma = np.dot(x.T, x)
    eigenvalues, eigenvectors = linalg.eigh(sigma)
    # sort it by descending order
    idx = np.argsort(-eigenvalues)
    # get the pcs whose total energy ratio is over threshold
    cur = 0.0
    total = np.sum(eigenvalues).item()
    for i in range(idx.shape[0]):
        cur += eigenvalues[idx[i]]
        if cur / total >= threshold:
            if i+1 > pc:  # if not enough for keeped pc
                pc = i+1
            break
    idx = idx[:pc]
    eigenvalues = eigenvalues[idx].copy()
    eigenvectors = eigenvectors[:, idx].copy()
    return eigenvalues, eigenvectors, mu


def norm_to_gray(x: np.ndarray):
    xx = 255 * (x - np.min(x)) / (np.max(x) - np.min(x))
    return xx.astype('uint8')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("you may want to run `mytrain.py <energy_ratio> <model_file> <train_data_csv>")
        exit(0)
    args = sys.argv[1:]
    threshold = float(args[0])
    save_model = args[1]
    data_list = args[2]
    print("train eigenface model...")
    print(f"train_data: {data_list}")
    print(f"energy ratio: {threshold}")
    print(f"output model: {save_model}")
    images = []
    print("load and prepare images...")
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
    x = np.array(images)  # (k, m*n)

    print("prepare done, start calculate pca")
    eigvalue, eigvector, mu = pca(x, 10, threshold=threshold)
    print("pca done, saving model...")
    np.savez(save_model,
             eigenvalue=eigvalue,
             eigenvector=eigvector,
             mu=mu,
             max_pc=eigvalue.shape[0],
             dimension=eigvector.shape[0])
    print("save done.")
    images = []
    for i in range(10):
        v = eigvector[:, i]
        img = v.reshape((80, 60))
        img = norm_to_gray(img)
        img = cv.resize(img, (120, 160), interpolation=cv.INTER_LINEAR)
        images.append(img)
    cv.imshow("first 10 eigen face", np.hstack(images))
    cv.imshow("mean", cv.resize(norm_to_gray(mu.reshape((80, 60))),
                                (120, 160), interpolation=cv.INTER_LINEAR))
    cv.waitKey(0)