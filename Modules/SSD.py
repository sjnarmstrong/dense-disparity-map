import numpy as np
import cv2

img1 = cv2.imread("../Datasets/tsukuba/scene1.row3.col1.ppm")
img2 = cv2.imread("../Datasets/tsukuba/scene1.row3.col3.ppm")
disp = cv2.imread("../Datasets/tsukuba/truedisp.row3.col3.pgm", cv2.IMREAD_GRAYSCALE)


W = 5
actualW = 2 * W + 1
V = W + np.arange(img1.shape[0] - 2 * W)
U = W + np.arange(img1.shape[1] - 2 * W)

WindowVectorV = np.arange(-W, W + 1).repeat(2 * W + 1)
WindowVectorU = np.tile(np.arange(-W, W + 1), 2 * W + 1)

Wv = V.reshape(-1, 1) + WindowVectorV
Wu = U.reshape(-1, 1) + WindowVectorU

ordered_img1 = img1[Wv[:, None], Wu[None, :]]
ordered_img2 = img2[Wv[:, None], Wu[None, :]]

SSD = np.sum(np.square(ordered_img1[:, :, None] - ordered_img2[:, None]), axis=(3,4))

best_i = np.argmin(SSD, axis=2)
best_i = (best_i*255/best_i.max()).astype(np.uint8)


cv2.imshow("asdf", best_i)
cv2.imshow("im1", img1)
cv2.imshow("im2", img2)
cv2.imshow("disp", disp)
cv2.waitKey(0)
