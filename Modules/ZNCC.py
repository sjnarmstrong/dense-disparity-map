import numpy as np
import cv2

img1 = cv2.imread("../Datasets/tsukuba/scene1.row3.col1.ppm")
img2 = cv2.imread("../Datasets/tsukuba/scene1.row3.col3.ppm")
disp = cv2.imread("../Datasets/tsukuba/truedisp.row3.col3.pgm", cv2.IMREAD_GRAYSCALE)


W = 2
actualW = 2 * W + 1
V = W + np.arange(img1.shape[0] - 2 * W)
U = W + np.arange(img1.shape[1] - 2 * W)

WindowVectorV = np.arange(-W, W + 1).repeat(2 * W + 1)
WindowVectorU = np.tile(np.arange(-W, W + 1), 2 * W + 1)

Wv = V.reshape(-1, 1) + WindowVectorV
Wu = U.reshape(-1, 1) + WindowVectorU

ordered_img1 = img1[Wv[:, None], Wu[None, :]]
ordered_img2 = img2[Wv[:, None], Wu[None, :]]

mean_img_1 = (ordered_img1 - np.mean(ordered_img1, axis=2)[:, :, None])#/np.std(ordered_img1, axis=2)[:, :, None]
mean_img_2 = (ordered_img2 - np.mean(ordered_img2, axis=2)[:, :, None])#/np.std(ordered_img2, axis=2)[:, :, None]

scales = 6*actualW * actualW
NCC = np.einsum("ijlm,iklm->ijk", mean_img_1, mean_img_2)/scales

best_i = np.argmax(NCC, axis=2)
best_i = (best_i*255/best_i.max()).astype(np.uint8)


cv2.imshow("asdf", best_i)
cv2.imshow("im1", img1)
cv2.imshow("im2", img2)
cv2.imshow("disp", disp)
cv2.waitKey(0)
