import numpy as np
import cv2


def calcZNCC(img1, img2, min_disp=0, max_disp=32, disp_scale=8, W=4, thresh=0.1):

    actualW = 2 * W + 1
    V = W + np.arange(img1.shape[0] - 2 * W)
    U = W + np.arange(img1.shape[1] - 2 * W)

    WindowVectorV = np.arange(-W, W + 1).repeat(2 * W + 1)
    WindowVectorU = np.tile(np.arange(-W, W + 1), 2 * W + 1)

    Wv = V.reshape(-1, 1) + WindowVectorV
    Wu = U.reshape(-1, 1) + WindowVectorU

    ordered_img1 = img1[Wv[:, None], Wu[None, :]]
    ordered_img2 = img2[Wv[:, None], Wu[None, :]]

    disparityMap = np.zeros(ordered_img1.shape[:2], dtype=np.uint8)

    mean_img_1 = (ordered_img1 - np.mean(ordered_img1, axis=2)[:, :, None])/(np.std(ordered_img1, axis=2)[:, :, None]+1e-10)
    mean_img_2 = (ordered_img2 - np.mean(ordered_img2, axis=2)[:, :, None])/(np.std(ordered_img2, axis=2)[:, :, None]+1e-10)

    scales = 3*actualW * actualW

    for row in range(len(mean_img_1[0])):
        NCC = np.einsum("ilm,iklm->ik", mean_img_1[:, row], mean_img_2[:, max(row-max_disp, 0):max(row-min_disp+1, 1)]) / scales
        best_i = np.argmax(NCC, axis=1)
        disparityArray = row-np.arange(max(row-max_disp, 0), max(row-min_disp+1, 1))
        best_i_scaled = (disparityArray[best_i]*disp_scale).astype(np.uint8)
        disparityMap[:, row] = np.where(NCC[np.arange(NCC.shape[0]), best_i]>thresh, best_i_scaled.flat, 0)
    return disparityMap

'''
img1 = cv2.imread("../Datasets/tsukuba/scene1.row3.col1.ppm")
img2 = cv2.imread("../Datasets/tsukuba/scene1.row3.col3.ppm")
disp = cv2.imread("../Datasets/tsukuba/truedisp.row3.col3.pgm", cv2.IMREAD_GRAYSCALE)
cv2.imshow("asdf", disparityMap)
cv2.imshow("im1", img1)
cv2.imshow("im2", img2)
cv2.imshow("disp", disp)
cv2.waitKey(0)
'''