import numpy as np
import cv2


def calcSSD(img1, img2, min_disp, max_disp, disp_scale, W=4):
    actualW = 2 * W + 1
    thresh = 255*3*actualW*actualW/3
    V = W + np.arange(img1.shape[0] - 2 * W)
    U = W + np.arange(img1.shape[1] - 2 * W)

    WindowVectorV = np.arange(-W, W + 1).repeat(2 * W + 1)
    WindowVectorU = np.tile(np.arange(-W, W + 1), 2 * W + 1)

    Wv = V.reshape(-1, 1) + WindowVectorV
    Wu = U.reshape(-1, 1) + WindowVectorU

    ordered_img1 = img1[Wv[:, None], Wu[None, :]]
    ordered_img2 = img2[Wv[:, None], Wu[None, :]]

    disparityMap = np.zeros(ordered_img1.shape[:2], dtype=np.uint8)

    for row in range(len(ordered_img1[0])):
        SSD = np.sum(np.square(ordered_img1[:, (row,), None] - ordered_img2[:, None, max(row-max_disp, 0):max(row-min_disp+1, 1)]), axis=(3,4))
        SSD.shape = SSD.shape[0], -1
        best_i = np.argmin(SSD, axis=1)
        disparityArray = row-np.arange(max(row-max_disp, 0), max(row-min_disp+1, 1))
        best_i_scaled = (disparityArray[best_i]*disp_scale).astype(np.uint8)
        disparityMap[:, row] = np.where(SSD[np.arange(SSD.shape[0]), best_i] < thresh, best_i_scaled.flat, 0)
        # disparityMap[:, row] = best_i.flat
        #print(row)
    return disparityMap

"""
img1 = cv2.imread("../Datasets/tsukuba/scene1.row3.col1.ppm")
img2 = cv2.imread("../Datasets/tsukuba/scene1.row3.col3.ppm")
disp = cv2.imread("../Datasets/tsukuba/truedisp.row3.col3.pgm", cv2.IMREAD_GRAYSCALE)

disparityMap = calcSSD(img1, img2, 0, 32, 8)
cv2.imshow("asdf", disparityMap)
cv2.imshow("im1", img1)
cv2.imshow("im2", img2)
cv2.imshow("disp", disp)
diff = disparityMap[15:-15,15:-15]-disp[18:-18,18:-18]
ERR1 = 100*np.count_nonzero(diff>8)/diff.size
ERR2 = 100*np.count_nonzero(diff>2*8)/diff.size
ERR3 = 100*np.count_nonzero(diff>3*8)/diff.size
ERR5 = 100*np.count_nonzero(diff>5*8)/diff.size
cv2.waitKey(0)
"""