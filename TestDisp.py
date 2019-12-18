from os import makedirs
from os.path import splitext

import numpy as np
import cv2

from Modules.SSD2 import calcSSD
from Modules.ZNCC2 import calcZNCC

base_dataset_directory = "../Datasets/"
base_output_directory = "../Outputs/Disparity/"
ref_images = ["im0.ppm","im0.ppm", "im0.ppm", "im0.ppm",
              "im0.pgm", "scene1.row3.col1.ppm", "im0.ppm",
              "im0.ppm", "im0.ppm", "scene1.row3.col1.ppm",
              "im0.ppm"]
dataset_dir = ["cones/","barn1/", "barn2/", "bull/",
               "map/", "ohta/ohta/", "poster/",
               "sawtooth/", "teddy-ppm-9/teddy/", "tsukuba/",
               "venus/"]
test_images = [["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["im1.pgm"],
               ["scene1.row3.col1.ppm", "scene1.row3.col2.ppm", "scene1.row3.col3.ppm", "scene1.row3.col4.ppm", "scene1.row3.col5.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"],
               ["scene1.row3.col1.ppm", "scene1.row3.col2.ppm", "scene1.row3.col3.ppm", "scene1.row3.col4.ppm", "scene1.row3.col5.ppm"],
               ["im1.ppm", "im2.ppm", "im3.ppm", "im4.ppm", "im5.ppm", "im6.ppm", "im7.ppm", "im8.ppm"]]
disparity_gt = [[None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                ["disp1.pgm"],
                [None, None, None, None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None],
                [None, None, "truedisp.row3.col3.pgm", None, None],
                [None, "disp2.pgm", None, None, None, "disp6.pgm", None, None]]

datasetsToDo= [False,False,False,False,False,False,False,False,True,False,False]
maxDisparities = [31,31,31,31,31,31,31,31,31,31,31]
maxDisparities2 = [91,31,31,31,31,31,31,31,91,31,31]
dispScales = [8,8,8,8,8,8,8,8,8,8,8]
dispScales2 = [2.8,8,8,8,8,8,8,8,2.8,8,8]


errorDict = {}

i=0
j=0

for i in range(len(dataset_dir)):
    if not datasetsToDo[i]:
        continue
    img1 = cv2.imread(base_dataset_directory+dataset_dir[i]+ref_images[i])

    errorsi={}
    firstD = True
    for j in range(len(test_images[i])):
        if disparity_gt[i][j] is None:
            continue
        ground_truth_disparity = cv2.imread(
            base_dataset_directory + dataset_dir[i] + disparity_gt[i][j], cv2.IMREAD_GRAYSCALE
        )
        errorsj = {}
        img2 = cv2.imread(base_dataset_directory+dataset_dir[i]+test_images[i][j])
        mdisp = maxDisparities[i] if firstD else maxDisparities2[i]
        dispS = dispScales[i] if firstD else dispScales2[i]
        firstD = False
        disparityMap = calcSSD(img1, img2, 0, mdisp, dispS)
        makedirs(base_output_directory + dataset_dir[i], exist_ok=True)
        cv2.imwrite(base_output_directory + dataset_dir[i] + "SSD_" + splitext(test_images[i][j])[0]+".png", disparityMap)
        cv2.imwrite(base_output_directory + dataset_dir[i] + "im1_" + splitext(ref_images[i])[0]+".png", img1)
        cv2.imwrite(base_output_directory + dataset_dir[i] + "im2_" + splitext(test_images[i][j])[0]+".png", img2)
        cv2.imwrite(base_output_directory + dataset_dir[i] + "GTDisp_" + splitext(test_images[i][j])[0]+".png", ground_truth_disparity)

        diff = np.abs(disparityMap[30:-30, 30:-30].astype(np.float32) - ground_truth_disparity[34:-34, 34:-34])
        ERR1 = 100 * np.count_nonzero(diff > dispS) / diff.size
        ERR2 = 100 * np.count_nonzero(diff > 2 * dispS) / diff.size
        ERR3 = 100 * np.count_nonzero(diff > 3 * dispS) / diff.size
        ERR5 = 100 * np.count_nonzero(diff > 5 * dispS) / diff.size
        print(ERR1,ERR2,ERR3,ERR5)

        diff_IMG = diff.astype(np.uint8)
        cv2.imwrite(base_output_directory + dataset_dir[i] + "ssdDif_" + splitext(test_images[i][j])[0]+".png", diff_IMG)
        errorsj["SSD"]=[ERR1,ERR2,ERR3,ERR5]

        disparityMap = calcZNCC(img1, img2, 0, mdisp, dispS)
        cv2.imwrite(base_output_directory + dataset_dir[i] + "ZNCC_" + splitext(test_images[i][j])[0]+".png", disparityMap)
        diff = np.abs(disparityMap[30:-30, 30:-30].astype(np.float32) - ground_truth_disparity[34:-34, 34:-34])
        ERR1 = 100 * np.count_nonzero(diff > dispS) / diff.size
        ERR2 = 100 * np.count_nonzero(diff > 2 * dispS) / diff.size
        ERR3 = 100 * np.count_nonzero(diff > 3 * dispS) / diff.size
        ERR5 = 100 * np.count_nonzero(diff > 5 * dispS) / diff.size
        errorsj["ZNCC"]=[ERR1,ERR2,ERR3,ERR5]

        diff_IMG = diff.astype(np.uint8)
        cv2.imwrite(base_output_directory + dataset_dir[i] + "znccDif_" + splitext(test_images[i][j])[0]+".png", diff_IMG)
        errorsi[test_images[i][j]] = errorsj
    errorDict[dataset_dir[i]] = errorsi
print(errorDict)


print("__________________________________________________________________________________________________________________________")
for datasetname in errorDict:
    derrs = errorDict[datasetname]
    for imname in derrs:
        print(datasetname+'&'+imname+"&%.1f&%.1f&%.1f&%.1f\\\\\\hline"%(derrs[imname]['SSD'][0], derrs[imname]['SSD'][1], derrs[imname]['SSD'][2], derrs[imname]['SSD'][3]))
print("__________________________________________________________________________________________________________________________")
for datasetname in errorDict:
    derrs = errorDict[datasetname]
    for imname in derrs:
        print(datasetname+'&'+imname+"&%.1f&%.1f&%.1f&%.1f\\\\\\hline"%(derrs[imname]['ZNCC'][0], derrs[imname]['ZNCC'][1], derrs[imname]['ZNCC'][2], derrs[imname]['ZNCC'][3]))