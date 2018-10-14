# coding=utf-8
import cv2
import os
import numpy as np
import argparse
import time


# 搜寻文件
def findAllFiles(root_dir, filter):
    print("Finding files ends with \'" + filter + "\' ...")
    separator = os.path.sep
    paths = []
    names = []
    files = []
    for parent, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filter):
                paths.append(parent + separator)
                names.append(filename)
    for i in range(paths.__len__()):
        files.append(paths[i] + names[i])
    print (names.__len__().__str__() + " files have been found.")
    paths.sort()
    names.sort()
    files.sort()
    return paths, names, files


# 获取特征点
def getKeyPoints(img1, img2, flag='sift'):
    good_matches = []
    good_kps1 = []
    good_kps2 = []

    good_out = []
    good_out_kp1 = []
    good_out_kp2 = []

    if flag == 'sift':
        sift = cv2.xfeatures2d_SIFT.create()
        kp1, des1 = cv2.xfeatures2d_SIFT.detectAndCompute(sift, img1, None)
        kp2, des2 = cv2.xfeatures2d_SIFT.detectAndCompute(sift, img2, None)
    elif flag == 'surf':
        surf = cv2.xfeatures2d_SURF.create()
        kp1, des1 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img1, None)
        kp2, des2 = cv2.xfeatures2d_SURf.detectAndCompute(surf, img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 筛选
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append(matches[i])
            good_kps1.append(kp1[matches[i][0].queryIdx])
            good_kps2.append(kp2[matches[i][0].trainIdx])

    print(
            "kp1:" + kp1.__len__().__str__() + ",kp2:" + kp2.__len__().__str__() + ",good matches:" + good_matches.__len__().__str__())
    for i in range(good_kps1.__len__()):
        good_out_kp1.append([good_kps1[i].pt[0], good_kps1[i].pt[1]])
        good_out_kp2.append([good_kps2[i].pt[0], good_kps2[i].pt[1]])
        good_out.append([good_kps1[i].pt[0], good_kps1[i].pt[1], good_kps2[i].pt[0], good_kps2[i].pt[1]])
    return good_out_kp1, good_out_kp2, good_out


# 影像透明叠加
def transparentOverlay(img1, img2):
    ret, mask = cv2.threshold(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img1,
                              img1,
                              mask=mask)
    img1_fg = cv2.bitwise_and(img2[:img1.shape[0], :img1.shape[1], :],
                              img2[:img1.shape[0], :img1.shape[1], :],
                              mask=mask_inv)
    dst = cv2.add(img1_bg, img1_fg)
    return dst


parser = argparse.ArgumentParser(description='Arguments for program.')
parser.add_argument('-input', default='.', help='Input directory for images.')
parser.add_argument('-type', default='jpg', help='File type of input images.')
parser.add_argument('-output', default='.', help='Output directory for result.')
parser.add_argument('-offX', default='300', help='Offset pixels in x direction.')
parser.add_argument('-offY', default='9000', help='Offset pixels in y direction.')
parser.add_argument('-autoCut', default='1', help='1 - cut the blank part of image;0 - keep the blank part.')
parser.add_argument('-every', default='1', help='1 - output every result for images;0 - only output the final result.')
parser.add_argument('-feature', default='sift', help='sift - sift features;surf - surf features.')
args = parser.parse_args()

# 寻找影像文件
paths, names, files = findAllFiles(args.input, args.type)

# 打开影像文件并存放于list中
imgs = []
for item in files:
    imgs.append(cv2.imread(item))

height = imgs[0].shape[0]
width = imgs[0].shape[1]
OFFSET_Y = int(args.offY)
OFFSET_X = int(args.offX)

total_height = height + OFFSET_Y
total_width = width + OFFSET_X

out = np.zeros([total_height, total_width, 3], np.uint8)
sum_offset_y = 0
sum_offset_x = 0

# 对影像两两进行匹配重采
t1 = time.time()
for i in range(imgs.__len__() - 1):
    if i == 0:
        print "\nFrame " + (i + 1).__str__() + "/" + imgs.__len__().__str__()
        good_out_kp1, good_out_kp2, good_out = getKeyPoints(imgs[i + 1], imgs[i], args.feature)
        affine_matrix, mask = cv2.estimateAffine2D(np.array(good_out_kp1), np.array(good_out_kp2))
        print affine_matrix
        tmp_img = cv2.warpAffine(imgs[i + 1],
                                 affine_matrix,
                                 (total_width, total_height),
                                 borderMode=cv2.BORDER_TRANSPARENT)
        out[:height, :width, :] = imgs[0]
        if args.every == '1':
            cv2.imwrite(args.output + os.path.sep + i.__str__().zfill(3) + ".jpg", out)
    else:
        print "\nFrame " + (i + 1).__str__() + "/" + imgs.__len__().__str__()
        good_out_kp1, good_out_kp2, good_out = getKeyPoints(imgs[i + 1], tmp_img, args.feature)
        affine_matrix, mask = cv2.estimateAffine2D(np.array(good_out_kp1), np.array(good_out_kp2))
        print affine_matrix
        sum_offset_x = affine_matrix[0][2]
        sum_offset_y = affine_matrix[1][2]
        tmp_img = cv2.warpAffine(imgs[i + 1],
                                 affine_matrix,
                                 (total_width, total_height),
                                 borderMode=cv2.BORDER_TRANSPARENT)
        out = transparentOverlay(out, tmp_img)
        if args.every == '1':
            cv2.imwrite(args.output + os.path.sep + i.__str__().zfill(3) + ".jpg", out)

# 保存结果
if args.autoCut == '1':
    out = out[:height + int(sum_offset_y), :width + int(sum_offset_x), :]
elif args.autuCut == '0':
    out = out
cv2.imwrite(args.output + os.path.sep + "out.jpg", out)

t2 = time.time()
print "\ncost time:" + (t2 - t1).__str__() + " s"
