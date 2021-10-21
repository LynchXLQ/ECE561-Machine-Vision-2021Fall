import cv2
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import image_show, image_read, kp_descriptor, match_points, RANSAC
from tqdm import tqdm

warnings.filterwarnings("ignore", category=Warning)

# images_path = 'Images'
# for each_image in os.listdir(images_path):
#     if each_image != 'panorama.jpg':
#         img = image_read(os.path.join(images_path, each_image))
#         sift = cv2.SIFT_create()
#         kps, des = sift.detectAndCompute(img, None)
#=================================================================================================
# Create image list with key points and descriptors
# images_lst = kp_descriptor()   # [[img1,kps1,des1], [img2,kps2,des2]]
#
# Locate the key points and create a list of position
# kp_position_lst = []
# for i in range(np.size(images_lst,axis=0)):   # i=0,1
#     kp_position_lst_ = []
#     for point in images_lst[i][1]:    # each key point among all key points in image_i
#         kp_position_lst_.append([point.pt[0], point.pt[1]])   # point.pt=(x,y)
#         kp_position_lst.append(kp_position_lst_)
#     print('* Number of key points: ', len(kp_position_lst_))
        # print(points_lst)
# print(kp_position_lst)
#==================================================================================================
pts = []
images_lst = []
kp_position_lst = []

image_folder = 'Images'
for fm in os.listdir(image_folder):
    if fm != 'panorama.jpg':
        img = image_read(os.path.join(image_folder, fm))
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        im = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        images_lst.append([img, kp, des])
        kp_position_lst_ = ([[p.pt[0], p.pt[1]] for p in kp])
        kp_position_lst.append(kp_position_lst_)
        image_show(im)
#==================================================================================================
# Match descriptors and create match images list with des vectors
total_match_lst = []
for m in range(len(images_lst) - 1):   # first image to match, m = 0
    match_lst_img_des = []
    for n in range(m + 1, len(images_lst)):    # second image to match, n = 1
        match_des_vec = match_points(des1=images_lst[m][2], des2=images_lst[n][2])    # [[i1,j1], [i2,j2],...]
        num_match_des = len(match_des_vec)
        if 0.04 * np.size(images_lst[m][2], axis=0) < num_match_des < 0.95 * np.size(images_lst[m][2], axis=0):
            print('* Total matched des: ',num_match_des)
            print('* Match des/total des: ', num_match_des / np.size(images_lst[m][2], axis=0))
            match_lst_img_des.append([m,n,match_des_vec])   # [[img_m, img_n, [i,j]],[img_m, img_n, [i,j]],...]
    match_lst_img_des_ = match_lst_img_des
    total_match_lst.append(match_lst_img_des_)
# print(total_match_lst)
#==================================================================================================
# Get the coordinates of the matching pairs
total_pair_coordinate_lst = []
for p in range(np.size(total_match_lst,axis=0)):
    total_pair_coordinate_lst_ = []
    for q in range(np.size(total_match_lst[p],axis=0)):
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for r in range(np.size(total_match_lst[p][q][2],axis=0)):    # [i,j]
            x1.append(kp_position_lst[total_match_lst[p][q][0]][total_match_lst[p][q][2][r][0]][0])
            y1.append(kp_position_lst[total_match_lst[p][q][0]][total_match_lst[p][q][2][r][0]][1])
            x2.append(kp_position_lst[total_match_lst[p][q][1]][total_match_lst[p][q][2][r][1]][0])
            y2.append(kp_position_lst[total_match_lst[p][q][1]][total_match_lst[p][q][2][r][1]][1])
        total_pair_coordinate_lst_.append([x1, y1, x2, y2, total_match_lst[p][q][0], total_match_lst[p][q][1]])
    total_pair_coordinate_lst.append(total_pair_coordinate_lst_)
#==================================================================================================
# Calculating homography using ransac and storing right pairs of images and homography
proj_match = []
for t in range(len(total_pair_coordinate_lst)):
    pair_match = []
    for u in range(len(total_pair_coordinate_lst[t])):
        if np.sqrt(sum((np.asarray(total_pair_coordinate_lst[t][u][0])) ** 2)) - np.sqrt(
                sum((np.asarray(total_pair_coordinate_lst[t][u][2])) ** 2)) > np.sqrt(
                sum((np.asarray(total_pair_coordinate_lst[t][u][1])) ** 2)) - np.sqrt(sum((np.asarray(total_pair_coordinate_lst[t][u][3])) ** 2)):
            if np.sqrt(sum((np.asarray(total_pair_coordinate_lst[t][u][0])) ** 2)) < np.sqrt(sum((np.asarray(total_pair_coordinate_lst[t][u][2])) ** 2)):
                matches = RANSAC(total_pair_coordinate_lst[t][u][0], total_pair_coordinate_lst[t][u][1], total_pair_coordinate_lst[t][u][2], total_pair_coordinate_lst[t][u][3],
                                 total_pair_coordinate_lst[t][u][5], total_pair_coordinate_lst[t][u][4])
            elif np.sqrt(sum((np.asarray(total_pair_coordinate_lst[t][u][0])) ** 2)) > np.sqrt(
                    sum((np.asarray(total_pair_coordinate_lst[t][u][2])) ** 2)):
                matches = RANSAC(total_pair_coordinate_lst[t][u][2], total_pair_coordinate_lst[t][u][3], total_pair_coordinate_lst[t][u][0], total_pair_coordinate_lst[t][u][1],
                                 total_pair_coordinate_lst[t][u][4], total_pair_coordinate_lst[t][u][5])
        else:
            if np.sqrt(sum((np.asarray(total_pair_coordinate_lst[t][u][1])) ** 2)) < np.sqrt(sum((np.asarray(total_pair_coordinate_lst[t][u][3])) ** 2)):
                matches = RANSAC(total_pair_coordinate_lst[t][u][0], total_pair_coordinate_lst[t][u][1], total_pair_coordinate_lst[t][u][2], total_pair_coordinate_lst[t][u][3],
                                 total_pair_coordinate_lst[t][u][5], total_pair_coordinate_lst[t][u][4])
            elif np.sqrt(sum((np.asarray(total_pair_coordinate_lst[t][u][1])) ** 2)) > np.sqrt(
                    sum((np.asarray(total_pair_coordinate_lst[t][u][3])) ** 2)):
                matches = RANSAC(total_pair_coordinate_lst[t][u][2], total_pair_coordinate_lst[t][u][3], total_pair_coordinate_lst[t][u][0], img_coor[t][u][1],
                                 total_pair_coordinate_lst[t][u][4], total_pair_coordinate_lst[t][u][5])
        if len(matches) > 0:
            pair_match.append(matches)
    if len(pair_match) > 0:
        proj_match.append(pair_match)

    # Storing images and homography matrices in right sequence

print((proj_match))
list_of_order = []
list_of_homo = []
list_of_order.append(proj_match[0][0][1])
list_of_order.append(proj_match[0][0][2])
list_of_homo.append(np.reshape(proj_match[0][0][0], (3, 3)))
for i in range(len(proj_match)):
    for j in range(len(proj_match[i])):
        if list_of_order[-1] == proj_match[i][j][1]:
            list_of_order.append(proj_match[i][j][2])
            list_of_homo.append(np.reshape(proj_match[i][j][0], (3, 3)))
        elif list_of_order[0] == proj_match[i][j][2]:
            list_of_order.insert(0, proj_match[i][j][1])
            list_of_homo.insert(0, np.reshape(proj_match[i][j][0], (3, 3)))

print(list_of_order)
print(list_of_homo)

img_list = list_of_order.copy()
homo_list = list_of_homo.copy()
dst2 = images_lst[img_list[len(img_list) - 1]][0]

# plotting the images using matplotlib

for ims in range(len(homo_list) - 1, -1, -1):
    try:
        H_new = Homo_next
    except:
        H_new = homo_list[ims]
    w = np.dot(H_new, np.array([0, 0, 1]))
    wd = np.dot(H_new, np.array([[dst2.shape[1]], [dst2.shape[0]], [1]]))
    w1 = int(abs(w[0] / w[2] - wd[0] / wd[2]))
    w2 = int(abs((w[1] / w[2]) - (wd[1] / wd[2])))
    dst2 = cv2.warpPerspective(dst2, H_new, (images_lst[img_list[ims]][0].shape[1] + dst2.shape[1], dst2.shape[0]))
    # dst2 = cv2.warpPerspective(dst2, H_new, (images[img_list[ims]][0].shape[1]  + w1 , max(w2,images[img_list[ims]][0].shape[0]))
    dst2[0:images_lst[img_list[ims]][0].shape[0], 0:images_lst[img_list[ims]][0].shape[1]] = images_lst[img_list[ims]][0]
    w = np.dot(H_new, np.array([0, 0, 1]))
    wd = np.dot(H_new, np.array([[dst2.shape[1]], [dst2.shape[0]], [1]]))
    if ims == 500:
        images_1 = [images_lst[img_list[ims - 1]][0], dst2]
        images_2 = []
        for ii in range(2):
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(images_1[ii], None)
            # im = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            images_2.append([ii, kp, des])
        pts_1 = ([[p.pt[0], p.pt[1]] for p in images_2[0][1]])
        pts_2 = ([[p.pt[0], p.pt[1]] for p in images_2[1][1]])
        matches = match_points(images_2[0][2], images_2[1][2])
        cols1 = []
        rows1 = []
        cols2 = []
        rows2 = []
        print(matches)
        for s in range(len(matches)):
            cols1.append(pts_1[matches[s][0]][0])
            rows1.append(pts_1[matches[s][0]][1])
            cols2.append(pts_2[matches[s][1]][0])
            rows2.append(pts_2[matches[s][1]][1])
        matches = RANSAC(cols2, rows2, cols1, rows1, 0, 1)
        Homo_next = np.reshape(matches[0], (3, 3))
    plt.figure(figsize=(16, 14))
    plt.title('Warped Image')
    plt.imshow(dst2)
    plt.savefig('./Images_panorama/panorama.jpg')
    plt.show()
# print(type(dst))
img = dst2

# img = img[0:int(abs(wd1)-abs(w1)), 0:int(abs(wd2)-abs(w2))]

img = np.asarray(img, dtype=np.uint8)
cv2.imwrite('Images_panorama/panorama.jpg', img)
cv2.waitKey(0)






