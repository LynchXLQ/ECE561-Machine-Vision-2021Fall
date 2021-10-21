import random
import numpy as np
import cv2
import os

from tqdm import tqdm


def image_read(path, grayscale = False, show = False):
    if grayscale:
        raw_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(raw_image, dsize=None, fx=0.2, fy=0.2)
    else:
        raw_image = cv2.imread(path)
        img = cv2.resize(raw_image, dsize=None, fx=0.2, fy=0.2)
    if show:
        image_show(img)
    return img



def image_show(image, delay = 1000):
    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Image',image)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def kp_descriptor():
    images_lst = []
    kp_position_lst = []
    image_folder = 'Images'
    my_images = os.listdir(image_folder)
    print('* Image name list: ',my_images)
    for image in my_images:
        raw_image = cv2.imread(f'{image_folder}/{image}', flags=0)
        current_image = cv2.resize(raw_image, None, fx=0.2, fy=0.2)
        # cv2.imshow('image',currentImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        sift = cv2.SIFT_create()
        all_keypoints, all_descriptors = sift.detectAndCompute(current_image, None)
        image_kp = cv2.drawKeypoints(current_image, all_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        images_lst.append([current_image, all_keypoints, all_descriptors])
        cv2.imshow('image_kp', image_kp)
        cv2.imwrite('Images_keypoints/Keypoints_'+image,image_kp)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return images_lst



def match_points(des1, des2):
    '''
    Matches corresponding points
    :param des1: descriptors in image1
    :param des2: descriptors in image2
    :return:
    '''
    match_des_vec = []
    for i in tqdm(range(np.size(des1,axis=0)), desc='Match des and create matching des list'):   # Row
        dist = np.sqrt(np.sum((np.subtract(des1[i], des2[0]))**2))
        dist_best = dist
        dist_sec = dist
        dist_best_vec = [i,0]
        dist_sec_vec = [i,0]
        for j in range(1, np.size(des2,axis=0)):
            dist = np.sqrt(np.sum((np.subtract(des1[i], des2[j])) ** 2))
            if dist < dist_best:
                dist_sec = dist_best
                dist_best=dist
                dist_best_vec = [i, j]    # [ith des of img1, jth des of img2]
            elif dist < dist_sec:
                dist_sec = dist
                dist_sec_vec=[i, j]
        if dist_best/dist_sec <= 0.70:
            match_des_vec.append(dist_best_vec)
    return match_des_vec

# a = [[1,1,1],
#      [2,2,2],
#      [3,3,3]]
#
# b = [[1,2,3],
#      [4,5,6],
#      [7,8,9]]
# matches = match_points(a,b)


def RANSAC(x_i, y_i, x_j, y_j, img1, img2):
    """
    removes outliers using homography equation.
    """
    Homo = []
    count_max = 0
    # print(len(x_j))
    # print(x_i[0])
    for _ in tqdm(range(600),desc='RANSAC'):
        i = random.sample(range(len(x_i)), 8)   # randomly pick 8 x_i from all x_i, len(x_i)=len(x_j)=292
        A = np.array([[x_i[i[0]], y_i[i[0]], 1, 0, 0, 0, -x_j[i[0]] * x_i[i[0]], -x_j[i[0]] * y_i[i[0]], -x_j[i[0]]]])
        A = np.append(A, [[0, 0, 0, x_i[i[0]], y_i[i[0]], 1, -y_j[i[0]] * x_i[i[0]], -y_j[i[0]] * y_i[i[0]], -y_j[i[0]]]], axis=0)
        for pt in range(1, 8):   # pt = 1,2,3,4,5,6,7
            A = np.append(A, [[x_i[i[pt]], y_i[i[pt]], 1, 0, 0, 0, -x_j[i[pt]] * x_i[i[pt]], -x_j[i[pt]] * y_i[i[pt]], -x_j[i[pt]]]], axis=0)
            A = np.append(A, [[0, 0, 0, x_i[i[pt]], y_i[i[pt]], 1, -y_j[i[pt]] * x_i[i[pt]], -y_j[i[pt]] * y_i[i[pt]], -y_j[i[pt]]]], axis=0)
        u, s, vh = np.linalg.svd(A, full_matrices=True)   # A:m*n u:m*m s:m*n v:n*n
        H = vh[-1]
        count = 0
        refined_matches = []
        search_in = np.setdiff1d([*range(len(x_i))], (i))
        for j in search_in:
            val = np.dot([[x_i[j], y_i[j], 1, 0, 0, 0, -x_j[j] * x_i[j], -x_j[j] * y_i[j], -x_j[j]], [0, 0, 0, x_i[j], y_i[j], 1, -y_j[j] * x_i[j], -y_j[j] * y_i[j], -y_j[j]]], H)
            val = np.sqrt(np.dot(np.transpose(val), val))
            if val <= 0.0095:
                count += 1
                refined_matches.append([x_i[j], y_i[j], x_j[j], y_j[j]])

        if count > count_max:
            count_max = count
            final_matches = refined_matches
            Homo = [H, img1, img2]
    return (Homo)













