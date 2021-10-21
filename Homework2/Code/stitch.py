import cv2
import os

image_folder = 'Images'
my_images = os.listdir(image_folder)
print(my_images)

sift = cv2.SIFT_create()

# Image 1
raw_image1 = cv2.imread(f'{image_folder}/{my_images[0]}', flags=0)
current_image1 = cv2.resize(raw_image1, None, fx=0.2, fy=0.2)
kp1, des1 = sift.detectAndCompute(current_image1, None)

# Image 2
raw_image2 = cv2.imread(f'{image_folder}/{my_images[1]}', flags=0)
current_image2 = cv2.resize(raw_image2, None, fx=0.2, fy=0.2)
kp2, des2 = sift.detectAndCompute(current_image2, None)

images = []
images.extend([raw_image1, raw_image2])
print(images)

sticher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
ret, pano = sticher.stitch(images)

cv2.imshow('Panorama', pano)
cv2.waitKey(0)