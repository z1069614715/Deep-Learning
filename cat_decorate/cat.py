import sys, time, os
from math import degrees, atan

import numpy as np

import cv2
from PIL import Image

from keras.models import load_model
from yolo import YOLO
from model_data.cat_body import resize_img, cv_show, label_cat_face

glasses_1 = cv2.imread('decorate_png/glasses.png', cv2.IMREAD_UNCHANGED)
cap_1 = cv2.imread('decorate_png/cap.png', cv2.IMREAD_UNCHANGED)
cap_2 = cv2.imread('decorate_png/cap2.png', cv2.IMREAD_UNCHANGED)
cap_3 = cv2.imread('decorate_png/cap3.png', cv2.IMREAD_UNCHANGED)
cap_4 = cv2.imread('decorate_png/cap4.png', cv2.IMREAD_UNCHANGED)


class cat():
    face_res = []

    # init yolo model and NasNet model
    y = YOLO()
    NasNet = load_model('model_data/NasNet.h5')

    def __init__(self, file):
        self.cat_img = cv2.imdecode(np.fromfile(file, np.uint8), cv2.IMREAD_COLOR)

        yolo_img = Image.open(file)
        yolo_res = self.y.detect_image(yolo_img)

        for i in yolo_res:
            left_, top_, right_, bottom_ = i
            cat_img = self.cat_img[top_:bottom_, left_:right_]
            img_resize, ratio, top, left = resize_img(cat_img)
            img_train = img_resize.astype('float32').reshape((-1, 224, 224, 3))
            img_train /= 255.0
            img_out = (self.NasNet.predict(img_train)[0]).astype('float32').reshape((-1, 2))

            self.face_res.append(((img_out - np.array([left, top])) / ratio + np.array([left_, top_])).astype(np.int))

    def glasses_decorate(self, num=1):
        if num == 1:
            glasses = glasses_1

        for face in self.face_res:
            eye_left, eye_right = face[0], face[1]
            glasses_center = np.mean([eye_left, eye_right], axis=0)
            glasses_size = np.linalg.norm(eye_left - eye_right) * 2

            glasses_rotate = degrees(atan(-(eye_left[1] - eye_right[1]) / (eye_left[0] - eye_right[0])))
            M = cv2.getRotationMatrix2D((glasses.shape[0] / 2, glasses.shape[1] / 2), glasses_rotate, 1)
            rotate_glasses = cv2.warpAffine(glasses, M, (glasses.shape[0], glasses.shape[1]))

            glasses_newsize = (int(glasses_size), int(glasses.shape[0] * glasses_size / glasses.shape[1]))
            new_glasses = cv2.resize(rotate_glasses, glasses_newsize, interpolation=cv2.INTER_AREA)

            new_img = self.cat_img.copy()
            if new_img.shape[2] == 3:
                new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2BGRA)

            b, g, r, a = cv2.split(new_glasses)
            mask = cv2.medianBlur(a, 3)

            x, y = glasses_center
            h, w, _ = new_glasses.shape
            if int(y - h / 2) < 0 or int(y + h / 2) > self.cat_img.shape[0] or int(x - w / 2) < 0 or int(x + w / 2) > \
                    self.cat_img.shape[1]:
                return
            clip_img = new_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

            img_1 = cv2.bitwise_and(clip_img, clip_img, mask=cv2.bitwise_not(mask))
            img_2 = cv2.bitwise_and(new_glasses, new_glasses, mask=mask)

            new_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img_1, img_2)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGRA2BGR)
            self.cat_img = new_img

    def cap_decorate(self, num=1):
        if num == 1:
            cap = cap_1
        elif num == 2:
            cap = cap_2
        elif num == 3:
            cap = cap_3

        for face in self.face_res:
            left_ear_right, right_ear_left = face[5], face[6]
            cap_size = np.linalg.norm(left_ear_right - right_ear_left)
            cap_center = np.mean([left_ear_right, right_ear_left], axis=0)

            cap_rotate = degrees(
                atan(-(left_ear_right[1] - right_ear_left[1]) / (left_ear_right[0] - right_ear_left[0])))
            M = cv2.getRotationMatrix2D((cap.shape[0] / 2, cap.shape[1] / 2), cap_rotate, 1)
            rotated_cap = cv2.warpAffine(cap, M, (cap.shape[0], cap.shape[1]))

            cap_newsize = (int(cap_size), int(cap.shape[0] * cap_size / cap.shape[1]))
            rotated_cap = cv2.resize(rotated_cap, cap_newsize)

            new_img = self.cat_img.copy()
            if new_img.shape[2] == 3:
                new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2BGRA)

            b, g, r, a = cv2.split(rotated_cap)
            mask = cv2.medianBlur(a, 5)

            x, y = cap_center
            h, w, _ = rotated_cap.shape
            if int(y - h / 2) < 0 or int(y + h / 2) > self.cat_img.shape[0] or int(x - w / 2) < 0 or int(x + w / 2) > \
                    self.cat_img.shape[1]:
                return
            temp = new_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
            img1 = cv2.bitwise_and(temp, temp, mask=cv2.bitwise_not(mask))
            img2 = cv2.bitwise_and(rotated_cap, rotated_cap, mask=mask)
            new_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1, img2)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGRA2BGR)

            self.cat_img = new_img

    def cap_decorate_test(self, num=1):
        if num == 1:
            cap = cap_1
        elif num == 2:
            cap = cap_2
        elif num == 3:
            cap = cap_3
        elif num == 4:
            cap = cap_4

        for face in self.face_res:
            left_ear_right, right_ear_left = face[5], face[6]
            left_ear_center, right_ear_center = face[4], face[7]
            cap_size = np.linalg.norm(left_ear_right - right_ear_left)
            cap_center = [(left_ear_right[0] + right_ear_left[0]) / 2,
                          ((left_ear_center[1] + right_ear_center[1]) / 2 + (left_ear_right[1] + right_ear_left[1]) / 2) / 2]
            # cv2.circle(self.cat_img, (int(cap_center[0]), int(cap_center[1])), radius=5, color=[0, 0, 255], thickness=-1)

            cap_rotate = degrees(
                atan(-(left_ear_right[1] - right_ear_left[1]) / (left_ear_right[0] - right_ear_left[0])))
            M = cv2.getRotationMatrix2D((cap.shape[0] / 2, cap.shape[1] / 2), cap_rotate, 1)
            rotated_cap = cv2.warpAffine(cap, M, (cap.shape[0], cap.shape[1]))

            cap_newsize = (int(cap_size), int(cap.shape[0] * cap_size / cap.shape[1]))
            rotated_cap = cv2.resize(rotated_cap, cap_newsize)

            new_img = self.cat_img.copy()
            if new_img.shape[2] == 3:
                new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2BGRA)

            b, g, r, a = cv2.split(rotated_cap)
            mask = cv2.medianBlur(a, 5)

            x, y = cap_center
            h, w, _ = rotated_cap.shape

            x_min = (int(x - w / 2) if int(x - w / 2) >= 0 else 0)
            y_min = (int(y - h / 2) if int(y - h / 2) >= 0 else 0)
            x_max = (int(x + w / 2) if int(x + w / 2) <= self.cat_img.shape[1] else self.cat_img.shape[1])
            y_max = (int(y + h / 2) if int(y + h / 2) <= self.cat_img.shape[0] else self.cat_img.shape[0])

            left_change = abs(int(x - w / 2) - x_min)
            right_change = abs(int(x + w / 2) - x_max)
            bottom_change = abs(int(y + h / 2) - y_max)
            top_change = abs(int(y - h / 2) - y_min)

            temp = new_img[y_min:y_max, x_min:x_max]
            crop_rotated_cap = rotated_cap[(top_change + 1 if top_change > 0 else 0):rotated_cap.shape[1] - bottom_change,
                          (left_change + 1 if left_change > 0 else 0):rotated_cap.shape[0] - right_change]
            mask = mask[(top_change + 1 if top_change > 0 else 0):rotated_cap.shape[1] - bottom_change,
                          (left_change + 1 if left_change > 0 else 0):rotated_cap.shape[0] - right_change]
            img1 = cv2.bitwise_and(temp, temp, mask=cv2.bitwise_not(mask))
            img2 = cv2.bitwise_and(crop_rotated_cap, crop_rotated_cap, mask=mask)
            new_img[y_min:y_max, x_min:x_max] = cv2.add(img1, img2)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGRA2BGR)

            self.cat_img = new_img


cat_ = cat('cat_image/2.jpg')
# cat_.cap_decorate_test()
cat_.cap_decorate_test(num=1)
cv_show(cat_.cat_img)
