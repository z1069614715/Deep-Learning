from model_data.cat_body import cv_show

import numpy as np
import cv2

img = cv2.imread('decorate_png/cap3.png', cv2.IMREAD_UNCHANGED)
r, g, b, a = cv2.split(img)
# mask = a
# # mask = cv2.medianBlur(a, 5)
# # mask = cv2.medianBlur(mask, 5)

img = img.astype(np.uint8)
a = a.astype(np.uint8)

img_ = cv2.bitwise_and(img, img, mask=a)
# img_ = cv2.resize(img_, (300, 300))
cv_show(img_)