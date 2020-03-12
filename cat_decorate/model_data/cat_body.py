import cv2

def cv_show(img):
    cv2.imshow('Pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_img(img, img_size=224, value=[0, 0, 0], inter=cv2.INTER_AREA):
    old_shape = img.shape[:2]
    ratio = img_size / max(old_shape)
    new_shape = [int(s * ratio) for s in old_shape[:2]]
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=inter)
    delta_h, delta_w = img_size - new_shape[0], img_size - new_shape[1]
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), borderType=cv2.BORDER_CONSTANT, value=value)
    return img, ratio, top, left

def label_cat_face(img, face_label):
    for i in face_label:
        for j in i:
            cv2.circle(img, (j[0], j[1]), radius=3, color=(0, 0, 255), thickness=-1)

    cv_show(img)