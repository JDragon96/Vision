import cv2
import numpy as np
import os
from tensorflow.keras.utils import to_categorical

img_size = 128

def labeling(mask_len, nmask_len):
    mask_label = np.ones((mask_len, 1))
    nmask_label = np.zeros((nmask_len, 1))
    total_label = np.concatenate([mask_label, nmask_label], axis=0)
    total_label = to_categorical(total_label, 2)

    return total_label

def img_loading(img_files):
    """
    :param img_files: 튜플 형식으로 (mask_path, non_mask_path)를 입력한다.
    :return: mask 이미지, non mask 이미지
    """
    mask_path, nmask_path = img_files

    # 파일 목록
    mask_list = os.listdir(mask_path)
    nmask_list = os.listdir(nmask_path)

    # 파일 길이
    mask_len = len(mask_list)
    nmask_len = len(nmask_list)
    print(f"mask length : {mask_len}")
    print(f"nmask length : {nmask_len}")

    # 이미지 로딩
    mask_img = []
    nmask_img = []
    for name in mask_list:
        img = cv2.imread(os.path.join(mask_path, name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_size, img_size), cv2.INTER_LANCZOS4).astype('float32') / 255.0
        mask_img.append(img)

    for name in nmask_list:
        img = cv2.imread(os.path.join(nmask_path, name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_size, img_size), cv2.INTER_LANCZOS4).astype('float32') / 255.0
        nmask_img.append(img)

    print(f"mask img shape : {np.shape(mask_img)}")
    print(f"nmask img shape : {np.shape(nmask_img)}")

    total_image = np.concatenate([mask_img, nmask_img], axis=0)
    print(f"Total image shape : {np.shape(total_image)}")

    total_label = labeling(mask_len, nmask_len)
    print(f"Total label shape : {np.shape(total_label)}")

    return total_image, total_label


def test_img_loading(test_img_files):
    test_mask_path, test_nmask_path = test_img_files

    # 파일 목록
    mask_list = os.listdir(test_mask_path)
    nmask_list = os.listdir(test_nmask_path)

    # 파일 길이
    mask_len = len(mask_list)
    nmask_len = len(nmask_list)
    print(f"mask length : {mask_len}")
    print(f"nmask length : {nmask_len}")

    # 이미지 로딩
    mask_img = []
    nmask_img = []
    for name in mask_list:
        img = cv2.imread(os.path.join(test_mask_path, name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_size, img_size), cv2.INTER_LANCZOS4).astype('float32') / 255.0
        mask_img.append(img)

    for name in nmask_list:
        img = cv2.imread(os.path.join(test_nmask_path, name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_size, img_size), cv2.INTER_LANCZOS4).astype('float32') / 255.0
        nmask_img.append(img)

    print(f"mask img shape : {np.shape(mask_img)}")
    print(f"nmask img shape : {np.shape(nmask_img)}")

    total_image = np.concatenate([mask_img, nmask_img], axis=0)
    print(f"Total image shape : {np.shape(total_image)}")

    total_label = labeling(mask_len, nmask_len)
    print(f"Total label shape : {np.shape(total_label)}")

    return total_image, total_label