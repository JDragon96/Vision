# -*- coding: utf-8 -*-
import numpy as np
import cv2

def sigmoid_transform(a, c, LUT):
    """
    시그모이드 변환은 이미지 대비를 강조하기 위한 변환법이다.
    경계선을 더욱 뚜렷하게 만들어 주는 효과가 있다.
    """
    i=0
    for i in range(256):
        b = (i+1)/255
        tmp = 1/(1 + np.exp((-a)*(b - c)))
        LUT[i] = tmp
    return LUT

if __name__=="__main__":
    image = cv2.imread("./test_image/abdomen SUPINE sens-197 highdose.bmp", cv2.IMREAD_GRAYSCALE)
    dst = image.copy()

    #멱함수 변환 시 사용되는 1차원 배열
    a = 10
    c = 0.5
    LUT = np.zeros(256, np.float32)
    LUT = sigmoid_transform(8, 0.5, LUT)

    w = image.shape[1]
    h = image.shape[0]

    for i in range (0,h) :
        for j in range(0, w) :
            dst[i][j] = LUT[image[i][j]]*255     # dst 이미지 변환(멱함수 변환)

    image = cv2.resize(image, (256, 256), cv2.INTER_LANCZOS4)
    dst = cv2.resize(dst, (256, 256), cv2.INTER_LANCZOS4)

    cv2.imshow("raw image", image)
    cv2.imshow("transformed image", dst)
    cv2.waitKey(0)