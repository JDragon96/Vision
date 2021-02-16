import cv2
import numpy as np

#멱함수 변환 함수
def power_transform(gamma):
    """
    멱함수 리스트 생성
    :param gamma: exp의 근을 의미한다.
    :param LUT: Look Up Table로 (i/255)^gamma의 값을 저장하는 list이다.
    """
    LUT = np.zeros(256, np.uint8)
    for i in range(0, 256):
        tmp=255.0*pow((i/255.0), gamma)
        if tmp>255 :
            tmp=255
        LUT[i]=tmp
    return LUT


if __name__=="__main__":
    image = cv2.imread("./test_image/chest.bmp", cv2.IMREAD_GRAYSCALE)
    h = image.shape[0]
    w = image.shape[1]
    dst = image.copy()

    # gamma 값(변동 가능)
    # 값이 0에 가까울 수록 이미지 대비가 커진다.
    gamma=0.5

    #멱함수 변환 시 사용되는 1차원 배열
    LUT = power_transform(gamma)

    for i in range (0, h):
        for j in range(0, w):
            dst[i][j] = LUT[image[i][j]]     # dst 이미지 변환(멱함수 변환)

    cv2.imshow("원본 이미지", image)
    cv2.imshow("멱함수 변환", dst)
    cv2.waitKey(0)
