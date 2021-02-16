"""
GaussianBlur : 이미지를 흐리게 만들어주는 Smoothing 기법이다.

Gaussian Blur를 이용한 이미지 경계선 강조기법
"""
import numpy as np
import cv2

def norm(img):
    if np.min(img) < 0:
        img = img - np.min(img)
    return ((img - np.min(img))/(np.max(img) - np.min(img)))


if __name__=="__main__":
    # 원본 이미지 출력
    image = cv2.imread('./test_image/hand.bmp', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("raw image", image)
    cv2.waitKey(0)

    # GaussianBlur를 이용한 가우시안 필터 적용
    f_image = np.float32(image)
    smooth1 = cv2.GaussianBlur(src=f_image,
                               ksize=(5, 5),
                               sigmaX=0,
                               dst=0)
    smooth2 = cv2.GaussianBlur(f_image, (11, 11), 0, 0)
    smooth3 = cv2.GaussianBlur(f_image, (17, 17), 0, 0)
    smooth4 = cv2.GaussianBlur(f_image, (23, 23), 0, 0)
    smooth5 = cv2.GaussianBlur(f_image, (35, 35), 0, 0)

    # 원본 - smooth 이미지 = object 경계선
    diff1 = f_image - smooth1
    diff2 = f_image - smooth2
    diff3 = f_image - smooth3
    diff4 = f_image - smooth4
    diff5 = f_image - smooth5
    print(np.max(diff1), np.min(diff1))

    # 원본 이미지 + smooth 이미지 => 경계선 강조!
    sharp = f_image + 3 * diff1 + 3 * diff2 + 3 * diff3 + 3 * diff4 + 3 * diff5
    sharp = norm(sharp)*255

    # im배열에 이미지 저장
    im = np.zeros((600, 1500))
    im[:300, :300] = cv2.resize(diff1, (300, 300))
    im[:300, 300:600] = cv2.resize(diff2, (300, 300))
    im[:300, 600:900] = cv2.resize(diff3, (300, 300))
    im[:300, 900:1200] = cv2.resize(diff4, (300, 300))
    im[:300, 1200:1500] = cv2.resize(diff5, (300, 300))
    im[300:, :300] = cv2.resize(sharp, (300, 300))
    im[300:, 300:600] = cv2.resize(f_image, (300, 300))


    # 이미지 텍스트 설정
    b,g,r = 255,255,255
    cv2.putText(im,  "5x5", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv2.LINE_AA)
    cv2.putText(im,  "11x11", (300,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv2.LINE_AA)
    cv2.putText(im,  "17x17", (600,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv2.LINE_AA)
    cv2.putText(im,  "23x23", (900,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv2.LINE_AA)
    cv2.putText(im,  "35x35", (1200,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv2.LINE_AA)
    cv2.putText(im, "sharp image", (0, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1, cv2.LINE_AA)
    cv2.putText(im, "raw image", (300, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1, cv2.LINE_AA)
    cv2.imshow("image", norm(im))
    cv2.waitKey(0)