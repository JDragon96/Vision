"""
Gabor 필터는 외곽선을 검출할 때 사용하는 필터

sigma : Kernel의 너비를 결정하는 요소. 값이 클수록 필터가 잘게 나뉜다.
theta : kernel의 방향성 결정. 검출하고 싶은 edge의 각도에 맞춰 필터를 설정하면 된다.
lambda : kernel의 sin함수 조절.
gamma : kernel의 가로, 세로 비율을 조절. 1 이하로 갈수록 타원형으로 바뀐다.
psi : 필터의 대칭이동 정도를 나타냄. 2/pi일 때 중심을 기준으로 좌우 대칭을 이룬다.
      3.14(pi)일 때 중앙이 최소값을 갖는다.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

def norm(img):
    """
    데이터 표준화 함수
    """
    if np.min(img) < 0:
        img = img + (-np.min(img))

    return ((img - np.min(img))/(np.max(img) - np.min(img)))

if __name__=="__main__":
    # Gabor 필터 생성
    kernel1 = cv2.getGaborKernel(ksize=(11, 11),
                                 sigma=3.12,
                                 theta=3.14/4,
                                 lambd=8,
                                 gamma=0.85,
                                 psi=3.14,
                                 ktype=cv2.CV_32F)
    kernel2 = cv2.getGaborKernel((11, 11), 3.12, 0, 8, 0.85, 3.14, ktype=cv2.CV_32F)
    kernel3 = cv2.getGaborKernel((11, 11), 3.12, -3.14/4, 8, 0.85, 3.14, ktype=cv2.CV_32F)
    print(f"gober shape : {kernel1.shape}")
    print(np.max(kernel1), np.min(kernel1))
    
    # 필터 복사
    kernel_o1 = kernel1
    kernel_o2 = kernel2
    kernel_o3 = kernel3

    # 필터 cv2 출력
    kernel_o1 = cv2.resize(kernel_o1, (300, 300), interpolation = cv2.INTER_NEAREST)
    kernel_o2 = cv2.resize(kernel_o2, (300, 300), interpolation = cv2.INTER_NEAREST)
    kernel_o3 = cv2.resize(kernel_o3, (300, 300), interpolation = cv2.INTER_NEAREST)
    
    filter = np.zeros((300, 900))
    filter[:, :300] = norm(kernel_o1)
    filter[:, 300:600] = norm(kernel_o2)
    filter[:, 600:] = norm(kernel_o3)
    cv2.imshow("filter image", filter)
    cv2.waitKey(0)
    
    # 이미지 로딩
    path = "./test_image/chest.bmp"
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("raw image", image)
    cv2.waitKey(0)

    # 이미지에 필터 적용
    dst1 = cv2.filter2D(image, cv2.CV_32FC1, kernel1)
    dst2 = cv2.filter2D(image, cv2.CV_32FC1, kernel2)
    dst3 = cv2.filter2D(image, cv2.CV_32FC1, kernel3)

    u_img1 = norm(dst1)
    u_img2 = norm(dst2)
    u_img3 = norm(dst3)

    u_img1 = cv2.resize(u_img1, (300, 300))
    u_img2 = cv2.resize(u_img2, (300, 300))
    u_img3 = cv2.resize(u_img3, (300, 300))

    b = np.zeros((300, 900))

    b[:, :300] = u_img1
    b[:, 300:600] = u_img2
    b[:, 600:] = u_img3

    cv2.imshow("tranformed image", b)
    cv2.waitKey(0)