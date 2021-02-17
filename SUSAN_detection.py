"""
SUSAN(Smallest Univalue Segmentation Assimilating Nucleus)

미분을 사용하지 않는 Edge Detection 기술이다. 또한 원형 Mask를 사용한다는 특이한 점이 있다.

딥러닝을 활용할 만한 HW가 구성되지 않았다면 유용한 segmentation 기법 중 하나이다.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def mask_generate(width=7):
    """
    SUSAN의 원형의 MASK를 사용하는 알고리즘이다.

    완벽한 원형 MASK가 아닌 뭉둑한 다이아몬드형 MASK를 생성하였다.
    """
    # 마스크 길이가 짝수일 때
    mask = np.zeros((width, width))
    f = int(width/4)
    t = int(width/2)
    j = 0
    if width % 2 == 0:
        for i in range(t):
            if j < f:
                mask[i, 0 + f - j : 3*f + j] = 1

            else:
                mask[i, : ] = 1
            j += 1

        for i in range(t):
            mask[t + i] = mask[t - i - 1]
        j=0

    # 마스크 길이가 홀수일 때
    center = int((width + 1)/2 - 1) # 길이가 7이면 [0,1,2,3,4,5,6] -> 3이 중심
    if width % 2 == 1:
        for i in range(center):
            mask[i, center - i - 1: center + i + 2] = 1

        mask[center, :] = 1

        for i in range(center):
            mask[width - center + i] = mask[center - i - 1]
    print(mask)
    return mask

def corner_detection(img, k):
    # 3. filtering
    for i in range(img.shape[0] - mask.shape[0] + 1):
        for j in range(img.shape[1] - mask.shape[0] + 1):
            ir = np.array(img[i: mask_width + i, j: mask_width + j])

            ir = ir[mask == 1]
            ir_nucleus = img[i + int((1 + mask_width) / 2),
                             j + int((1 + mask_width) / 2)]

            # SUSAN 계산
            # nucleus와 비슷한 픽셀 수 계산
            cm = np.sum(np.exp(-((ir - ir_nucleus) / k) ** 6))

            # geometric threshold값보다 유사 픽셀수가 적을 때
            # g값이 클수록 edge검출에 더 엄격하다.
            if cm <= g:
                cm = g - cm
            else:
                cm = 0
            output[i + int((1 + mask_width) / 2), j + int((1 + mask_width) / 2)] = cm

    return output


def gaussian(sigma, x, y):
    """
    해당 가우시안 함수는 2차 필터를 구성할 때 사용하는 공식이다.
    1차 필터는 y항을 제거한 후 x항만 사용하면 된다.
    """
    grad_xy = 1 / ((2 * np.pi) * pow(sigma, 4))
    grad_exp = math.exp((-(x ** 2 + y ** 2)) / (2 * (sigma ** 2)))

    hxy = grad_xy * grad_exp
    return hxy


def smooth(img, sigma):
    grad = np.zeros((3, 3))
    # 1. Gaussian kernel 생성
    for x in range(-1, 2, 1):
        for y in range(-1, 2, 1):
            grad[x+1, 1-y] = gaussian(sigma, x, y)

    smooth_img = cv2.filter2D(img,
                     -1,
                     grad)
    return smooth_img

def denoising_img(image):
    """
    np.median()은 입력 데이터의 평균을 계산한다.

    예를들어 image[0:3, 0:3]의 3x3을 때어내 [1,1]에 평균값을 집어넣어 smooth를 해준다.
    """
    output = image.copy()
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            output[i, j] = np.median([output[i-1][j],
                                      output[i+1][j],
                                      output[i][j-1],
                                      output[i][j+1],
                                      output[i-1][j-1],
                                      output[i+1][j+1],
                                      output[i+1][j-1],
                                      output[i-1][j+1]])
    return output

def norm(img):
    return ((img - np.min(img))/(np.max(img) - np.min(img)))

if __name__ == "__main__":
    # 0. 파라미터 설정
    g = 30  # g는 geometric threshold를 의미한다. 해당되는 픽셀 수를 비교하기 위함. filter 요소 개수보다 작아야함
    k = 12  # 밝기 차이 임계값

    # 1. 이미지 불러오기
    img = cv2.imread("susan_input1.png", cv2.IMREAD_GRAYSCALE)
    output = np.zeros(img.shape)
    print(img.shape)

    # 2. Mask 이미지 생성
    # 마스크 width는 홀수 사용 권장. 짝수는 nucleus 설정이 원활하지 못하다.
    # 필터 크기가 작을수록 노이즈에 취약하다.(g값을 키울 수 없기 때문)
    mask = mask_generate(7)
    mask_width = mask.shape[0]

    # corner detection
    # edge라고 감지된 부분은 0보다 큰 값을 갖는다.
    output = corner_detection(img, k)


    # 4. 최종 출력
    finaloutput = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    finaloutput[output != 0] = [255, 255, 0]
    plt.imshow(finaloutput)
    plt.show()
    

    # 5. 두 번째 이미지
    img = cv2.imread("susan_input2.png", 0)
    img_copy = img.copy()
    output = corner_detection(img, k)

    finaloutput2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    finaloutput2[output != 0] = [255, 255, 0]
    plt.imshow(finaloutput2)
    plt.show()


    # 6. 두 번째 이미지처럼 노이즈가 많은 경우에는 제거를 해야한다.
    im1 = denoising_img(img_copy)

    # Gaussian 의 sigma값이 1 이상으로 커지면, 커다란 object의 edge는 잘 확인
    # 하지만 1 이하의 값은 차이가 거의 없는 object또한 잘 구분을 해서 smooth해준다.
    im2 = smooth(im1, 0.7)

    # corner detect
    output = corner_detection(im2, k)
    print(im2.shape)
    finaloutput3 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    finaloutput3[output != 0] = [255, 255, 0]

    total = np.zeros((286, 512, 3))
    total[30:286, :256, :] = finaloutput2
    total[30:286, 256:, :] = finaloutput3
    cv2.putText(total, "Before smooth", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.putText(total, "After smooth", (266, 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.imshow("final image", total)
    cv2.waitKey(0)

