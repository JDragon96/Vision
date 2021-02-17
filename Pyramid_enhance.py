import cv2
import numpy as np

def norm(img):
  if np.min(img) < 0:
    img = img - np.min(img)
  return ((img - np.min(img))/(np.max(img) - np.min(img)))

def clip(img):
  return np.clip(img, 0, 255)

if __name__=="__main__":
    image = cv2.imread("./test_image/abdomen.bmp", cv2.IMREAD_GRAYSCALE)

    # Gaussian Pyramid
    kernel = (5, 5)

    G0 = np.float32(image)  # 400
    G1 = cv2.GaussianBlur(src=G0,
                          ksize=kernel,
                          sigmaX=1.4)
    G1 = cv2.pyrDown(G1)    # 200

    G2 = cv2.GaussianBlur(src=G1,
                          ksize=kernel,
                          sigmaX=1.4)
    G2 = cv2.pyrDown(G2)    # 100

    G3 = cv2.GaussianBlur(src=G2,
                          ksize=kernel,
                          sigmaX=1.4)
    G3 = cv2.pyrDown(G3)    # 50
    print(np.shape(G3))

    # Laplacian Pyramid
    L0 = G0 - cv2.GaussianBlur(src=cv2.pyrUp(G1),
                               ksize=kernel,
                               sigmaX=1.4)

    L1 = G1 - cv2.GaussianBlur(src=cv2.pyrUp(G2),
                               ksize=kernel,
                               sigmaX=1.4)

    L2 = G2 - cv2.GaussianBlur(src=cv2.pyrUp(G3),
                               ksize=kernel,
                               sigmaX=1.4)

    # Result
    a = 3
    R3 = G3
    R2 = L2 * a + cv2.GaussianBlur(src=cv2.pyrUp(R3),
                                   ksize=kernel,
                                   sigmaX=1.4)

    R1 = L1 * a + cv2.GaussianBlur(src=cv2.pyrUp(R2),
                                   ksize=kernel,
                                   sigmaX=1.4)

    R0 = L0 * a + cv2.GaussianBlur(src=cv2.pyrUp(R1),
                                   ksize=kernel,
                                   sigmaX=1.4)

    cv2.imshow("Laplacian", L0 + 128)
    cv2.imshow("Gaussian", G1)
    cv2.imshow("Result", clip(R1))
    print(np.max(R0))


    # 이미지 한변에 출력
    im = np.empty((750, 1200))

    im[:400, :400] = G0
    im[:400, 400:800] = L0 + 128
    im[:400, 800:1200] = R0

    im[400:600, :200] = G1
    im[400:600, 400:600] = L1 + 128
    im[400:600, 800:1000] = R1

    im[600:700, :100] = G2
    im[600:700, 400:500] = L2 + 128
    im[600:700, 800:900] = R2

    im[700:750, :50] = G3
    im[700:750, 800:850] = R3

    cv2.putText(im, "Gaussian Pyramid", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                thickness=1)
    cv2.putText(im, "Laplacian Pyramid", (410, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                thickness=1)
    cv2.putText(im, "enhance image", (810, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                thickness=1)

    cv2.imshow("final image", im)