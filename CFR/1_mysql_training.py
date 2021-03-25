"""
원하는 시간대 정해지면 데이터 로딩해올 구간 설정하기
"""
import cv2
import numpy as np
import tensorflow as tf
import requests
import json
import pymysql
import base64

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def sql_loading(j):
    curs = conn.cursor()
    sql = "select image,faceCounts,data_list from `ncloud-mysql`.test5 where count = %s"
    curs.execute(sql, j)
    data = curs.fetchone()

    return data

def data_split(data):
    images = data[0]    # 이미지 bytes 데이터
    #images = base64.decodebytes(images)
    encoded_img = np.fromstring(images, dtype=np.uint8)  # bytes를 list로
    image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)  # list를 이미지화

    try:
        faceCounts = data[1]    # 얼굴 수
        roi_datalist = json.loads(data[2])  # 얼굴 윤곽선
    except:
        print("값이 제대로 입력되지 못했습니다.")
        faceCounts=1
        roi_datalist = [[1,1,1,1]]

    return image, faceCounts, roi_datalist

def image_cut(img, face_line, nface):
    images = []
    for i in range(nface):
        if (256 - face_line[i][0] < 128) and (256 - face_line[i][1] < 128):
            image = img[128:,
                    128:,
                    :]

        elif (256 - face_line[i][0] >= 128) and (256 - face_line[i][1] < 128):
            image = img[128:,
                    face_line[i][0]:face_line[i][0] + 128,
                    :]

        elif (256 - face_line[i][0] < 128) and (256 - face_line[i][1] >= 128):
            image = img[face_line[i][1]:face_line[i][1] + 128:,
                    128:,
                    :]

        else:
            image = img[face_line[i][1]:face_line[i][1] + 128,
                    face_line[i][0]:face_line[i][0] + 128,
                    :]

        images.append(image)
    return images


def sql_save(img,k):
    encode = cv2.imencode('.jpg', img)
    encode = encode[1].tostring()

    curs = conn.cursor()
    sql = "UPDATE `ncloud-mysql`.test5 " \
          "SET result_image=%s " \
          "WHERE count =%s"
    curs.execute(sql, (encode, k))
    conn.commit()


def camera(start_point = 0):
    k = 1
    while(True):
        start_point += 1
        while(True):
            data = sql_loading(start_point)  # MySQL에서 데이터 가져오기
            if data != None:
                break

        frame, face_count, face_line = data_split(data) # 데이터 분할
        #frame = cv2.fastNlMeansDenoisingColored(frame, None, 15, 15, 5, 10)
        frame = cv2.bilateralFilter(frame, 5, 75, 75)
        frame = cv2.resize(frame, (frame_width, frame_height))
        img = frame / 255.0 # img는 학습용으로 사용되는 데이터, frame은 출력용 데이터
        print(np.max(img))
        #print(f"frame shape : {np.shape(frame)}")

        train_images = image_cut(img, face_line, face_count)

        # 학습 진행하기
        save_check = 0
        pred_list = []

        for nface in range(face_count):
            image = train_images[nface]
            image = image[np.newaxis, :, :, :]
            pred = models.predict(image)*100
            result = np.argmax(pred)
            pred_list.append(result)
            #print(f"face number {nface} : {pred}")

            if result == 1:
                colors = (0, 255, 0)
            else:
                colors = (0, 0, 255)
                save_check = 1

            text = f"{round(pred[0][result], 2)}%" # 박스 위에 출력할 정확도

            cv2.rectangle(frame, (face_line[nface][0], face_line[nface][1]),
                          (face_line[nface][0] + face_line[nface][2], face_line[nface][1] + face_line[nface][3]),
                          colors, 1, cv2.LINE_AA)
            cv2.putText(frame, text, (face_line[nface][0], face_line[nface][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, colors)

            cv2.imshow("Python_Algorithm", image[0])

        cv2.imshow("Frame", frame)
        for i in range(len(pred_list)):
            if pred_list[i] == 1:
                save_check = 1
        
        
        if save_check == 1:
            print("저장완료 ", k)
            sql_save(frame, k)
        k += 1


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":


    # 기본 화상카메라 설정
    default_src = 0
    cap = cv2.VideoCapture(default_src)
    title_name = 'Recognition Video'
    frame_width = 256
    frame_height = 256

    # API 셋팅
    #client_id = ""
    #client_secret = ""
    url = "https://naveropenapi.apigw.ntruss.com/vision/v1/face"
    headers = {'X-NCP-APIGW-API-KEY-ID': client_id, 'X-NCP-APIGW-API-KEY': client_secret}
    
    # MySQL 기본세팅
    conn = pymysql.connect(host='',
                           user='kspp',
                           password='~',
                           db='ncloud-mysql',
                           charset='utf8')


    # 모델 로딩
    models = tf.keras.models.load_model('./save_model/128model3')
    models.load_weights("./save_cp/128model4")

    # 메인루프 실행
    camera(0)