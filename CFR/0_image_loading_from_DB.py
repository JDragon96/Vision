import cv2
import sys
import requests
import pymysql
from datetime import datetime
import os
import time
import json
import numpy as np

client_id = "client_id 정보"
client_secret = "client_secret 정보"
url = "얼굴인식 API URL 정보"  # // 얼굴감지
headers = {"client_id 정보": client_id, "client_secret 정보": client_secret}

conn = pymysql.connect(host='host 주소', user='user ID', password='user Password',
                       db='db 정보', charset='utf8')

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
j = 1

while (True):

    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()
    sql = "select image from `DB 정보`.Table 정보 where count = %s"
    curs.execute(sql, j)
    print(j)
    rows = curs.fetchone()
    conn.commit()
    if rows != None:
        images = rows[0]
        img = np.fromstring(images, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        cv2.imshow("database_image", img)
        key = cv2.waitKey(1)

        files = {'image': images}
        headers = {"client_id 정보": client_id, "client_secret 정보": client_secret}
        response = requests.post(url, files=files, headers=headers)
        rescode = response.status_code
        print(rescode)
        if (rescode == 200):
            print(type(response.text), response.text)
            print('success!')
            a = response.text
            print(a)
            print(type(a))
            api_data = json.loads(a)
            print(type(api_data))
            faceCounts = api_data['info']['faceCount']
            api_data_faces = api_data['faces']
            roi_data_list = []
            roi_data_list2 = []

            for i in range(0, faceCounts):
                api_data_faces_values = api_data_faces[i]
                keyList = api_data_faces_values.keys()
                for item in keyList:
                    if item == 'roi':
                        data = api_data_faces_values[item]
                        roi_data_list.append(data)

            for i in range(0, faceCounts):
                roi_x = roi_data_list[i]['x']
                roi_y = roi_data_list[i]['y']
                roi_width = roi_data_list[i]['width']
                roi_height = roi_data_list[i]['height']
                roi_data_list2.append([roi_x, roi_y, roi_width, roi_height])

            json_data_list = json.dumps(roi_data_list2)
            sql = "UPDATE `DB 정보`.Table 정보 SET faceCounts=%s,data_list=%s WHERE count =%s"
            curs.execute(sql, (faceCounts, json_data_list, j))

            print('data_transfer')
            conn.commit()
            j += 1
        else:
            print("Error Code:" + str(rescode))
            print(response.text)

        if key == 27:  # esc 키를 누르면 닫음
            break

conn.close()
cv2.destroyAllWindows()