import cv2
import pymysql
from datetime import datetime
import os
import time
import numpy as np

conn = pymysql.connect(host='host 주소', user='user ID', password='user Password',
                       db='db 정보', charset='utf8')

cap = cv2.VideoCapture(0)
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("시작시간 : ", now)

j = 0

while (True):
    ret, cam = cap.read()

    if (ret):
        j += 1
        cv2.imshow('camera', cam)
        key = cv2.waitKey(1)
        cam = cv2.resize(cam, (256, 256))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        encoded_string = cv2.imencode('.jpg', cam)
        encode = encoded_string[1].tostring()

        curs = conn.cursor()
        sql = """insert into `DB 정보`.Table 정보(timer,image) values (%s,%s)"""
        curs.execute(sql, (now, encode))
        conn.commit()

        if key == 27:  # esc 키를 누르면 닫음
            break

cap.release()
conn.close()
cv2.destroyAllWindows()