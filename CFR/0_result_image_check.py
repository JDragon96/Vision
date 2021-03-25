import cv2
import pymysql
from datetime import datetime
import time
import json
import numpy as np

conn = pymysql.connect(host='host 주소', user='user ID', password='user Password',
                       db='db 정보', charset='utf8')

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("시작시간 : ", now)
j = 0

while (True):

    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()
    sql = "select result_image from `DB 정보`.Table 정보 where count = %s"
    curs.execute(sql, j)
    print(j)

    now_save = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    rows = curs.fetchone()
    conn.commit()

    if rows[0] != None:
        images = rows[0]
        images = np.fromstring(images, dtype=np.uint8)
        images = cv2.imdecode(images, cv2.IMREAD_COLOR)
        cv2.imshow("database_image", images)
        key = cv2.waitKey(100)

        if key == 27:  # esc 키를 누르면 닫음
            break

conn.close()
cv2.destroyAllWindows()