import matplotlib.pyplot as plt
import numpy as np
from keras.backend import clear_session
from keras.models import model_from_json
import pywt
from myo_python_master import myo
from threading import Lock,Thread
from collections import deque
import time
import serial
import numpy as np

pre_pred = 10
test_range = 100  # 케라스로 한번에 테스트할 데이터 크기
test_ch = 8     # 데이터 채널 수
time_part = 1   # CRNN 구조 중 RNN(LSTM) 부분에서 전체데이터를 몇파트로 나눠서 테스트 할지
widths=200      # SAWT 중 CWT 과정에 쓰일 width
result = np.zeros((5,5))
model_folder='./model_save/'  # 케라스 모델 저장되어있는 폴더
model_file_name = "97"  # 저장된 케라스 모델 파일 이름
process_mode = False   # 전처리 과정을 진행 모드 (True / False)w
use_robot = True

motion_set= ["0 cylinder","1 hook","2 pen","3 rock","4 release"]

# ["0 card","1 fist","2 good","3 ok","4 pinch","5 relax","6 v"]
# ["0 card","1 fist","2 good","3 ok","4 pinch","5 v"]
# ["0 fist","1 good","2 ok","3 pinch","4 relax","5 v"]
# ["0 card","1 good","2 ok","3 pinch","4 relax","5 v"]
# ["0 good","1 ok","2 pinch","3 relax","4 v"]
motion_num = 5
each_motion_time = 1000    # 한 동작당 몇번씩 테스트
full_test_time=motion_num*each_motion_time
correct_label=[]
for i in range(motion_num):
    for j in range(each_motion_time):
        correct_label.append(i)
print(correct_label)
pred_label=[]


########################################################################################################################
#   만드로(아두이노) 설정
if(use_robot==True):
    # 만드로로 보낼때
    ser = serial.Serial(
        port='COM4',
        baudrate=115200
    )


########################################################################################################################
#   암밴드 사용 위한 class
class EmgCollector(myo.DeviceListener):
    """
    Collects EMG data in a queue with *n* maximum number of elements.
    """

    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    # myo.DeviceListener

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))


########################################################################################################################
#   현재 시간 계산
########################################################################################################################
#   시간 계산 함수
def now_time(start,end):
    tmp_time = end-start
    return(tmp_time)


########################################################################################################################
#   keras 모델 불러오기
def keras_load_model():
    json_file = open(model_folder + model_file_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    load_model = model_from_json(loaded_model_json)
    load_model.load_weights(model_folder + model_file_name + ".h5")
    print("Loaded model from disk")
    #print(load_model)
    return load_model

########################################################################################################################
#   실제로 실행되는 부분
def main():
    data_time_ms = 3 # 암밴드 샘플링 타임 (ms)
    eachset_range = test_range//time_part # 한번에 모아질 데이터 길이(샘플 수)
    eachset_time_ms=eachset_range*data_time_ms
    test_data = np.zeros((test_range, test_ch), np.float32) # 케라스로 테스트 할 데이터 프레임
    emg_dataset = np.zeros((eachset_range, test_ch), np.int32) # 한번에 모으는 데이터 프레임
    shape_change_test_data = np.zeros((1, test_range * test_ch), np.float32)  # 테스트 데이터 모양 변경위한 프레임
    count = 0
    pre_move = 10

    # 필요한 변수들 선언 및 초기설정
    num=0
    data_numbers = 1
    time_1by10 = eachset_range // 10
    check=False
    now_motion = 0
    test_time=0
    fig,ax=plt.subplots()
    a=0
    b=0
    c=0
    d=0
    e=0
    z = 0
    # 암밴드 init
    myo.init()
    hub = myo.Hub() # Hub() class는 _ffi파일에 존재함
    listener = EmgCollector(data_numbers)

    load_model=keras_load_model()   # 저장돼있는 케라스 모델 불러오기

########################################################################################################################
#   암밴드로 데이터 획득, 계속 반복
    while hub.run(listener.on_event, data_time_ms):

        emg_data = listener.get_emg_data()  # 암밴드에서 emg 데이터 획득
        emg_data = np.array([x[1] for x in emg_data])   # 획득한 데이터 중에 raw emg 신호만 분리
        # 획득한 emg 데이터가 비어있을 경우 (제대로 측정이 안되었을때)
        if len(emg_data)==0:
            print('\n', 'emg신호 찾는중 ')

        # emg 데이터가 제대로 들어오면 동작 실행
        elif len(emg_data)>0:
            # emg 데이터 확인되면 메시지 출력 (처음 1회)
            if(check==False):
                print("신호 확인 / test start")
                check=True
            else:
                # 데이터 획득 시작할 때 동작
                if(num==0):
                    save_start_time = time.time()   # 데이터 획득 시작시간 체크
                '''
                # 측정시간 화면에 나타내기 위한 코드
                if ((num+1)%time_1by10==0):
                    elapsed_time=eachset_time_ms//time_1by10    #((num+1)/(time_1by10))
                    print("\relapsed time : "+str(elapsed_time) + "  ["+("|"*int(elapsed_time))+("-"*(100-int(elapsed_time)))+"]",end='')
                ''' # 측정시간 화면에 나타내주기 (측정시간이 너무 짧을땐 사용하지 않음)

                # 획득한 데이터를 처리하기 편하게 특정 모양의 다른 변수로 옮겨서 저장
                for i in range(test_ch):
                    emg_dataset[num][i] = emg_data[0][i]
                    # (8,)배열에 저장 emg_dataset은 10msec의 sEMG값 저장
                    # 0 ~ 8까지 각 채널에 대한 1개 데이터를 저장한다.
                num+=1

                # 원하는 길이만큼 데이터 모이면 동작
                if(num==eachset_range):  # 모아진 데이터 노말라이즈
                    num = 0 # 데이터 카운터 초기화
                    test_time += 1
                    save_end_time = time.time()   # 데이터 다 모인 후 시간 체크
                    saving_time = now_time(save_start_time, save_end_time)    # 데이터 수집에 걸리는 시간
                    print("s : ",saving_time)
                    process_start = time.time()   # 전처리 시작시간 체크
                    # 테스트 데이터 변환 (앞부분 잘라내고 뒷부분에 새로 모은 데이터 붙여서 데이터 구성)
                    test_data[:test_range-eachset_range][:]=test_data[eachset_range:][:]
                    test_data[test_range-eachset_range:][:] = emg_dataset[:][:]

# (100,8)만큼의 데이터 취득 후, time_part = 4이면 0~75에다가 0~25개의 test_data 값을 입력하고 75~100에다가 25개의 emg_dataste을 입력
########################################################################################################################
#   전처리 진행
                    if(process_mode==True):
                        cwt_data_list = []
                        fulldata = np.zeros((test_data.shape[0], test_data.shape[1]), np.float32)
                        # cwt 과정 진행
                        for i in range(test_data.shape[1]):
                            width = [1, widths + 1]
                            mother_wavelet_name = 'morl'
                            tmpdata = np.abs(test_data[:, i])
                            cwtdata = pywt.cwt(tmpdata, width,
                                               mother_wavelet_name)  # signal.cwt(tmpdata,signal.ricker,width)
                            cwttmp = np.asarray(cwtdata[0])
                            cwt_data_list.append(cwttmp.transpose())  # cwtdata=cwtdata.transpose()
                            full_cwt_data = np.float32(cwt_data_list[i])
                            # cwt 결과 average 하는 과정
                            for j in range(test_data.shape[0]):
                                data_avrg = np.average(full_cwt_data[j])
                                fulldata[j, i] = data_avrg

                    # 전처리 진행
                    else:
                        fulldata = test_data # (100,8)
                    
                    test_data = np.abs(test_data)
                    
                    testdata = np.zeros((8, 100), np.float32)
                    for i in range(8):
                        for j in range(100):
                            testdata[i][j] = test_data[j][i]
                    test_image = np.zeros((32, 40), np.float32)

                    data_chanels = 8
                    data_samples = 100

                    test_data2 = []
                    test_elements_sum = 0
                    test_all_average = []
                    for i in range(data_chanels):
                        for k in range(10):
                            test_data2 = testdata[i][k*10:k*10+10]
                            for h in range(len(test_data2)):
                                test_elements_sum += test_data2[h]
                            average = test_elements_sum/10
                            test_all_average.append(average)
            
                            test_elements_sum = 0
                            average = 0
                    i = 0
                    for ch in range(data_chanels):
                        for wd in range(10):
                            for n in range(4):
                                for m in range(4):
                                    test_image[(31-ch*4)-n][m+4*wd] = test_all_average[i]
                            i+=1
                    
                    # 데이터 모양 케라스에 입력하기 좋은 형태로 변형
                    """for i in range(test_range):
                        for k in range(test_ch):
                            shape_change_test_data[0][(i * test_ch) + k] = fulldata[i][k]"""
                    process_end=time.time() # 전처리 과정 끝난 시간 체크
                    processing_time=now_time(process_start,process_end) # 전처리 과정 걸린 시간 체크
                    print("p : ",processing_time)

                    if(test_time==0):
                        shape_change_test_data = np.zeros((32,40,1), np.float32)
                    elif(test_time<4 and test_time>0):
                        print("init. test")
                    elif(test_time==4):
                        time.sleep(1),
                        print("\n", "next motion = ", motion_set[now_motion])

                        for i in range(2):
                            print("\rwait " + str(2 - i) + " sec", end='')
                            time.sleep(1)
                        print("\rwait " + "0" + " sec")
                        print("start")
                    else:
                    
########################################################################################################################
#   케라스로 동작 예측

                        keras_test_start = time.time()  # 케라스 test 시작 시간 체크
                        # 케라스 input 에 맞게 데이터 모양 변환
                        test_input_data = test_image.reshape(1, 32, 40, 1)
                        keras_pred = load_model.predict(test_input_data)  # 케라스로 어떤 동작인지 예측
                        argmax_pred = np.argmax(keras_pred, axis=1)
                        keras_test_end=time.time()  # 케라스 test 끝난 시간 체크
                        keras_time=now_time(keras_test_start,keras_test_end)    # 케라스 test 걸린 시간 체크

                        print("k : ",keras_time) # test에 걸린 시간
                        print("delay : ", (saving_time + processing_time + keras_time) - ((test_range*(data_time_ms/1000))/time_part))
                        # 저장, 전처리, 딥러닝 시간 - 데이터 취득 시 걸린 시간
                        #shape_change_test_data = np.zeros((32,40,1), np.float32)  # 테스트 데이터 프레임 초기화
                        pred_label.append(argmax_pred)
                        print("predict : ", argmax_pred , ", motion : ",motion_set[argmax_pred[0]])  # 예측한 값 출력
                        print(len(pred_label),"/",full_test_time)





########################################################################################################################
#   만드로(아두이노)로 신호 전송
                        if(argmax_pred==0):
                            a = a+1
                        elif(argmax_pred == 1):
                            b = b+1
                        elif(argmax_pred == 2):
                            c = c+1
                        elif(argmax_pred == 3):
                            d = d+1
                        elif (argmax_pred == 4):
                            e = e + 1


                        if(count == 0):
                            pre_pred = argmax_pred
                            count += 1
                        elif(count < 2):
                            if(pre_pred == argmax_pred):
                                pre_pred = argmax_pred
                                count += 1
                            else:
                                pre_pred = argmax_pred
                                count = 0
                        else:
                                if(pre_pred == argmax_pred):
                                    print(" 아두이노 데이터 전송")
                                    # if (use_robot == True):
                                    predict_signal = str(argmax_pred[0] + 1).encode()
                                    ser.write(predict_signal)  # 아두이노로 신호 보냄
                                    count = 0
                                    #pre_move = argmax_pred
                                    #pre_pred = argmax_pred
                                else:
                                    count = 0


                        pre_pred = argmax_pred
                        print(count)
########################################################################################################################
#   실시간 동작 예측 정확도 확인 및 출력

                        if(len(pred_label)%(full_test_time/motion_num)==0 and len(pred_label)!=full_test_time):
                            time.sleep(1)
                            print("\n",motion_set[now_motion],"motion end")
                            now_motion+=1
                            print("\n","next motion = ",motion_set[now_motion])
                            print(a, " ", b, " ", c, " ", d, " ", e)
                            for i in range(5):
                                if(i == 0):
                                    result[z][i] = a
                                elif(i==1):
                                    result[z][i] = b
                                elif (i == 2):
                                    result[z][i] = c
                                elif (i == 3):
                                    result[z][i] = d
                                elif (i == 4):
                                    result[z][i] = e
                            z = z+1
                            print(result)
                            for i in range(3):
                                print("\rwait "+str(3-i)+" sec",end='')
                                time.sleep(1)
                            print("\rwait "+ "0"+ " sec")
                            print("start")
                            test_time=0

                        if(len(pred_label)==full_test_time):
                            hub.stop()
                            clear_session()
                            #K.clear_session()
                            del load_model
                            print("test end")
                            #print(pred_label)
                            correct_pred=[]
                            for i in range(full_test_time):
                                correct_pred.append(correct_label[i]==pred_label[i])
                            correct_sum=correct_pred.count(True)
                            print(correct_sum,"/",full_test_time,"  =  ",correct_sum/full_test_time)
                            ax.plot(pred_label,label="pred",marker=".",linestyle="None")
                            ax.plot(correct_label,label="correct")
                            ax.set_ylabel("class")
                            ax.set_xlabel("test time")
                            break


########################################################################################################################
if __name__ == '__main__':
    main()
    plt.legend()
    plt.show()