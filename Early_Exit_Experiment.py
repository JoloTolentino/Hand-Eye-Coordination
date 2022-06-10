from modules.Early_Exit import Predictor
from modules.Communication import ProstheticBT
from modules.Jorjin import jorjin
import threading
import cv2

def stream(glasses):
    while True:
        data = glasses.cv_stream()
        cv2.imshow("stream",data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

Bt = ProstheticBT("24:A1:60:74:E9:7E","ESPArm")
glasses = jorjin("Test")
thread= threading.Thread(target=stream, args=(glasses))
thread.start()

while True:
    data = glasses.cv_frame()
    # cv2.imshow("info",data)
    test = Predictor(data,Bt)
    