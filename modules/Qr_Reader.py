import cv2
from modules.Jorjin import jorjin
import time


class QR_Reader:
    def __init__(self,coms):
        QR_detetctor = cv2.QRCodeDetector()
        self.Condition = False
        self.Name = "QR-Reader"
        print('Intializing {}'.format(self.name()))
        Android_Glasses = jorjin("QR-Reader")
        if self.Condition != True:
            while True: 
                stream = Android_Glasses.cv_frame()
                data,_,_= QR_detetctor.detectAndDecode(stream)
                cv2.imshow("QR-Reader stream",stream)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if data == "Activate":
                    coms.send("<2>") 
                    print('activate')
                    time.sleep(2)
                    self.Condition = True  
                    break    

    def condition(self):
        return self.Condition

    def message(self):
        if self.Condition:
            return "<2>"
    def name(self):
        return self.Name


