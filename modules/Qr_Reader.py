import cv2
from modules.Jorjin import jorjin
import time


class QR_Reader:
    def __init__(self):
        QR_detetctor = cv2.QRCodeDetector()
        self.Condition = False
        self.Name = "QR-Reader"
        print('Intializing {}'.format(self.name()))
        Android_Glasses = jorjin("QR-Reader")
        self.stream  = None 
        if self.Condition != True:
            while True: 
                stream = Android_Glasses.cv_frame()
                self.Feed = stream
                data,_,_= QR_detetctor.detectAndDecode(stream)
            
                if data == "Activate":
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

    def feed(self):
        return self.Feed

