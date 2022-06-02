import cv2
from modules.Jorjin import jorjin



class QR_Reader:
    def __init__(self):
        QR_detetctor = cv2.QRCodeDetector()
        self.Condition = True
        self.Name = "QR-Reader"
        print('Intializing {}'.format(self.name()))
        Android_Glasses = jorjin("QR-Reader")
        if self.Condition != True:
            while True: 
                stream = Android_Glasses.cv_frame()
                data,_,_= QR_detetctor.detectAndDecode(stream)
                if data == "Activate":
                    # coms.send("<2>") 
                    self.Condition = True             
                    break    

    def condition(self):
        return self.Condition

    def name(self):
        return self.Name


