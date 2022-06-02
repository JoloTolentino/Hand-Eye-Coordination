from tkinter import Image
# import pyautogui
import pygetwindow as gw
from PIL import ImageGrab
import numpy as np

class jorjin :
    'Takes in the first POV of the amputee for  Gesture Classification' 

    def __init__(self,node_name = None):
        self.Window_ID = "msm8909w"
        self.Data_Directory = "./data"
        self.Process = gw.getWindowsWithTitle(self.Window_ID)[0]
        if node_name:
            print(node_name)
    
    def cv_frame(self):
        left,right,top,bottom = int(self.Process.left)+60 ,int(self.Process.left)-60 + int(self.Process.width),int(self.Process.top)+250,int(self.Process.top)+ int(self.Process.height)-250
        bbox = (left,top,right,bottom)
        feed = ImageGrab.grab(bbox)
        # feed.show()

        feed =  np.array(feed) 
        Processed = np.copy(feed)
        Processed[:,:,0] = feed[:,:,2]
        Processed[:,:,2] = feed[:,:,0]

        return feed

    def pil_frame(self):
        left,right,top,bottom = int(self.Process.left)+60 ,int(self.Process.left)-60 + int(self.Process.width),int(self.Process.top)+250,int(self.Process.top)+ int(self.Process.height)-250
        bbox = (left,top,right,bottom)
        feed = ImageGrab.grab(bbox)

        return feed
        
        
            
