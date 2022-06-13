
import PySimpleGUI as sg
import cv2
import numpy as np
from Config import Armature
from modules.Jorjin import jorjin
import threading 
import sys


def Backend():
    process = Armature()
    Daemon = threading.Thread(target=User_Interface)
    Daemon.start()
    process.begin()



def User_Interface():
    sg.theme('DarkTanBlue')
    
    layout = [[sg.Text('Armature', size=(25, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='feed')],
              [sg.Button('Start', size=(10, 1), font='Helvetica 14'),
               sg.Button('Stop', size=(10, 1), font='Helvetica 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ]]

 
    window = sg.Window('Armature Demo',
                       layout, location=(2800, 400),element_justification='c')

 
    Android_Glasses = jorjin() 
    Stream = None

    while True:
        event, _ = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return
        if event == 'Start':Stream = True
        if event == 'Stop':Stream = False;window['feed'].update(data=None);break
        if Stream :frame = Android_Glasses.cv_frame(); imgbytes = cv2.imencode('.png', frame)[1].tobytes(); window['feed'].update(data=imgbytes)

            
if __name__ == "__main__":
    Backend()