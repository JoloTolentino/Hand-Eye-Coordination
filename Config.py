from modules.Communication import ProstheticBT 
from modules.Qr_Reader import QR_Reader
from modules.Early_Exit import Predictor
import os


class Linked_List:
    def __init__(self,node,communication):
        print("Linked List")
        self.process = node
        self.next = {"QR-Reader":Predictor,
                     "Early-Exit-Ensemble Module":QR_Reader}
        self.communication = communication
     
    def begin(self):
        self.process.begin(self.communication)
        if self.process.condition():
            self.process = Node(self.next[self.process.library.name()]()) 

    def stream(self):
        return  self.process.stream()
class Node:
    def __init__(self,val):
        os.system("cls")
        self.library  = val
        self.next = None

    def begin(self,communication):
        while True: 
                if self.library.Condition:
                    communication.send(self.library.message())
                    break
    def condition(self):
        return self.library.condition() 

    def stream(self):
        return self.library.feed()

class Armature: 
    def __init__(self): 
        self.BT_Coms = ProstheticBT("24:A1:60:74:E9:7E","ESPArm") 
        
        
    def begin(self):
        Head_node = Node(QR_Reader())
        Sequence =  Linked_List(Head_node,self.BT_Coms)
        self.state = True 
        while self.state:
            Sequence.begin()
            
    def kill(self):
        print('Terminating Backend....')
        self.state = False


    
        
