from modules.Communication import ProstheticBT 
from modules.Qr_Reader import QR_Reader
from modules.Early_Exit import Predictor
from modules.Jorjin import jorjin


class Linked_List:
    def __init__(self,node,communication):
        self.process = node
        self.next = {"QR-Reader":Predictor,
                     "Early-Exit-Ensemble Module":QR_Reader}
        
        
        self.communication = communication

    def begin(self):
        self.process.begin(self.communication)
        print(self.process.library.name())
        if self.process.condition():
            self.process = Node(self.next[self.process.library.name()](),self.communication) 

class Node:
    def __init__(self,val):
        self.library  = val
        self.next = None

    def begin(self,communication):
        while True: 
                if self.library.Condition:
                    communication.send(self.library.message())
                    break
    def condition(self):
        return self.library.condition() 

def main(): 
    BT_Coms = ProstheticBT("24:A1:60:74:E9:7E","ESPArm") 
    
    Head_node = Node(QR_Reader())
    Sequence =  Linked_List(Head_node,BT_Coms)
    while True:   
        Sequence.begin()





if __name__ == "__main__":
    main()