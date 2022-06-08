import numpy as np

class Hard_Voting:
    'Hard voting filter module '
    
    def __init__(self):
        self.memory = []
        self.Condition = False
        print(self.memory)
    def verificaiton(self,data):
        if len(self.memory) < 5: self.memory.insert(0,data[0]);self.Condition = False
        self.Condition = True if len(np.unique(self.memory))==1 and len(self.memory)== 5 else False
        print("memory: {} status : {} unique: {} ".format( self.memory,self.Condition,len(np.unique(self.memory))))
        
        if self.Condition:
            return self.Condition,self.memory

        if len(self.memory)==5:
            self.memory.pop()
            print("popped memory : {}".format(self.memory))
        
        return False,self.memory

    def condition(self):
        return self.Condition
    
    def name(self):
        print(" Hard voting Node")
    
    

    

test = Hard_Voting()

test.verificaiton([3])
test.verificaiton([3])
test.verificaiton([4])
test.verificaiton([3])
test.verificaiton([3])
test.verificaiton([3])
test.verificaiton([3])
test.verificaiton([3])
