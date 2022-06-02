
import torch
from torch.nn import functional as Function
from torchvision import transforms
from PIL import Image
import json 
from PIL import ImageGrab
try: 
    from modules.Modified_Resnet import EarlyExitResnet 
    from modules.Jorjin import jorjin
    from modules.Hard_Voting import Hard_Voting

except: 
    from Modified_Resnet import EarlyExitResnet
    from Jorjin import jorjin
    from Hard_Voting import Hard_Voting



class Predictor: 
    def __init__ (self,comms):
        
        self.network = EarlyExitResnet()
        self.Name = "Early-Exit-Ensemble Module"
        weights = "./modules/models/Early_Exit-Final.h5"
        self.network.load_state_dict(torch.load(weights))
        with open('./modules/Mapping.json','r') as data: 
            self.Json = json.load(data)
        with open('./modules/Torch_Mapping.json','r') as data:
            self.Torch_Map = json.load(data)

        self.network.eval().cuda()
        self.device = torch.device("cuda") 
        print('initializing {}'.format(self.Name))
        self.Hard_Voting_Filter = Hard_Voting()
        self.Hard_Voting_Queue = []
        self.Prediction()
    

    def Prediction(self): 
        
        
        self.Condition = False
        self.Predicting = True 
        stream = jorjin("Early Exit Ensemble") 
        
        while self.Predicting: 
            data  = stream.pil_frame()
            input_tensor = self.Preprocess(data.convert('RGB'))
            input_batch = input_tensor.unsqueeze(0).cuda()
            self.filter(self.Predict(input_batch))
            if self.Condition:
                self.Predicting = False
                break
            
    def Preprocess(self,data): 
        mean = [0.485,0.456,0.406]
        std = [0.229,0.224,0.225]
        Preprocessing = transforms.Compose([transforms.Resize((224,224)),
                                    
                                    transforms.RandomRotation(degrees = 20),
                                    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,std)])

        return Preprocessing(data)

    def filter(self,data):
        self.Condition,self.Hard_Voting_Queue = self.Hard_Voting_Filter.verificaiton(data)
    
    def condition(self):
        return self.Condition

    def Predict(self,input_batch):
        with torch.no_grad():
            logits = self.network(input_batch) 
        prediction = logits[-1]                                   
        prediction = torch.topk(prediction, 1)[1][0].cpu().detach().numpy()   
        print(prediction) 

        return  prediction 

    def name(self):
        return "Early-Exit-Ensemble Module"

 