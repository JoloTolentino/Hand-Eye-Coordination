import cv2
import torch
from torchvision import transforms
import json 
import numpy as np 
import threading
try: 
    from modules.Modified_Resnet import EarlyExitResnet 
    from modules.Jorjin import jorjin
    from modules.Hard_Voting import Hard_Voting

except: 
    from Modified_Resnet import EarlyExitResnet
    from Jorjin import jorjin
    from Hard_Voting import Hard_Voting


class Objectness_Filter: 
    def __init__ (self, data): 
        pass
    

    def EdgeDetection(image):
        copy_image = image.copy()
        gaus = cv2.GaussianBlur(copy_image,(3,3),0)
        a= cv2.cvtColor(copy_image,cv2.COLOR_BGR2GRAY) 
        a = np.array(a)
    
       
        a = np.log(a) #best
        A = np.zeros(np.array(a).shape)
        num_cols = a.shape[1]
        num_rows = a.shape[0]
        left_col = a[:,:num_cols-1]
        right_col  = a[:,1:num_cols]
        top_col = a[:num_rows-1,:] 
        bot_col = a[1:num_rows,:]
        compare = right_col-left_col
        compare2 = top_col-bot_col

        A[:,1:num_cols]+= compare
        A[1:num_rows,:]+= compare2 

        return ((A**2)*256) #11 bit grayscale image


    def ColorContrast(image): 
        copy_image = image.copy()
        a= cv2.cvtColor(copy_image,cv2.COLOR_BGR2GRAY) 

        a = np.array(a)
        
        a = np.log2(a) #best
        
        A = np.zeros(np.array(a).shape)
        num_cols = a.shape[1]
        num_rows = a.shape[0]

        left_col = a[:,:num_cols-1]
        right_col  = a[:,1:num_cols]

        top_col = a[:num_rows-1,:] 
        bot_col = a[1:num_rows,:]

        compare = right_col-left_col
        compare2 = top_col-bot_col

        A[:,1:num_cols]+= compare
        A[1:num_rows,:]+= compare2 

        return np.floor((A)*255) 







class Predictor: 
    def __init__ (self,comms= None):


        self.obj_gesture_map = {
        "person":"<1>",
        "backpack ":"<1>",
        "chair ":"<3>",
        "couch ":"<3>",
        "potted plant": "<3>", 
        "bed": "<1>",
        "dining table" :"<1>",
        "toilet":"<4>",
        "laptop": "<4>",
        "remote": "<4>",
        "keyboard": "<4>",
        "cell phone": "<1>",
        "umbrella" :"<1>", 
        "microwave": "<4>",
        "oven": "<5>", 
        "sink": "<1>",
        "regfigerator": "<3>",
        "book" :"<1>", 
        "clock": "<4>",
        "vase": "<1>",
        "scissors": "<1>",
        "teddy bear": "<1>",
        "door handle":"<1>",
        "handbag" :"<1>",
        "bottle": "<1>",
        "cup":"<1>",
        "utensils": "<1>",
        "Garbage": "<1>",
        "light switch": "<4>",
        "bowl": "<3>"}

        if comms: self.comms = comms
        
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
        glasses = jorjin("Early Exit Ensemble") 
        
        while self.Predicting: 
            video = glasses.cv_frame()
            data  = glasses.pil_frame()
            cv2.imshow("data",video)
            input_tensor = self.Preprocess(data.convert('RGB'))
            input_batch = input_tensor.unsqueeze(0).cuda()
            self.filter(self.Predict(input_batch))
            if self.Condition:
                if self.comms:self.comms.send(self.message)
                self.Predicting = False
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
        if self.Condition: 
            self.message = np.unique(self.Hard_Voting_Queue)[0]
    
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

 