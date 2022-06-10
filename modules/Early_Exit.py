import cv2
import torch
from torchvision import transforms
import json 
import numpy as np 
import os
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
        self.Object_Status = False
        self.Objectnes_filter(data)
    
    def Objectnes_filter(self,data):
        
        Color_Contrast_Features = self.Color_Contrast(data)
        
        Blurred = cv2.blur(data,(7,7))
        Edge_Features = cv2.Canny(Blurred,100,100)
        Dilated_Edge_Features = cv2.dilate(Edge_Features,np.ones((5,5),np.uint8),iterations=1)

        Scaled_Color_Contrast_Mask = cv2.convertScaleAbs(Color_Contrast_Features)
        Scaled_Edge_Features = cv2.convertScaleAbs(Dilated_Edge_Features)
        
        
        kernel = np.ones((5,5),np.uint8)
        Closing = cv2.morphologyEx(Scaled_Color_Contrast_Mask,cv2.MORPH_CLOSE,kernel=kernel)
        Opening = cv2.morphologyEx(Closing,cv2.MORPH_OPEN,kernel=kernel)

        Saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (_, Saliency_Map) = Saliency.computeSaliency(data)
        CV2_SaliencyMap = cv2.convertScaleAbs(Saliency_Map*255)

        threshMap = cv2.threshold(CV2_SaliencyMap   , 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        consolidated = cv2.bitwise_and(Opening,threshMap) 
        consolidated = cv2.morphologyEx(consolidated, cv2.MORPH_CLOSE, kernel=kernel)# take out later
        consolidated = cv2.medianBlur(consolidated,7)
        kernel =  np.ones((11,11),np.uint8)
        consolidated = cv2.morphologyEx(consolidated, cv2.MORPH_OPEN, kernel)*255
        
        final = cv2.bitwise_and(consolidated,Scaled_Edge_Features)
        
        gestureActivation = "False"
        contours,_ = cv2.findContours(final,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        try: 
            c = max(contours, key = cv2.contourArea)
            _,_,w,h = cv2.boundingRect(c)    
            area = h*w
            if area > 1000 :os.system('cls') ; self.Object_Status = True; gestureActivation = 'True'; print("Objects",end = "\r")
        except: 
            print('No objects',end='\r')
            

    

    def Color_Contrast(self,image): 
        copy_image = image.copy()
        gray= cv2.cvtColor(copy_image,cv2.COLOR_BGR2GRAY) 

        gray_array = np.array(gray)
        
        log_transform = np.log2(gray_array) #best
        
        Color_Contrast_Gradient = np.zeros(np.array(log_transform).shape)
        num_cols = log_transform.shape[1]
        num_rows = log_transform.shape[0]

        left_col = log_transform[:,:num_cols-1]
        right_col  = log_transform[:,1:num_cols]

        top_col = log_transform[:num_rows-1,:] 
        bot_col = log_transform[1:num_rows,:]

        Discrete_Difference_X = right_col-left_col
        Discrete_Difference_Y = top_col-bot_col

        Color_Contrast_Gradient[:,1:num_cols]+= Discrete_Difference_X
        Color_Contrast_Gradient[1:num_rows,:]+= Discrete_Difference_Y 
       

        return np.floor((Color_Contrast_Gradient)*255) 



class Predictor: 
    def __init__ (self):

        self.obj_gesture_map = {
        "person":"<1>","backpack ":"<1>","chair ":"<3>","couch ":"<3>","potted plant": "<3>", 
        "bed": "<1>","dining table" :"<1>","toilet":"<4>","laptop": "<4>","remote": "<4>",
        "keyboard": "<4>","cell phone": "<1>","umbrella" :"<1>", "microwave": "<4>","oven": "<5>", 
        "sink": "<1>","regfigerator": "<3>","book" :"<1>", "clock": "<4>","vase": "<1>","scissors": "<1>",
        "teddy bear": "<1>","door handle":"<1>","handbag" :"<1>","bottle": "<1>","cup":"<1>","utensils": "<1>",
        "Garbage": "<1>","light switch": "<4>","bowl": "<3>"}
        
        
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

        glasses = jorjin("Early Exit Ensemble Module")
        while True: 
            stream = glasses.cv_frame()

            
            self.Objectness_Filter = Objectness_Filter(stream)
            if self.Objectness_Filter.Object_Status:
                print('initializing {}'.format(self.Name))
                self.Hard_Voting_Filter = Hard_Voting()
                self.Hard_Voting_Queue = []
                self.Prediction()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        

    def Prediction(self): 
        
        
        self.Condition = False
        self.Predicting = True 
        glasses = jorjin("Early Exit Ensemble") 
        
        while self.Predicting: 
            data  = glasses.pil_frame()
            input_tensor = self.Preprocess(data.convert('RGB'))
            input_batch = input_tensor.unsqueeze(0).cuda()
            self.filter(self.Predict(input_batch))
            if self.Condition:
                self.comms.send(str(self.message))
                self.Objectness_Filter.Object_Status = False
                self.Predicting = False
                
            

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

 