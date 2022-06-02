
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torchvision.models.resnet import ResNet, BasicBlock
class EarlyExitResnet(ResNet):
    def __init__(self,pretrained = None):
        super().__init__(BasicBlock, [2, 2, 2, 2])

        if pretrained:
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4

        self.layer1[0].add_module("ac",
                        nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(64,30)
                       ))

                                    
        self.layer1[1].add_module("ac",
                        nn.Sequential(
                        nn.Linear(64,128), #new addition
                        nn.ReLU(),
                        nn.Linear(128,30)
                        ))

            
        self.layer2[0].add_module("ac",
                        nn.Sequential(
                        nn.Linear(128, 256),
                        nn.ReLU(), 
                        nn.Linear(256,30),
                        ))


        self.layer2[1].add_module("ac",
                        nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(128, 256),
                        nn.Linear(256,30)
                        ))
        self.layer3[0].add_module("ac", 
                        nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(256, 30)
                        ))
        self.layer3[1].add_module("ac",
                        nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(256,30)
                        ))

        self.layer4[0].add_module("ac",
         nn.Sequential(
                nn.Linear(512,256),
                nn.ReLU(),
                nn.Linear(256,30),             
                
        ))

        self.layer4[1].add_module("ac",
        nn.Sequential(
                nn.Linear(512,256),
                nn.ReLU(),
                nn.Linear(256, 30),
                
        ))
        self.device = ("cuda")

        self.fc = nn.Sequential(
                            nn.Linear(512,256),
                            nn.ReLU(),
                            nn.Linear(256,30),                      
                            ).to(self.device)




   
    def GlobalPool(self,ResidualBlock:torch.tensor) ->torch.tensor:
        return ResidualBlock.mean([2,3])
    
    def forward(self, x) -> torch.tensor:
        x = self.conv1(x).to(self.device)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
          
   
        x = self.layer1[0](x)
        pool1 = self.GlobalPool(x)
        exit1 = self.layer1[0].ac(pool1)
     
     
        x  = self.layer1[1](x)
        pool2 = self.GlobalPool(x)
        exit2 = self.layer1[1].ac(pool2)
       

        x = self.layer2[0](x)
        pool3 = self.GlobalPool(x)
        exit3 = self.layer2[0].ac(pool3)
        
        x = self.layer2[1](x)
        pool4 = self.GlobalPool(x)
        exit4 = self.layer2[1].ac(pool4)
        

        x = self.layer3[0](x)
        pool5 = self.GlobalPool(x)
        exit5 = self.layer3[0].ac(pool5)
       
        x= self.layer3[1](x)
        pool6 = self.GlobalPool(x)
        exit6 = self.layer3[1].ac(pool6)
        

        x = self.layer4[0](x)
        pool7= self.GlobalPool(x)
        exit7 = self.layer4[0].ac(pool7)
        
        x = self.layer4[1](x)
        pool8 = self.GlobalPool(x)
        exit8 = self.layer4[1].ac(pool8)
        
        
       
        x = self.avgpool(x)
        x=x.view(x.size(0), -1)
        x= self.fc(x)
        

        return exit1,exit2,exit3,exit4,exit5,exit6,exit7,exit8,x         
    



class StackedEarlyExit(nn.Module):
    def __init__(self, Early_Exit):
        super(StackedEarlyExit, self).__init__()
        self.device = ("cuda")
        self.Early_Exit = Early_Exit.eval().cuda()
        self.Ensemble_Classifier = nn.Sequential(nn.Linear(90,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,30))
    
    def One_Hot(self,indexes):
        feature_layer =np.zeros(30)
        for index in indexes:
            feature_layer[index] = 1  
        return torch.FloatTensor(feature_layer)

    def forward(self,x):
        
        preds = self.Early_Exit(x)[2:]
        batch_size = preds[0].shape[0]
        
        'One Hot Encoding of Predictions'
        predictions = [] 
        for pred in preds: 
            top_vals = torch.topk(pred,1)[1]
            predictions.append(top_vals)
        features = []
        for index in range(0,batch_size):
            prediction_feature = []
            for layers in predictions:
                prediction_feature.append(self.One_Hot(layers[index].cpu().detach().numpy()))
            features.append(torch.cat(prediction_feature))

        early_exit_logits = torch.stack(features).cuda() 
        stacked_prediction = self.Ensemble_Classifier(early_exit_logits)
 

        return stacked_prediction   
    
   


