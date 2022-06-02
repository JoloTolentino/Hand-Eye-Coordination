
from pyexpat import features
import torch
import torch.nn as nn
from torch.nn import functional as Function
import torch.optim as optim
from torchvision import transforms,datasets,models
from PIL import Image
# from torchsummary import summary
from Modified_Resnet import EarlyExitResnet,StackedEarlyExit 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import json
import os 
import numpy as np
from sklearn.linear_model import LogisticRegression
from statistics import mode






validation_dir = "./test/" 


with open('./COCO_DATA/Mapping.json','r') as data: 
    Json = json.load(data)

with open('./Torch_Mapping.json','r') as data:
    Torch_Map = json.load(data)


Map = {Json[key]:key for key in Json}
print(Map)
print(Torch_Map)



Confusion_Matrix_Labels = [Map[int(val)] for val in Torch_Map] 
print(Confusion_Matrix_Labels)

Images = [] 
Ground_Truth = [] 


for names in os.listdir(validation_dir):
    for images in os.listdir(os.path.join(validation_dir,names)):
         Images.append(os.path.join(validation_dir,names,images))
         Ground_Truth.append(Torch_Map[str(Json[names])]) 


device = torch.device("cuda")
mean = [0.485,0.456,0.406]
std = [0.229,0.224,0.225]

Preprocessing = transforms.Compose([transforms.Resize((224,224)),
                                    
                                    transforms.RandomRotation(degrees = 20),
                                    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,std)])

 
network = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).to("cuda")
# transfer_learning_weights = "./Transfer-Learning.h5"
# network.fc = nn.Sequential(
#     nn.Linear(512, 256),# Fully Connected Layer
#     nn.ReLU(), # activation function 
#     nn.Linear(256, 30)).to(device)
# network.load_state_dict(torch.load(transfer_learning_weights))
# network.to(device).cuda()
# network.eval()


############# EARLY EXIT RESNET ###############

network = EarlyExitResnet(network)
# weights = './Early-Exit-Features-original_LR_5-15_layer_loss_normal.h5' - 73.19
weights = "./test-inverse-50.h5"
network.load_state_dict(torch.load(weights))
network.to(device).cuda()
network.eval()

# stacked_network = StackedEarlyExit(network)
# stacked_network_weights = './Stacked_Early_Exit_Weights.h5' 
# stacked_network.load_state_dict(torch.load(stacked_network_weights))
# stacked_network.eval() 



# summary(network,(3,224,224))
    
correct = 0.0

votingCorrects = 0.0
stacked_network_correct =0.0
Total = len(Images)

gt = []
Predictions  = [] 
Stacked_Predictions = []
layer1 = []
layer2 = [] 
layer3 = []
layer4 = []
layer5 = []

# test_class = nn.Sequential(nn.Linear(150,30),nn.Softmax(dim=0))

for images,label in zip(Images,Ground_Truth): 
    input_tensor = Preprocessing(Image.open(images).convert("RGB"))
   
    input_batch = input_tensor.unsqueeze(0).cuda()    
    input_batch.to(device)
    
    estimator = 3 
    stack = np.zeros((5,estimator)) 
    with torch.no_grad():
        outputs = network(input_batch)  
        
    Fully_Connected_Layer_Preds= outputs[-1]
    Fully_Connected_Pred = torch.topk(Fully_Connected_Layer_Preds, 1)[1][0].cpu().detach().numpy() 
   
    Predictions.append(Fully_Connected_Pred)
    # Predictions.append(Fully_Connected_Pred[0])
   
    gt.append(label)
    
    
    if label in Fully_Connected_Pred: correct+=1
    
  

# print(len(gt), len(Predictions))
print(layer1)








mat_con = np.round(confusion_matrix(gt,Predictions),decimals=1)
score = correct/Total
print(score,stacked_network_correct/Total)



fig, px = plt.subplots(figsize=(7.5, 7.5))
px.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
for m in range(mat_con.shape[0]):
    for n in range(mat_con.shape[1]):
        px.text(x=m,y=n,s=mat_con[m, n], va='center', ha='center', size='xx-large')
        xaxis = np.arange(len(Confusion_Matrix_Labels))
        # px.set_xticks(xaxis)
        px.set_yticks(xaxis)
        # px.set_xticklabels(Confusion_Matrix_Labels,fontsize = 15)
        px.set_yticklabels(Confusion_Matrix_Labels,fontsize = 15)

plt.xlabel('Predictions', fontsize=30)
plt.ylabel('Actuals', fontsize=30)
plt.title('Confusion Matrix', fontsize=30)
plt.show()