import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np


def adjust_rate(lr,lrdecacy,epoch):
    lr=lr*(lrdecacy**epoch)
    return lr
a=0.05
b=0.05

class csvdataset(Dataset):
    def __init__(self,filepath="D://python study/csv文件/train.csv"):
        df=pandas.read_csv(
            filepath,
            encoding='utf-8',
            names=['A','B','C','D','E'],
            dtype={'A':np.float32,'B':np.float32,'C':np.float32,'D':np.float32,'E':np.float32},
            skip_blank_lines=True)
        x=df.iloc[:,0:4].values
        y=df.iloc[:,4].values
        
        self.x=torch.from_numpy(x)
        self.y=torch.from_numpy(y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self,index):
        return self.x[index],self.y[index] 

csv_dataset=csvdataset()
csv_dataloader=DataLoader(csv_dataset,batch_size=128,shuffle=True)







class Net(nn.Module):
    def __init__(self,ind,hid,otd):
        super(Net,self).__init__()
        self.net=nn.Sequential(nn.Linear(4,10),nn.ReLU(),nn.Linear(10,1))
    
    def forward(self,x):
        x=self.net(x)
        return x
  
device=torch.device("cuda" if torch.cuda.is_available else "cpu")  
model=Net(4,10,1).to(device)    

print(model)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=a,betas=(0.9,0.99))

def train(dataloader,model,criterion,optimizer):
    for batch,(x,y) in enumerate(dataloader):
        optimizer.zero_grad()
        x,y=x.to(device),y.to(device)
        y=y.unsqueeze(1)
       
       
        loss=criterion(model(x),y)
        
        loss.backward()
        optimizer.step()
        if batch % 1 == 0:
            loss=loss.item()
            print(batch,loss)
            
epochs=100
for t in range(epochs):           
    train(csv_dataloader,model,criterion,optimizer)
    #a=adjust_rate(a,0.9,t+1)

print(model.state_dict())
