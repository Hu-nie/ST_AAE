import torch
import numpy as np
import os
from model import *
from utils import *

# System Parameters
# 1. Mini batch size
mb_size = 128
# 2. Missing rate
p_miss = 0.2
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10
# 5. Train Rate
train_rate = 0.8

#% Data
dataset_file = 'data/V_228.csv'
# Data generation
Data = np.loadtxt(dataset_file, delimiter=",",skiprows=1)

# Parameters
No = len(Data)
Dim = len(Data[0,:])

# Hidden state dimensions
H_Dim1 = Dim
H_Dim2 = Dim

# Normalization (0 to 1)
Min_Val = np.zeros(Dim)
Max_Val = np.zeros(Dim)

for i in range(Dim):
    Min_Val[i] = np.min(Data[:,i])
    #print(np.min(Data[:,i]))
    Data[:,i] = Data[:,i] - np.min(Data[:,i])
    Max_Val[i] = np.max(Data[:,i])
    Data[:,i] = Data[:,i] / (np.max(Data[:,i]) + 1e-6)    
    



# Missing introducing
p_miss_vec = p_miss * np.ones((Dim,1)) 
   
Missing = np.zeros((No,Dim))

for i in range(Dim):
    A = np.random.uniform(0., 1., size = [len(Data),])
    B = A > p_miss_vec[i]
    Missing[:,i] = 1.*B

    
# Train Test Division    
   
idx = np.random.permutation(No)

Train_No = int(No * train_rate)
Test_No = No - Train_No
    
# Train / Test Features
trainX = Data[idx[:Train_No],:]
testX = Data[idx[Train_No:],:]

# Train / Test Missing Indicators
trainM = Missing[idx[:Train_No],:]
testM = Missing[idx[Train_No:],:]

netD = NetD()
netG = NetG()


optimD = torch.optim.Adam(netD.parameters(), lr=0.001)
optimG = torch.optim.Adam(netG.parameters(), lr=0.001)


bce_loss = torch.nn.BCEWithLogitsLoss(reduction="elementwise_mean")
mse_loss = torch.nn.MSELoss(reduction="elementwise_mean")


i = 1
# Start Iterations
for it in range(5000): 
    #%% Inputs
    mb_idx = sample_idx(Train_No, mb_size)
    X_mb = trainX[mb_idx,:]  

    Z_mb = sample_Z(mb_size, Dim) 
    M_mb = trainM[mb_idx,:]  
    H_mb1 = sample_M(mb_size, Dim, 1-p_hint)
    H_mb = M_mb * H_mb1 
    
    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
    
    X_mb = torch.tensor(X_mb).float()
    New_X_mb = torch.tensor(New_X_mb).float()
    Z_mb = torch.tensor(Z_mb).float()
    M_mb = torch.tensor(M_mb).float()
    H_mb = torch.tensor(H_mb).float()
    
    # Train D
    G_sample = netG(X_mb, New_X_mb, M_mb)
    D_prob = netD(X_mb, M_mb, G_sample, H_mb)
    D_loss = bce_loss(D_prob, M_mb)

    optimD.zero_grad()
    D_loss.backward()
    optimD.step()
    
    
    # Train G
    G_sample = netG(X_mb, New_X_mb, M_mb)
    D_prob = netD(X_mb, M_mb, G_sample, H_mb)
    D_prob.detach_()
    G_loss1 = ((1 - M_mb) * (torch.sigmoid(D_prob)+1e-8).log()).mean()/(1-M_mb).sum()
    G_mse_loss = mse_loss(M_mb*X_mb, M_mb*G_sample) / M_mb.sum()
    G_loss = G_loss1 + alpha*G_mse_loss
    
    G_loss.backward()
    optimG.step()
    optimG.zero_grad()
    
    G_mse_test = mse_loss((1-M_mb)*X_mb, (1-M_mb)*G_sample) / (1-M_mb).sum()


    if it % 100 == 0:
        print('Iter: {}'.format(it),end='\t')
        print('Train_loss: {:.4}'.format(np.sqrt(G_mse_loss.item())),end='\t')
        print('Test_loss: {:.4}'.format(np.sqrt(G_mse_test.item())),end='\t')
        print('G_loss: {:.4}'.format(G_loss),end='\t')
        print('D_loss: {:.4}'.format(D_loss))


def test_loss(X, M):
    
    
    #%% MSE Performance metric
    MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
    return MSE_test_loss

G_sample = netG(X_mb, New_X_mb, M_mb)

Z_mb = sample_Z(Test_No, Dim) 
M_mb = testM
X_mb = testX
        
New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

X_mb = torch.tensor(X_mb).float()
M_mb = torch.tensor(M_mb).float()
New_X_mb = torch.tensor(New_X_mb).float()

MSE_final= test_loss(X=X_mb, M=M_mb)
print('Final Test RMSE: ' + str(np.sqrt(MSE_final.item())))

imputed_data = M_mb * X_mb + (1-M_mb) * G_sample
print("Imputed test data:")
np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})


print(imputed_data.detach().numpy())

# Normalization (0 to 1)
renomal = imputed_data 

for i in range(Dim):
    renomal[:,i] = renomal[:,i]* (Max_Val[i]+1e-6)
    renomal[:,i] = renomal[:,i]+ Min_Val[i]
    
print(renomal.cpu().detach().numpy())

renomal.size()