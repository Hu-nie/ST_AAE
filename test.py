import torch
import numpy as np
# from tqdm import tqdm
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from utils import *

dataset_file = 'data/V_228.csv'
use_gpu = True  # set it to True to use GPU and False to use CPU
if use_gpu:
    torch.cuda.set_device(1)

# System Parameters
# 1. Mini batch size
mb_size = 128
# 2. Missing rate
p_miss = 0.2
# 3. Hint rate
p_hint = 0.8
# 4. Loss Hyperparameters
alpha = 10
# 5. Train Rate
train_rate = 0.8

# Data

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


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(456, 228, bias=True)
        self.fc2 = nn.Linear(228, 228, bias=True)
        self.fc3 = nn.Linear(228, 228, bias=True)
        self.ReLU = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

        # nn.init.zeros_(self.fc.bias)
        # nn.init.zeros_(self.fc2.bias)
        # nn.init.zeros_(self.fc3.bias)

        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self,  New_X, M):
        inputs = torch.cat(dim=1, tensors=[New_X, M])
        inputs = inputs.float()
        outputs = self.ReLU(self.fc(inputs))
        outputs = self.ReLU(self.fc2(outputs))
        return self.sigmoid(self.fc3(outputs))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(456, 228, bias=True)
        self.fc2 = nn.Linear(228, 228, bias=True)
        self.fc3 = nn.Linear(228, 228, bias=True)
        self.ReLU = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

        # nn.init.zeros_(self.fc.bias)
        # nn.init.zeros_(self.fc2.bias)
        # nn.init.zeros_(self.fc3.bias)

        # nn.init.xavier_uniform_(self.fc.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self,New_X, H):
        inputs = torch.cat(dim=1, tensors=[New_X, H])
        inputs = inputs.float()
        outputs = self.ReLU(self.fc(inputs))
        outputs = self.ReLU(self.fc2(outputs))
        return self.sigmoid(self.fc3(outputs))


generator = Generator()
discriminator = Discriminator()



def discriminator_loss(M, New_X, H):
    # Generator
    G_sample = generator(New_X,M)
    # Combine with original data
    Hat_New_X = New_X * M + G_sample * (1-M)

    # Discriminator
    D_prob = discriminator(Hat_New_X, H)

    #%% Loss
    D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1-M) * torch.log(1. - D_prob + 1e-8))
    return D_loss

def generator_loss(X, M, New_X, H):
    #%% Structure
    # Generator
    G_sample = generator(New_X,M)

    # Combine with original data
    Hat_New_X = New_X * M + G_sample * (1-M)

    # Discriminator
    D_prob = discriminator(Hat_New_X, H)

    #%% Loss
    G_loss1 = -torch.mean((1-M) * torch.log(D_prob + 1e-8))
    MSE_train_loss = torch.mean((M * New_X - M * G_sample)**2) / torch.mean(M)

    G_loss = G_loss1 + alpha * MSE_train_loss 

    #%% MSE Performance metric
    MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
    return G_loss, MSE_train_loss, MSE_test_loss
    
def test_loss(X, M, New_X):
    #%% Structure
    # Generator
    G_sample = generator(New_X,M)

    #%% MSE Performance metric
    MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
    return MSE_test_loss, G_sample

# optimizer_D = torch.optim.Adam(params=theta_D)
# optimizer_G = torch.optim.Adam(params=theta_G)

optimizer_D = torch.optim.Adam(params=generator.parameters())
optimizer_G = torch.optim.Adam(params=discriminator.parameters())

# Start Iterations
for it in tqdm(range(5000)):    
    
    #%% Inputs
    mb_idx = sample_idx(Train_No, mb_size)
    X_mb = trainX[mb_idx,:]  
    
    Z_mb = sample_Z(mb_size, Dim) 
    M_mb = trainM[mb_idx,:]  
    H_mb1 = sample_M(mb_size, Dim, 1-p_hint)
    H_mb = M_mb * H_mb1
    
    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
    

    X_mb = torch.tensor(X_mb)
    M_mb = torch.tensor(M_mb)
    H_mb = torch.tensor(H_mb)
    New_X_mb = torch.tensor(New_X_mb)
    
    
    optimizer_D.zero_grad()
    D_loss_curr = discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
    D_loss_curr.backward()
    optimizer_D.step()
 
    optimizer_G.zero_grad()
    #print(M_mb.shape)
    G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
    G_loss_curr.backward()
    optimizer_G.step()    
        
    #%% Intermediate Losses
    if it % 100 == 0:
        print('Iter: {}'.format(it),end='\t')
        print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())),end='\t')
        print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())),end='\t')
        print('G_loss: {:.4}'.format(D_loss_curr),end='\t')
        print('D_loss: {:.4}'.format(G_loss_curr))


Z_mb = sample_Z(Test_No, Dim) 
M_mb = testM
X_mb = testX
        
New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

X_mb = torch.tensor(X_mb)
M_mb = torch.tensor(M_mb)
New_X_mb = torch.tensor(New_X_mb)

    
MSE_final, Sample = test_loss(X=X_mb, M=M_mb, New_X=New_X_mb)
        
print('Final Test RMSE: ' + str(np.sqrt(MSE_final.item())))


imputed_data = M_mb * X_mb + (1-M_mb) * Sample
print("Imputed test data:")
np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})
print(imputed_data.detach().numpy())

# Normalization (0 to 1)
renomal = imputed_data 

for i in range(Dim):
    renomal[:,i] = renomal[:,i]* (Max_Val[i]+1e-6)
    renomal[:,i] = renomal[:,i]+ Min_Val[i]
    
print(renomal.cpu().detach().numpy())

