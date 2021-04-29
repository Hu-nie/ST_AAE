import torch

class NetD(torch.nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.fc1 = torch.nn.Linear(228*2, 228)
        self.fc2 = torch.nn.Linear(228, 228)
        self.fc3 = torch.nn.Linear(228, 228)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()
    
    
    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]
        
        
    def forward(self, x, m, g, h):
        """Eq(3)"""
        inp = m * x + (1-m) * g 
        inp = torch.cat((inp, h), dim=1)
        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
#         out = self.sigmoid(self.fc3(out)) # [0,1] Probability Output
        out = self.fc3(out)
        
        return out    

class NetG(torch.nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        self.fc1 = torch.nn.Linear(228*2, 228)
        self.fc2 = torch.nn.Linear(228, 228)
        self.fc3 = torch.nn.Linear(228, 228)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()
    
    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]
        
        
    def forward(self, x, z, m):
        inp = m * x + (1-m) * z
        inp = torch.cat((inp, m), dim=1)
        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out)) # [0,1] Probability Output
#         out = self.fc3(out)
        
        return out 