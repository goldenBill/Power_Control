import torch
import torch.nn as nn
import numpy as np

class Approx(nn.Module):
    def __init__(self, inp=4, oup=2, hidden_dim=20):
        super(Approx, self).__init__()
        self.inp = inp
        self.oup = oup
        self.hidden_dim = hidden_dim
        self.architecture = nn.Sequential(
            nn.Linear(self.inp, self.hidden_dim, bias = True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias = True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.oup, bias = True),
            nn.BatchNorm1d(self.oup),
            nn.ReLU(inplace = True))
        
    def forward(self, h, g):
        channel_info = torch.cat((h, g), dim = 1)
        return self.architecture(channel_info)
    
class Dual(nn.Module):
    def __init__(self, inp=2, oup=1, Gamma = 1, P = 1):
        super(Dual, self).__init__()
        self.inp = inp + 1
        self.oup = oup
        self.Gamma = Gamma
        self.P = P
        self.dual_variable = nn.Sequential(
            nn.Linear(self.inp, self.oup, bias = False)
            )
        
    def forward(self, approx, g):
        coeff = torch.cat((power_constrain(approx, self.P), IT_constrain(approx, g, self.Gamma).reshape(-1, 1)), dim = 1)
        return self.dual_variable(coeff)
    
class Lagrange(nn.Module):
    def __init__(self):
        super(Lagrange, self).__init__()
        
    def forward(self, approx, dual, h):
        result = -objective(approx, h) + dual
        return torch.mean(result)
    
def objective(x, h):
    return torch.log(1 + torch.sum(x*h, dim = 1))
    
def power_constrain(x, P):
    return (x - P)
    
def IT_constrain(x, g, Gamma):
    return torch.sum(x*g, dim = 1) - Gamma
    
    
    
if __name__ == "__main__":
    N = 2
    P = 1
    model_Approx = Approx(inp = 2*N, oup = N, hidden_dim = 10*N)
    model_Dual = Dual(inp = N, oup = 1, Gamma = 1, P=torch.ones(N)*P)
    
    h_data = torch.zeros(10, N)
    h_data.exponential_(lambd=1)
    g_data = torch.zeros(10, N)
    g_data.exponential_(lambd=1)
    
    output_Approx = model_Approx(h_data, g_data)
    print(output_Approx.size())
    output_Dual = model_Dual(output_Approx, g_data)
    print(output_Dual.size())