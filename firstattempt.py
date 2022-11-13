from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import os
import ast
import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image as PImage
device="cpu"
class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(64, z_dim)
        self.fc2 = nn.Linear(64, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Tanh(),
        )

        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)

        self.encoder.apply(self.weights_init)
        self.decoder.apply(self.weights_init)

    @staticmethod
    def weights_init(layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_uniform_(layer.weight)

    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        mu, std, esp = mu.to(device), std.to(device), esp.to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20,10),
            )
    def forward(self,x):
        output=self.linear_relu_stack(x)
        return output
def coords(m):
    for i in range(2, len(m) - 3):
        if m[i] == ',':
            return m[2:i],m[(i+2):(len(m)-2)]
model = VAE(z_dim=20)
model.load_state_dict(torch.load("20_lat_1_beta.torch", map_location=torch.device('cpu')))
model2=NeuralNetwork()
optimizer = torch.optim.SGD(model2.parameters(), 1e-3)
with open('x_y.txt', 'r') as f: 
    lines = f.readlines()
epoch=0
for line in lines:
        epoch=epoch+1
        x, y = line.split('=')[0], line.split('=')[1]
        a=line.split('=')
        joint1=a[2]
        joint2=a[3]
        joint3=a[4]
        joint4=a[5]
        joint5=a[6]
        x1,y1=coords(joint1)
        x2,y2=coords(joint2)
        x3,y3=coords(joint3)
        x4,y4=coords(joint4)
        x5,y5=coords(joint5)
        x, y = x.split(' '), y.split(' ')
        x = [i for i in x if i]
        y = [i for i in y if i]
        x[0] = x[0][1:]
        y[0] = y[0][1:]
        x[-1] = x[-1][:-1]
        y[-1] = y[-1][:-1]

        x = [float(i) for i in x if i]
        y = [float(i) for i in y if i]
        
        S=np.zeros(360, dtype='complex_')
        i=0
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=359
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=1
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=358
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=2
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        input_list=[float(np.real(S[358])), float(np.real(S[359])), float(np.real(S[0])), float(np.real(S[1])),float(np.real(S[2])), float(np.imag(S[358])), float(np.imag(S[359])), float(np.imag(S[0])), float(np.imag(S[1])), float(np.imag(S[2]))]
        input_tensor=torch.tensor(input_list)
        latent_vector=model.encode(input_tensor)
        prediction=model2(latent_vector[0])
        output_list=[float(x1),float(x2),float(x3),float(x4),float(x5),float(y1),float(y2),float(y3),float(y4),float(y5)]
        output_tensor=torch.tensor(output_list)
        loss_function=nn.MSELoss()
        loss=loss_function(prediction,output_tensor)
        if(epoch<10000):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        elif(epoch>=10000):
            print("NEW EPOCH")
            print("Joint locations (first predicted followed by actual)")
            print("Predicted Joint 1: "+str(prediction[0].item())+", "+str(prediction[5].item()))
            print("Actual Joint 1: "+str(float(x1))+", "+str(float(y1)))
            print("Predicted Joint 2: "+str(prediction[1].item())+", "+str(prediction[6].item()))
            print("Actual Joint 2: "+str(float(x2))+", "+str(float(y2)))
            print("Predicted Joint 3: "+str(prediction[2].item())+", "+str(prediction[7].item()))
            print("Actual Joint 3: "+str(float(x3))+", "+str(float(y3)))
            print("Predicted Joint 4: "+str(prediction[3].item())+", "+str(prediction[8].item()))
            print("Actual Joint 4: "+str(float(x4))+", "+str(float(y4)))
            print("Predicted Joint 5: "+str(prediction[4].item())+", "+str(prediction[9].item()))
            print("Actual Joint 5: "+str(float(x5))+", "+str(float(y5)))
            print(f"loss: {loss:>7f}")
