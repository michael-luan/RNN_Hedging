# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:17:04 2022

@author: micha
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Simulator:
    
    def __init__(self, T=0.25, NdT=90, S0=10, kappa=2, theta=np.log(10), sigma=0.4):
        
        self.T = T
        self.NdT = NdT
        self.t = np.linspace(0,T,self.NdT+1)
        self.dt = self.t[1]-self.t[0]
        
        self.S0 = S0
        
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        
        self.theta_X = self.theta - 0.5*self.sigma**2/self.kappa
        
        
    def Sim(self, batch_size = 1024):
        
        X = np.zeros((batch_size, self.NdT+1))
        X[:,0] = np.log(self.S0)
        
        for n in range(self.NdT):
            
            X[:,n+1] = self.theta_X + np.exp(-self.kappa*self.dt) * (X[:,n] - self.theta_X)
            X[:,n+1] += self.sigma*np.sqrt(self.dt)* np.random.randn(batch_size)
            
        S = np.exp(X)
        
        return S

model = Simulator()
S = model.Sim(10_000)


plt.plot(model.t, S[:500,:].T, alpha=0.1, linewidth=1)

lvls = [0.1,0.5,0.9]
qtl = np.quantile(S, lvls, axis=0)
for i, lvl in enumerate(lvls):
    plt.plot(model.t, qtl[i,:], linewidth=1, label=r'$q_{' + str(lvls[i]) + '}$')
    
plt.plot(model.t, S[0,:], color='k', linewidth=1)
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$S_t$', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.ylim(5,15)
plt.show()
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
from GOU_Simulator import Simulator
from scipy.stats import norm


# option contract parameters
T = 1/4
K1 = 9.5
K2 = 10.5  

# market environment parameters
sigma = 0.4
r = 0.02
mu = 0.05
kappa = 2
theta = np.log(10)
S0 = 10
trans_fee = 0.005

# trading parameters
NdT = 90
t = np.linspace(0,T,NdT)
dt = t[1]-t[0]  


# what is bull spread, why/when we use
#tau = T-t
def CallPrice(S,K,tau,sigma,r):
    dp = (np.log(S/K)+(r+0.5*sigma**2)*tau)/(np.sqrt(tau)*sigma)
    dm = (np.log(S/K)+(r-0.5*sigma**2)*tau)/(np.sqrt(tau)*sigma)
    
    return S*norm.cdf(dp) - K*np.exp(-r*tau)*norm.cdf(dm) 


def ShortBullSpreadPrice(S,K1,K2,tau,sigma,r):
    C1 = CallPrice(S, K1,tau, sigma, r)
    C2 = CallPrice(S, K2, tau, sigma, r)
    return (C1-C2)


C0=ShortBullSpreadPrice(S0,K1,K2,T,sigma,r)
print(ShortBullSpreadPrice(S0,K1,K2,T,sigma,r)) 

def CallDelta(S,K,tau,sigma,r):
    tau+=1e-10
    dp = (np.log(S/K)+(r+0.5*sigma**2)*tau)/(np.sqrt(tau)*sigma)
    
    return norm.cdf(dp)

def ShortBullSpreadDelta(S,K1,K2,tau,sigma,r):
    D1 = CallDelta(S, K1, tau, sigma, r)
    D2 = CallDelta(S, K2, tau, sigma, r)
    return (D1-D2)

print(ShortBullSpreadDelta(S0,K1,K2,T,sigma,r))

class GruNet(nn.Module):
    
    def __init__(self, ni, hs, nl, nodes, layers):
        super(GruNet, self).__init__()
        
        self.ni = ni
        
        # GRU layer
        self.GRU = nn.GRU(input_size=ni, hidden_size=hs, num_layers=nl)
        
        # GRU to FFN

        self.prop_in_to_h = nn.Linear(ni + 1, nodes)

        self.prop_h_to_h = []
        for i in range(layers-1):
            self.prop_h_to_h.append(nn.Linear(nodes, nodes))

        # FFN to output
        self.prop_h_to_out = nn.Linear(nodes, 1)

    def forward(self, x):
        
        # input into  GRU layer (seq, batch, feature)
        # h_all, h_last = self.GRU(x)
        # sequence is time
        # batch: S0,S1,...S10
        _, h_last = self.GRU(x)
        catted = torch.cat((x[-1, :, :].unsqueeze(dim = 0), h_last), dim = 2)
        
        h = torch.sigmoid(self.prop_in_to_h(catted))

        for layer in self.prop_h_to_h:
            h = torch.sigmoid(layer(h))

        y = self.prop_h_to_out(h)
        
        return y

    def parameters(self):
        params = list(self.GRU.parameters())
        params += list(self.prop_in_to_h.parameters())
        #params.requires_grad = True
        for prop in self.prop_h_to_h:
            params += list(prop.parameters())
            
            
        params += list(self.prop_h_to_out.parameters())
        
        return params

def RunHedge(S, alpha):
   
    bank = - alpha[:,0]*S[:,0] - abs(alpha[:,0])*TC
    
    for i in range(NdT): #0 to 89, alpha 1 to 90
        
        # accumulate bank account to next time step
        bank *= np.exp(r*dt)
        
        # rebalance the position
        bank -= ((alpha[:,i+1]-alpha[:,i]) * S[:,i+1] + abs(alpha[:,i+1]-alpha[:,i])*TC)

                
    # liquidate terminal assets, and pay what you owe from the contract
    bank += alpha[:,-1]*S[:,-1] -(S[:,-1]-K1)*(S[:,-1]>K1)  + (S[:,-1]-K2)*(S[:,-1]>K2)
        
    return bank

def Hedge(S, alpha):
    
    C0 = ShortBullSpreadDelta(S0, K1, K2, T, sigma, r)
    transaction = abs(alpha[:, 0 , 0]) * TC
    # start the bank account with value of contract and purchasing initial shares
    
    
    bank = C0 - alpha[:,0,0]*(S[:,0]) - transaction
    
    for i in range(NdT-1):
        
        # accumulate bank account to next time step
        bank *= np.exp(r*dt)
        
        # rebalance the position
        transaction = abs(alpha[:, i + 1, 0] - alpha[:, i, 0]) * TC
        bank -= (alpha[:,i+1,0]-alpha[:,i,0]) * S[:,i+1] + transaction
        
    # liquidate terminal assets, and pay what you owe from the contract
    # here, we short the call at K1 and we long the call at K2
    bank += alpha[:,-1,0]*S[:,-1] - ((S[:,-1]-K1)*(S[:,-1]>K1) - (S[:,-1]-K2)*(S[:,-1]>K2))
    
    return bank

def Sim(net,nsims,lag):
    S = model.Sim(nsims)
    S1 = torch.tensor(S)
    delta = np.zeros((nsims, 92+lag)) #included t=-1 in the begining
    delta[:,1+lag] = CallDelta(S0, K1, T, sigma, r) - CallDelta(S0, K2, T, sigma, r)
    M = np.zeros((nsims, 91+lag))
    M[:,lag] = -delta[0,1]*S[0,0]- abs(delta[0,1]) * TC
    deltaO = np.zeros((nsims,lag+1))
    delta_ = torch.tensor(deltaO)
    for i in range(1, NdT-lag): #alpha 1 to 90
        #update M_i, delta_i+1

        S_current = np.reshape(S[:,i-1:i+lag],(lag+1,nsims))

        delta_prev = np.reshape(delta[:,i+lag:i+2*lag+1],(lag+1,nsims))

        M_current = np.reshape(M[:,i+lag-1: i+2*lag],(lag+1,nsims))

        x = torch.zeros((lag + 1, nsims,3))
        x[:,:,0] = torch.tensor(S_current,requires_grad=True) # asset price
        x[:,:,1] = torch.tensor(M_current,requires_grad=True)
        x[:,:,2] = torch.tensor(delta_prev,requires_grad=True)
        

        #GRU input=3,hidden state=2, hiden layer=1, FFN 16 nodes, 5 layers
        
        delta_current_net = net(x/10000)
        delta_current_net =  delta_current_net[0,:]
        delta_current = delta_current_net.detach().numpy().reshape(nsims)   
        delta_ = torch.cat((delta_,delta_current_net), 1)
        
        # Note I am updating current delta, but my current delta is at Delta[t+1]
        delta[:, i+1] = delta_current
        M[:,i] *= np.exp(r*dt)
        M[:,i] = M[:,i-1] - (delta[:, i+1]-delta[:, i]) * S[:,i] + abs(delta[:, i+1]-delta[:, i]) * TC
#         delta_2 = np.delete(delta, 0, axis=1)
#         delta_2 = np.array(delta_2)
        #print(delta_.shape)
    if lag != 0:
        S_current = np.reshape(S[:,90 - lag:90+lag],(lag+1,nsims))
        M_current = np.reshape(M[:,89:90+lag],(lag+1,nsims))
        delta_X = np.reshape(delta[:,89:90+lag],(lag+1,nsims))
        x2 = torch.zeros((lag + 1, nsims,3))
        x2[:,:,0] = torch.tensor(2*S_current/S0-1,requires_grad=True) # asset price
        x2[:,:,1] = torch.tensor(M_current,requires_grad=True)
        x2[:,:,2] = torch.tensor(delta_X,requires_grad=True)
    else:
        S_current = np.reshape(S[:,89 - lag:90+lag],(lag+1,nsims))
        M_current = np.reshape(M[:,89:90+lag],(lag+1,nsims))
        delta_X = np.reshape(delta[:,89:90+lag],(lag+1,nsims))
        x2 = torch.zeros((lag + 1, nsims,3))
        x2[:,:,0] = torch.tensor(2*S_current/S0-1,requires_grad=True) # asset price
        x2[:,:,1] = torch.tensor(M_current,requires_grad=True)
        x2[:,:,2] = torch.tensor(delta_X,requires_grad=True)
   # print(delta_)
    alpha = net(x2/10000)
    #print(alpha[0,:])
    delta_ = torch.cat((delta_,alpha[0,:]), 1)
    bank = RunHedge(S1, delta_)
        #delta_ = torch.tensor(delta_)
        #print(delta_current_net.type())
    #bank = bank[0,:]
    return bank



def Plot_PnL(loss_hist, net, name=""):
    plt.figure(figsize=(10,6))
    plt.suptitle('RNN Loss History And PnL Distribution of Terminal Strategy',fontsize = 16, y = 1.1)
    plt.subplot(1,2,1)
    plt.plot(loss_hist, color = "red")
    plt.xlabel('iteration',fontsize=16)
    plt.ylabel('loss',fontsize=16)

    plt.subplot(1,2,2)
    PnL = Sim(net, 10000,1)
    PnL_BS = sim_valuation(10_000)
    plt.hist(PnL_BS.detach().numpy(), bins=np.linspace(-1,1,51), alpha=0.6,color="skyblue", ec="blue",label="Black-Scholes")
    plt.hist(PnL.detach().numpy(), bins=np.linspace(-1,1,51),color="orange", ec="black", alpha=0.5, label="Network Prediction")
    # calculate 10 quantile threshold
    q10_threshold = numpy.quantile(PnL.detach().numpy(), 0.1)
    q10_mean = numpy.average(PnL.detach().numpy(), weights=(PnL.detach().numpy() <= q10_threshold))
    plt.axvline(x=q10_mean, label=r'CVaR={:.3f}'.format(q10_mean), c='r', alpha=0.7) 

    
    plt.xlabel('P&L',fontsize=16)
    plt.ylabel('Count',fontsize=12)
    plt.ylim(0,1800)
    plt.legend()
    
    plt.tight_layout(pad=2)    
    plt.show()  
    
def FitNet(net, name=""):
    mini_batch_size = 100
    

    # create  optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    #CrossEntropyLoss = nn.CrossEntropyLoss()

    Nepochs = 1000
    loss_hist = []
    
    if not name == "":
        Plot_PnL(loss_hist, net, name + "_dist_0")  
    else:
        Plot_PnL(loss_hist, net)
    
#     torch.autograd.set_detect_anomaly(True)

    for epoch in tqdm(range(Nepochs)):  # loop over the dataset multiple times


        # grab a mini-batch from simulations
        PnL = Sim(net, mini_batch_size, 2)
        #PnL = torch.tensor(PnL)
        #PnL = Variable(torch.tensor(PnL, requires_grad = True))
        # zero the parameter gradients
        optimizer.zero_grad()
        
        #loss function
        q = torch.tensor([0.1])
        VaR_10 = np.quantile(PnL.detach().numpy(),q)
        CVaR_10 = PnL[PnL.detach().numpy()<=VaR_10].mean()
        loss = -CVaR_10
        # propogate the sensitivity of the output to the model parameters 
        # backwards through the computational graph
        #loss.requires_grad_(True)
    
        loss.backward()
        

        # update the weights and biases by taking a SGD step
        optimizer.step()

        # store running loss
        loss_hist.append(  loss.item() )

        # plot output every 200 iterations
        if( ( (epoch) % 100 == 0) and (epoch>10) ):
            print(epoch)
            if not name == "":
                Plot_PnL(loss_hist, net, name + "_dist" + str(int(epoch/200)).zfill(2) )
           
            else:
                Plot_PnL(loss_hist, net)
  



    print(epoch)
    Plot_PnL(loss_hist, net)

    print('Finished Training')
    
    return loss_hist

def CallPrice(S,K,tau,sigma,r):
    dp = (np.log(S/K)+(r+0.5*sigma**2)*tau)/(np.sqrt(tau)*sigma)
    dm = (np.log(S/K)+(r-0.5*sigma**2)*tau)/(np.sqrt(tau)*sigma)
    
    return S*norm.cdf(dp) - K*np.exp(-r*tau)*norm.cdf(dm)

def bullspreadPrice(S, K1, K2, tau, sigma, r):
    return CallPrice(S, K1, tau, sigma, r) - CallPrice(S, K2, tau, sigma, r)

###############################################################################

def CallDelta(S,K,tau,sigma,r):
    tau+=1e-10

    dp = (np.log(S/K)+(r+0.5*sigma**2)*tau)/(np.sqrt(tau)*sigma)
    
    return norm.cdf(dp)

def bullspreadDelta(S, K1, K2, tau, sigma, r):
    return CallDelta(S, K1, tau, sigma, r) - CallDelta(S, K2, tau, sigma, r)

###############################################################################

def Hedge(S, alpha):
    print(alpha.shape)
    C0 = bullspreadPrice(S0, K1, K2, T, sigma, r)
    transaction = abs(alpha[:, 0 , 0]) * TC
    # start the bank account with value of contract and purchasing initial shares
    
    
    bank = C0 - alpha[:,0,0]*(S[:,0]) - transaction
    
    for i in range(NdT-1):
        
        # accumulate bank account to next time step
        bank *= np.exp(r*dt)
        
        # rebalance the position
        transaction = abs(alpha[:, i + 1, 0] - alpha[:, i, 0]) * TC
        bank -= (alpha[:,i+1,0]-alpha[:,i,0]) * S[:,i+1] + transaction
        
    # liquidate terminal assets, and pay what you owe from the contract
    # here, we short the call at K1 and we long the call at K2
    bank += alpha[:,-1,0]*S[:,-1] - ((S[:,-1]-K1)*(S[:,-1]>K1) - (S[:,-1]-K2)*(S[:,-1]>K2))
    
    return bank

###############################################################################

def HedgeRNN(S, alpha):

    C0 = bullspreadPrice(S0, K1, K2, T, sigma, r)
    transaction = abs(alpha[:, 0 ]) * trans_fee
    # start the bank account with value of contract and purchasing initial shares
    
    
    bank = C0 - alpha[:,0]*(S[:,0]) - transaction
    
    for i in range(NdT-1):
        
        # accumulate bank account to next time step
        bank *= np.exp(r*dt)
        
        # rebalance the position
        transaction = abs(alpha[:, i + 1, 0] - alpha[:, i, 0]) * trans_fee
        bank -= (alpha[:,i+1,0]-alpha[:,i,0]) * S[:,i+1] + transaction
        
    # liquidate terminal assets, and pay what you owe from the contract
    # here, we short the call at K1 and we long the call at K2
    bank += alpha[:,-1,0]*S[:,-1] - ((S[:,-1]-K1)*(S[:,-1]>K1) - (S[:,-1]-K2)*(S[:,-1]>K2))
    
    return bank


###############################################################################

def sim_valuation(nsims = 10_000):
    
    model = Simulator()
    S = model.Sim(10_000)
    S = torch.tensor(S)
    # print("T", (T - np.matlib.repmat(t, nsims, 1)).shape)
    alpha_BS = torch.unsqueeze(
        torch.tensor(bullspreadDelta(S.detach().numpy(), K1, K2, T - np.matlib.repmat(t, nsims, 1), sigma, r)), dim=2)
    bank_BS = Hedge(S, alpha_BS)
    return bank_BS



PnL_BS = sim_valuation()

plt.hist(PnL_BS.detach().numpy(),bins = 50,  alpha=0.6,color="skyblue", ec="blue")
plt.xlabel('P&L',fontsize=16)
plt.ylabel('Freq.',fontsize=16)
plt.title("Black-Scholes Hedging Strategy P&L", fontsize = 16)


def CVaR(sample, quantile):
    # requires sample of a dist and the percent quantile out of 1
    quantile = torch.tensor([quantile])
    percent = np.quantile(sample.detach().numpy(), quantile)
    mask = sample.detach().numpy() <= percent
    cvar = sample[mask].mean()
    return cvar


#%%

np.random.seed(10928391)
net = GruNet(3, 1, 1, 6, 16)
loss_hist = FitNet(net, "net, 16 layers of 6")