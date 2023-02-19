import torch

from env import Env
from bangbang import Control
from policy import Policy
from DAgger import DAgger, beta, exp_decay, indicator

## DAgger params
beta_fn = exp_decay
N = 100 # epochs
T = 500 # iterations per epoch
num_envs = 10
plot = True
export = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # mps is worse in ~every possible way
print("Using device: {}".format(device))
print()

model = DAgger(
    Env=Env,Expert=Control,Policy=Policy,Beta=beta,Beta_fn=exp_decay,
    N=N,T=T,num_envs=num_envs,plot=plot,export=export,device=device
    )

