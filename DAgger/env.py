import torch

def rk4(func,t,x,u,dt):
    k1 = func(t,x,u)
    k2 = func(t,x+dt*k1/2,u)
    k3 = func(t,x+dt*k2/2,u)
    k4 = func(t,x+dt*k3,u)
    return x+(k1+2*k2+2*k3+k4)*dt/6

class Env:
    def __init__(self,n=1,device="cpu"):
        self.m = 1
        self.u_ = 1 #action constraint
        self.dt = 0.01
        self.eps = 0.1
        self.n = n
        self.device = device
        self.A = torch.tensor([
            [0,0,1,0],
            [0,0,0,1],
            [0,0,0,0],
            [0,0,0,0]
            ],dtype=torch.float,device=self.device)
        self.B = torch.tensor([
            [0,0],
            [0,0],
            [1,0],
            [0,1]
            ],dtype=torch.float,device=self.device)
        self.x_dim = 4
        self.u_dim = 2

    def reset(self):
        rnd_pos = 2*torch.rand(self.n,2,device=self.device)-1 #[-10,10)
        rnd_vel = 2*torch.rand(self.n,2,device=self.device)-1 #[-1,1)
        self.x = torch.cat((rnd_pos,rnd_vel),dim=1)
        self.t = torch.zeros(self.n,device=self.device)
        return self.x

    def reset_idx(self,idx):
        rnd_pos = 2*torch.rand(len(idx),2,device=self.device)-1 #[-10,10)
        rnd_vel = 2*torch.rand(len(idx),2,device=self.device)-1 #[-1,1)
        self.x[idx,...] = torch.cat((rnd_pos,rnd_vel),dim=1)
        self.t[idx] = 0
        return self.x

    def step(self,action):
        self.u = torch.clamp(action,-self.u_,self.u_).to(self.device)
        self.x = rk4(self.dynamics,self.t,self.x,self.u,self.dt)

        dones = torch.flatten(torch.where(torch.linalg.norm(self.x,dim=1)<self.eps,1,0))
        env_id = torch.argwhere(torch.linalg.norm(self.x,dim=1)<self.eps)[:,0]

        self.reset_idx(env_id)

        return self.x,dones

    def norm(self,x,dim=1):
        #mps doesnt have support for torch.linalg.norm()
        #this silution only partially works, as mps doesnt support torch.argwhere() either
        return torch.sum(torch.square(x),dim=dim).pow(0.5)

    def dynamics(self,t,x,u):
        return torch.einsum("ij,jkl->ikl",self.A,x.view(self.x_dim,1,-1)).view(-1,self.x_dim) + torch.einsum("ij,jkl->ikl",self.B,u.view(self.u_dim,1,-1)).view(-1,self.x_dim) #batch matrix multiply

if __name__ == "__main__":
    env = Env(10)
    env.reset()

    act = torch.zeros(10,2)
    env.step(act)

