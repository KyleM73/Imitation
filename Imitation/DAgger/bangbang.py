import torch

class Control(torch.nn.Module):
    def __init__(self,eps=0.01):
        super(Control, self).__init__()
        self.eps = eps
    def forward(self,x):
        #assumes double integrator, x [N,4]
        cond1 = torch.bitwise_and(x[:,2:] < 0,x[:,:2] < x[:,2:]**2/2)
        cond2 = torch.bitwise_and(x[:,2:] > 0,x[:,:2] < -x[:,2:]**2/2)
        conds = torch.bitwise_or(cond1,cond2)
        u = torch.where(conds,1.,-1.)
        uconds = torch.bitwise_and(torch.abs(x[:,:2]) < self.eps,torch.abs(x[:,2:]) < self.eps)
        u = torch.where(uconds,0.,u)
        return u #[N,2]

if __name__ == "__main__":
    from env import Env
    env = Env(1)
    expert = Control()
    x = env.reset()
    for i in range(5000):
        u = expert(x)
        x,done = env.step(u)
        if sum(done) != 0:
            print("done.")
            break
