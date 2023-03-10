import matplotlib.pyplot as plt
import torch

def Plot(Env,expert,policy,max_iter=10_000,device="cpu"):
    env = Env(1,device=device)
    x = env.reset()
    env_star = Env(1)
    env_star.reset()
    env_star.x = x
    x_star = env_star.x
    assert torch.allclose(x,x_star)

    expert = expert.to(device)
    policy = policy.to(device)

    x_data = []
    u_data = []
    x_star_data = []
    u_star_data = []
    done = [0]
    done_star = [0]

    for _ in range(max_iter):
        if sum(done_star) == 0:
            u_star = expert(x_star)
            x_star_data.append(x_star[0,...].tolist())
            u_star_data.append(u_star[0,...].tolist())
            x_star,done_star = env_star.step(u_star)
        if sum(done) == 0:
            u = policy(x)
            x_data.append(x[0,...].tolist())
            u_data.append(u[0,...].tolist())
            x,done = env.step(u)
        if sum(done) != 0 and sum(done_star) != 0:
            print("Test Success.")
            break
    else:
        print("Test Failure.")
    print()

    fig,ax = plt.subplots()
    fig.suptitle("Trajectory")
    ax.plot([x[0] for x in x_star_data],[x[1] for x in x_star_data],label="expert")
    ax.plot([x[0] for x in x_data],[x[1] for x in x_data],label="policy")
    ax.set(xlim=(-1,1), ylim=(-1,1))
    ax.legend()
    fig.set_tight_layout(True)

    fig_v,ax_v = plt.subplots(2)
    fig_v.suptitle("Velocity")
    ax_v[0].get_xaxis().set_visible(False)
    ax_v[0].plot([x[2] for x in x_star_data],label="vx expert")
    ax_v[0].plot([x[2] for x in x_data],label="vx policy")
    ax_v[0].set(ylim=(-1,1))
    ax_v[0].legend()
    ax_v[1].plot([x[3] for x in x_star_data],label="vy expert")
    ax_v[1].plot([x[3] for x in x_data],label="vy policy")
    ax_v[1].set(ylim=(-1,1))
    ax_v[1].legend()
    fig_v.set_tight_layout(True)


    fig_u,ax_u = plt.subplots(2)
    fig_u.suptitle("Control")
    ax_u[0].get_xaxis().set_visible(False)
    ax_u[0].plot([u[0] for u in u_star_data],label="ux expert")
    ax_u[0].plot([u[0] for u in u_data],label="ux policy")
    ax_u[0].legend()
    ax_u[1].plot([u[1] for u in u_star_data],label="uy expert")
    ax_u[1].plot([u[1] for u in u_data],label="uy policy")
    ax_u[1].legend()
    fig_u.set_tight_layout(True)

    plt.show()

if __name__ == '__main__':
    from env import Env
    from bangbang import Control
    from policy import Policy
    expert = Control()
    net = Policy()
    plot(Env,expert,net)
