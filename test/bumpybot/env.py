import pybullet as p
from pybullet import getEulerFromQuaternion as Q2E
import pybullet_utils.bullet_client as bc
import pybullet_data


class env:
    def __init__(self, headless=False):
        if headless:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = bc.BulletClient(connection_mode=p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        raise NotImplementedError

    def reset(self):
        self.setup()
        obs = self.get_obs()
        return obs

    def setup(self):
        ## Initiate simulation
        self.client.resetSimulation()

        ## Set up simulation
        self.client.setTimeStep(0.01)
        self.client.setPhysicsEngineParameter(numSolverIterations=int(30))
        self.client.setPhysicsEngineParameter(enableConeFriction=0)
        self.client.setGravity(0,0,-9.8)

        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_rewards(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        p.disconnect()




