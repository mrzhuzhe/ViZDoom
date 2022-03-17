from setup_env import MyDoom
from time import sleep
from torchbeast.core import environment
import torch

_env = MyDoom(render=True)

total_steps = 100

state = _env.reset()


_env = environment.Environment(_env, device=torch.device("cpu"))        
env_output = _env.initial()

for step in range(total_steps):    
    #state, reward, done, info = _env.step(_env.gym_env.action_space.sample())
    #print(reward, done, info)
    env_output = _env.step(torch.tensor(_env.gym_env.action_space.sample()))
    for key in env_output:
        if key != "frame":
            print(key, env_output[key])
    sleep(0.2)
_env.close()