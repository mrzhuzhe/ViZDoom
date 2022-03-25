
from time import sleep
from torchbeast.core import environment
import torch

from sample_factory.utils.utils import AttrDict
from sample_factory.envs.doom.doom_utils import make_doom_env
from sample_factory.envs.tests.test_envs import default_doom_cfg

def make_standard_dm(env_config):
    cfg = default_doom_cfg()
    cfg.env_frameskip = 2
    env = make_doom_env('doom_duel', cfg=cfg, env_config=env_config)
    env.skip_frames = cfg.env_frameskip
    return env


worker_index = 0
num_steps = 1000
env_config = AttrDict({'worker_index': worker_index, 'vector_index': 0, 'safe_init': False})
multi_env = make_standard_dm(env_config)

multi_env = environment.Environment(multi_env, device=torch.device("cpu"))        
env_output = multi_env.initial()

#obs = multi_env.reset()
visualize = True
P = 2

for i in range(num_steps):
    #actions = [multi_env.action_space.sample()] * len(obs)
    actions = [multi_env.action_space.sample()] * P
    env_output = multi_env.step(torch.tensor(actions))
    #env_output = multi_env.step(actions)
    obs = env_output["frame"]
    dones = env_output["done"]
    if visualize:
        multi_env.render()

    #print(obs[0]["measurements"]) # obs is an array dict_keys(['obs', 'measurements'])  obs shape (3, 72, 128) 
    # ViZDoom/multiplaytest/sample_factory/envs/doom/wrappers/additional_input.py measurements 23 is info mation
    
    # print(infos) 49 info 
    #print(rew) # already has reward shaping 
    #print(dones)

    for key in env_output:
    #    print(key, env_output[key].shape)
        if key == "episode_return":
            print(key, env_output[key])
    
    #if all(dones[0]):
    #    multi_env.reset()

multi_env.close()



"""
a = TestDoom()
a.test_doom_multiagent_parallel()

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

"""            
