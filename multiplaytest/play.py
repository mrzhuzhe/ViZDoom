
from time import sleep
from torchbeast.core import environment
import torch

from sample_factory.utils.utils import AttrDict
from sample_factory.envs.doom.doom_utils import make_doom_env
from sample_factory.envs.tests.test_envs import default_doom_cfg
from torchbeast.monobeast import Net


def make_standard_dm(env_config):
    cfg = default_doom_cfg()
    cfg.env_frameskip = 2
    env = make_doom_env('doom_duel', cfg=cfg, env_config=env_config)
    env.skip_frames = cfg.env_frameskip
    return env


worker_index = 0
num_steps = 25201
env_config = AttrDict({'worker_index': worker_index, 'vector_index': 0, 'safe_init': False})
multi_env = make_standard_dm(env_config)

_info_len = 23
actor_model = Net(multi_env.observation_space['obs'].shape, multi_env.action_space, _info_len)

multi_env = environment.Environment(multi_env, device=torch.device("cpu"))        
env_output = multi_env.initial()

#obs = multi_env.reset()
visualize = True
P = 2


#_model_path_ = '/mnt/e28833eb-0c99-4fe2-802a-09fa58d9c9f5/code/ViZDoom/multiplaytest/runs/multi_player/test/model.tar'
_model_path_ = '/mnt/e28833eb-0c99-4fe2-802a-09fa58d9c9f5/code/ViZDoom/multiplaytest/runs/multi_player/test2/model.tar'

actor_model.load_state_dict(torch.load(
                _model_path_,
                map_location=torch.device("cpu")
            )["model_state_dict"])

actor_model.eval()
"""
for k in env_output.keys():
    print(env_output[k].shape)
    print(torch.index_select(env_output[k], 2, torch.tensor([0])).shape)
    print(torch.index_select(env_output[k], 2, torch.tensor([1])).shape)

"""
for i in range(num_steps):
    #actions = [multi_env.action_space.sample()] * len(obs)
    #actions = [multi_env.action_space.sample()] * P
    #inp1 = 
    #inp1 = torch.index_select(env_output, -3, 0)
    #inp2 = torch.index_select(env_output, -3, 1)
    
    
    agent_output1 = actor_model({ k: torch.index_select(v, 2, torch.tensor([0])) for k,v in env_output.items()})
    #agent_output2 = actor_model({ k: torch.index_select(v, 2, torch.tensor([1])) for k,v in env_output.items()})
    #print(agent_output1["action"])
    agent_output2 = torch.tensor([[[[ 0,  0,  0,  0,  0,  0, 10]]]])

    env_output = multi_env.step(torch.flatten(torch.concat([agent_output1["action"], agent_output2], dim=2), 0, 2))
    
    
    #env_output = multi_env.step(actions)
    
    obs = env_output["frame"]
    dones = env_output["done"]
    if visualize:
        multi_env.render()
    #sleep(0.04)

    #print(dones)
    #for key in env_output:
    #    if key == "episode_return" or key == "done":
    #        print(key, env_output[key])
    
    #if all(dones[0]):
    #    sleep(5)
    #    multi_env.reset()

multi_env.close() 