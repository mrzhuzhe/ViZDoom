import torch

from torchbeast.monobeast import AtariNet, create_env
from impala_train import conf
from torchbeast.core import environment
import copy
from torchbeast.action_distributions import get_action_distribution

Net = AtariNet


#fameInput = torch.rand((1, 1, 2, 3, 128, 72))
#InfoInput = torch.rand((1, 1, 2, 23))
#rewardInput = doneInput = lastActionInput = torch.rand((1, 2, 1))

flags = conf()
env = create_env(flags)
_info_len = 23

#print(env.observation_space['obs'].shape[0], env.action_space)

actor_model = Net(env.observation_space['obs'].shape, env.action_space, _info_len)


env = environment.Environment(env, device=torch.device("cpu") )        
env_output = env.initial()

#env_output = { k: torch.flatten(val, 0, 1) for k, val in env_output.items()}
#env_output['frame'] = torch.flatten(env_output['frame'], 1, 2)
#env_output['info'] = torch.flatten(env_output['info'], 1, 2)
print(env_output['frame'].shape)

#agent_state = actor_model.initial_state(batch_size=1)
agent_output = actor_model(env_output)

#print(agent_output)
for k in agent_output:
    print(k, agent_output[k].shape)

behavior_policy_action_distribution = get_action_distribution(actor_model.action_space, torch.flatten(agent_output["policy_logits"], 0, 2))
#behavior_policy_action_distribution = get_action_distribution(actor_model.action_space, agent_output["policy_logits"])
print(behavior_policy_action_distribution.entropy())