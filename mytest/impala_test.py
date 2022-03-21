import torch
import os
import logging
from torchbeast.monobeast import create_env, Net
from torchbeast.core import environment
from time import sleep

def test(flags, num_episodes: int = 10):
    
    checkpointpath = "/mnt/e28833eb-0c99-4fe2-802a-09fa58d9c9f5/code/ViZDoom/mytest/runs/reward_shaping_multiinp/torchbeast-20220318-223310/model.tar"

    gym_env = create_env(flags)
    env = environment.Environment(gym_env, torch.device("cpu"))
    model = Net(gym_env.observation_space.shape, gym_env.action_space.n, flags.use_lstm)
    model.eval()
    checkpoint = torch.load(checkpointpath)
    #, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    observation = env.initial()
    returns = []
    while len(returns) < num_episodes:
        agent_outputs = model(observation)
        policy_outputs, _ = agent_outputs
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
        sleep(0.1)
    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )


if __name__ == "__main__":

    class conf:
        def __init__(self):
            self.xpid = None
            self.use_lstm = False
            self.render = True

    flags = conf()
    test(flags)