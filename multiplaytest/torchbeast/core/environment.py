# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The environment class for MonoBeast."""
import torch
import numpy as np

def _format_frame(frame, device):
    # neeed to be gray scale
    frame = torch.from_numpy(frame).to(device)
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).

P = 2
#_num_action_ = 7
class Environment:
    def __init__(self, gym_env, device):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self.device = device
        self.action_space = gym_env.action_space

    def render(self):
        self.gym_env.render()

    def initial(self):
        initial_reward = torch.zeros(1, 1,  P,  device=self.device)
        # This supports only single-tensor actions ATM.
        #initial_last_action = torch.zeros(1, P, _num_action_, dtype=torch.int64, device=self.device)
        self.episode_return = torch.zeros(1, 1, P, device=self.device)
        self.episode_step = torch.zeros(1, 1, P, dtype=torch.int32, device=self.device)
        initial_done = torch.ones(1, 1, P, dtype=torch.uint8, device=self.device)
        
        obs = self.gym_env.reset()
        initial_frame = _format_frame(np.array([obs[i]['obs'] for i in range(P)]), self.device)

        game_info = torch.zeros(1, 1, P, 23, dtype=torch.float32, device=self.device)

        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            #last_action=initial_last_action,
            info=game_info
        )

    def step(self, action):
        obs, reward, done, game_info = self.gym_env.step(action)
        self.episode_step += 1
        episode_step = self.episode_step

        #print("------------------", done)
        if all(done):
            #print("------------------", done)
            obs = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1, P, device=self.device)
            self.episode_step = torch.zeros(1, 1, P, dtype=torch.int32, device=self.device)
        
        # maybe need a faster way to 
        frame = _format_frame(np.array([obs[i]['obs'] for i in range(P)]), device=self.device)

        reward = torch.tensor(reward, device=self.device).view(1, 1, P)
        done = torch.tensor(done, device=self.device).view(1, 1, P)

        self.episode_return += reward
        episode_return = self.episode_return
        
        game_info = torch.from_numpy(np.array([obs[i]['measurements'] for i in range(P)])).to(self.device)
    
        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            #last_action=action.view((1, ) + action.shape),
            info=game_info.view((1, 1) + game_info.shape)
        )
    def reset(self):
        return self.initial()

    def close(self):
        self.gym_env.close()