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
from zmq import device


def _format_frame(frame, device):
    frame = torch.from_numpy(frame).to(device)
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).


class Environment:
    def __init__(self, gym_env, device):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self.movement_reward = None
        self.device = device

    def initial(self):
        initial_reward = torch.zeros(1, 1,  device=self.device)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64, device=self.device)
        self.episode_return = torch.zeros(1, 1, device=self.device)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32, device=self.device)
        initial_done = torch.ones(1, 1, dtype=torch.uint8, device=self.device)
        initial_frame = _format_frame(self.gym_env.reset(), self.device)

        game_info = torch.zeros(1, self.gym_env.info_length, dtype=torch.float32, device=self.device)
        self.movement_reward = torch.zeros(1, 1, device=self.device)

        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
            movement_reward=self.movement_reward,
            info=game_info
        )

    def step(self, action):
        frame, reward, done, game_info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        self.movement_reward += game_info["movement_reward"]
        movement_reward = self.movement_reward

        if done:
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1, device=self.device)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32, device=self.device)
            self.movement_reward = torch.zeros(1, 1, device=self.device)

        frame = _format_frame(frame, device=self.device)
        reward = torch.tensor(reward, device=self.device).view(1, 1)
        done = torch.tensor(done, device=self.device).view(1, 1)
        game_info = torch.from_numpy(game_info["game_info"]).to(self.device)

        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
            movement_reward=movement_reward,
            info=game_info
        )
    def reset(self):
        return self.initial()

    def close(self):
        self.gym_env.close()