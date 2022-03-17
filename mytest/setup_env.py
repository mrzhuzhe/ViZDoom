import os
from random import choice
from time import sleep
from typing import Any, Dict, Tuple
import vizdoom as vzd
import numpy as np

import cv2
import gym
from gym import Env
from gym.spaces import Discrete, Box


WIDTH = 160
HEIGHT = 100
TICRATE = 350
_ACTION_ = 8
class MyDoom(Env):
    #def __init__(self, render=False, config='../scenarios/deadly_corridor_s1.cfg'):
    def __init__(self, render=False, config='../scenarios/battle.cfg'):
    #def __init__(self, render=False, config='../scenarios/basic.cfg'):
        self.game = vzd.DoomGame()
        self.game.load_config(config)

        if render == False:           
            self.game.set_window_visible(False)
        else:            
            #self.game.set_mode(vzd.Mode.PLAYER)
            #self.game.set_mode(vzd.Mode.ASYNC_PLAYER)
            #self.game.set_ticrate(TICRATE)
            self.game.set_window_visible(True)
        
        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(HEIGHT, WIDTH, 1), dtype=np.uint8) 
        self.action_space = Discrete(_ACTION_)

        # Game variables: HEALTH DAMAGE_TAKEN HITCOUNT SELECTED_WEAPON_AMMO
        """
        self.damage_taken = 0
        self.hitcount = 0        
        """
        self.info_length = 2
        self.ammo = 20
        self.health = 100

        
    def step(self, action):
        actions = np.identity(_ACTION_, dtype=np.uint8)
        movement_reward = self.game.make_action(actions[action], 4)
        done =  self.game.is_episode_finished()
        reward = 0
        if self.game.get_state():
            _stat = self.game.get_state()
            state = _stat.screen_buffer
            state = self.resize(state)
            #info = { "ammo": _stat.game_variables[0], "health": _stat.game_variables[1] }
            #info = { "health": _stat.game_variables[0] }
            
            # Reward shaping
            #health, damage_taken, hitcount, ammo  = _stat.game_variables
            POSITION_X, POSITION_Y, ANGLE, SELECTED_WEAPON, SELECTED_WEAPON_AMMO, HEALTH, USER2 =  _stat.game_variables

            self.ammo = SELECTED_WEAPON_AMMO
            self.health = HEALTH
            """
            # Calculate reward deltas
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.ammo
            self.ammo = ammo

            #print(movement_reward, damage_taken_delta*10, hitcount_delta*200, ammo_delta*5)
            # https://github.com/mrzhuzhe/ViZDoom/blob/master/doc/Types.md#gamevariable
            #reward = movement_reward + damage_taken_delta*10 + hitcount_delta*400  + ammo_delta*5
            reward = movement_reward + hitcount_delta*200 
            """
            reward = movement_reward 
            #info = { "POSITION_X": POSITION_X, "POSITION_Y": POSITION_Y, "ANGLE": ANGLE, "SELECTED_WEAPON": SELECTED_WEAPON, "SELECTED_WEAPON_AMMO": SELECTED_WEAPON_AMMO, "HEALTH": HEALTH, "USER2": USER2  }
        else:
            state = np.zeros(self.observation_space.shape)
        
        #info = { "ammo": self.ammo, "health": self.health }
        info = np.array([[self.ammo/20 , self.health/100 ]], dtype=np.float32)
        return state, reward, done, info

    def resize(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (HEIGHT, WIDTH, 1))
        return state
    def reset(self):
        self.game.new_episode()
        _stat = self.game.get_state()
        return self.resize(_stat.screen_buffer)
    def close(self) -> None:
        return self.game.close()
