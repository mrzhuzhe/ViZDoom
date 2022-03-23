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


WIDTH = 640
HEIGHT = 480
TICRATE = 350
_ACTION_ = 15

# global port and network timeout 
DEFAULT_UDP_PORT = int(os.environ.get('DOOM_DEFAULT_UDP_PORT', 40300))
# This try except block is to increase the env timeout connection flag in travis
try:
    vizdoom_env_timeout = int(os.environ['TRAVIS_VIZDOOM_ENV_TIMEOUT'])
except KeyError:
    vizdoom_env_timeout = 4

class MyDoom(Env):
    def __init__(self, 
    player_id, num_agents, max_num_players, num_bots, 
    respawn_delay=0, timelimit=0.0,
    render=False, config='../scenarios/ssl2.cfg'):
        self.game = vzd.DoomGame()
        self.game.load_config(config)
        
        
        self.player_id = player_id
        self.num_agents = num_agents  # num agents that are not humans or bots
        self.max_num_players = max_num_players
        self.num_bots = num_bots
        
        self.respawn_delay = respawn_delay
        self.timelimit = timelimit
        
        self.initialized = False
        self.init_info = None

        if render == False:           
            self.game.set_window_visible(False)
        else:            
            self.game.set_mode(vzd.Mode.SPECTATOR)
            #self.game.set_mode(vzd.Mode.ASYNC_PLAYER)
            #self.game.set_ticrate(TICRATE)
            self.game.set_window_visible(True)
        
        self._player_init()

        self.observation_space = Box(low=0, high=255, shape=(HEIGHT, WIDTH, 1), dtype=np.uint8) 
        self.action_space = Discrete(_ACTION_)

        self.info_length = 2
        self.health = 100
        self.ammo = 20

    def _is_server(self):
        return self.player_id == 0

    def _game_init(self, with_locking=True, max_parallel=10):
        self.game.init()
  
    def _player_init(self):
        if self.initialized:
            # Doom env already initialized!
            return

        #self._create_doom_game(self.mode)
        port = DEFAULT_UDP_PORT if self.init_info is None else self.init_info.get('port', DEFAULT_UDP_PORT)

        if self._is_server():

            # This process will function as a host for a multiplayer game with this many players (including the host).
            # It will wait for other machines to connect using the -join parameter and then
            # start the game when everyone is connected.
            game_args_list = [
                f'-host {self.max_num_players}',
                f'-port {port}',
                '-deathmatch',  # Deathmatch rules are used for the game.
                f'+timelimit {self.timelimit}',  # The game (episode) will end after this many minutes have elapsed.
                '+sv_forcerespawn 1',  # Players will respawn automatically after they die.
                '+sv_noautoaim 1',  # Autoaim is disabled for all players.
                '+sv_respawnprotect 1',  # Players will be invulnerable for two second after spawning.
                '+sv_spawnfarthest 1',  # Players will be spawned as far as possible from any other players.
                '+sv_nocrouch 1',  # Disables crouching.
                '+sv_nojump 1',  # Disables jumping.
                '+sv_nofreelook 1',  # Disables free look with a mouse (only keyboard).
                '+sv_noexit 1',  # Prevents players from exiting the level in deathmatch before timelimit is hit.
                f'+viz_respawn_delay {self.respawn_delay}',  # Sets delay between respanws (in seconds).
                f'+viz_connect_timeout {vizdoom_env_timeout}',
            ]
            self.game.add_game_args(' '.join(game_args_list))

            # Additional commands:
            #
            # disables depth and labels buffer and the ability to use commands
            # that could interfere with multiplayer game (should use this in evaluation)
            # '+viz_nocheat 1'

            # Name your agent and select color
            # colors:
            # 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            self.game.add_game_args(f'+name AI{self.player_id}_host +colorset 0')

        else:
            # Join existing game.
            self.game.add_game_args(
                f'-join 127.0.0.1:{port} '  # Connect to a host for a multiplayer game.
                f'+viz_connect_timeout {vizdoom_env_timeout} '
            )

            # Name your agent and select color
            # colors:
            # 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            self.game.add_game_args(f'+name AI{self.player_id} +colorset 0')

        self.game.set_episode_timeout(int(self.timelimit * 60 * self.game.get_ticrate()))

        self._game_init()  # locking is handled by the multi-agent wrapper       
        self.initialized = True

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
            POSITION_X, POSITION_Y, ANGLE, SELECTED_WEAPON,\
            SELECTED_WEAPON_AMMO, HEALTH, ARMOR,\
            USER2, DEATHCOUNT, FRAGCOUNT, HITCOUNT, DEAD, \
            WEAPON0, WEAPON1, WEAPON2, WEAPON3, WEAPON4, WEAPON5, \
            WEAPON6, WEAPON7, WEAPON8, WEAPON9, AMMO0, AMMO1, AMMO2,\
            AMMO3, AMMO4, AMMO5, AMMO6, AMMO7, AMMO8, AMMO9, ATTACK_READY,\
            DAMAGECOUNT, PLAYER_NUMBER, PLAYER_COUNT, PLAYER1_FRAGCOUNT, \
            PLAYER2_FRAGCOUNT, PLAYER3_FRAGCOUNT, PLAYER4_FRAGCOUNT, PLAYER5_FRAGCOUNT, \
            PLAYER6_FRAGCOUNT, PLAYER7_FRAGCOUNT, PLAYER8_FRAGCOUNT, PLAYER9_FRAGCOUNT =  _stat.game_variables

            ammo_delta = SELECTED_WEAPON_AMMO - self.ammo
            health_delta = HEALTH - self.health
            
            self.ammo = SELECTED_WEAPON_AMMO
            self.health = HEALTH

            ammo_reward = 0
            # ammo picked up 
            if ammo_delta > 0:
                ammo_reward = 0.5
            # medic picked up
            medic_reward = 0
            if health_delta > 0:
                medic_reward = 0.5

            reward = movement_reward + ammo_reward + medic_reward
            
        else:
            state = np.zeros(self.observation_space.shape)
        info = { 
            'game_info': np.array([[self.ammo/20 , self.health/100 ]], dtype=np.float32),
            "movement_reward": movement_reward }
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
