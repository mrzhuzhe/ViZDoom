from enum import Enum
import socket
import gym
from setUpEnv import WIDTH, HEIGHT, _ACTION_
import numpy as np 
from gym.spaces import Discrete, Box
import time

import threading
from multiprocessing import Process
from queue import Empty, Queue

def is_udp_port_available(port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', port))
        sock.close()
    except OSError as exc:
        print(f'UDP port {port} cannot be used {str(exc)}')
        return False
    else:
        return True

def find_available_port(start_port, increment=1000):
    port = start_port
    while port < 65535 and not is_udp_port_available(port):
        port += increment

    print('Port %r is available', port)
    return port

def safe_get(q, timeout=1e6, msg='Queue timeout'):
    """Using queue.get() with timeout is necessary, otherwise KeyboardInterrupt is not handled."""
    while True:
        try:
            return q.get(timeout=timeout)
        except Empty:
            log.warning(msg)


def udp_port_num(env_config):
    if env_config is None:
        return DEFAULT_UDP_PORT
    port_to_use = DEFAULT_UDP_PORT + 100 * env_config.worker_index + env_config.vector_index
    return port_to_use


class TaskType(Enum):
    INIT, TERMINATE, RESET, STEP, STEP_UPDATE, INFO, SET_ATTR = range(7)


class MultiAgentEnvWorker:
    def __init__(self, player_id, make_env_func, env_config, use_multiprocessing=False, reset_on_init=True):
        self.player_id = player_id
        self.make_env_func = make_env_func
        self.env_config = env_config
        self.reset_on_init = reset_on_init
 
        self.process = threading.Thread(target=self.start)
        self.task_queue, self.result_queue = Queue(), Queue()

        self.process.start()

    def _init(self, init_info):
        log.info('Initializing env for player %d, init_info: %r...', self.player_id, init_info)
        env = init_multiplayer_env(self.make_env_func, self.player_id, self.env_config, init_info)
        if self.reset_on_init:
            env.reset()
        return env

    @staticmethod
    def _terminate(env):
        if env is None:
            return
        env.close()

    @staticmethod
    def _get_info(env):
        """Specific to custom VizDoom environments."""
        info = {}
        if hasattr(env.unwrapped, 'get_info_all'):
            info = env.unwrapped.get_info_all()  # info for the new episode
        return info

    def _set_env_attr(self, env, player_id, attr_chain, value):
        """Allows us to set an arbitrary attribute of the environment, e.g. attr_chain can be unwrapped.foo.bar"""
        assert player_id == self.player_id

        attrs = attr_chain.split('.')
        curr_attr = env
        try:
            for attr_name in attrs[:-1]:
                curr_attr = getattr(curr_attr, attr_name)
        except AttributeError:
            log.error('Env does not have an attribute %s', attr_chain)

        attr_to_set = attrs[-1]
        setattr(curr_attr, attr_to_set, value)

    def start(self):
        env = None

        while True:
            data, task_type = safe_get(self.task_queue)

            if task_type == TaskType.INIT:
                env = self._init(data)
                self.result_queue.put(None)  # signal we're done
                continue

            if task_type == TaskType.TERMINATE:
                self._terminate(env)
                break

            results = None
            if task_type == TaskType.RESET:
                results = env.reset()
            elif task_type == TaskType.INFO:
                results = self._get_info(env)
            elif task_type == TaskType.STEP or task_type == TaskType.STEP_UPDATE:
                # collect obs, reward, done, and info
                action = data
                env.unwrapped.update_state = task_type == TaskType.STEP_UPDATE
                results = env.step(action)
            elif task_type == TaskType.SET_ATTR:
                player_id, attr_chain, value = data
                self._set_env_attr(env, player_id, attr_chain, value)
            else:
                raise Exception(f'Unknown task type {task_type}')

            self.result_queue.put(results)



class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents):
        gym.Env.__init__(self)
        self.num_agents = num_agents
        self.observation_space = Box(low=0, high=255, shape=(HEIGHT, WIDTH, 1), dtype=np.uint8) 
        self.action_space = Discrete(_ACTION_)

    def _player_initialized(self):
        if self.initialized:
            return

        self.workers = [
            MultiAgentEnvWorker(i, self.make_env_func, self.env_config, reset_on_init=self.reset_on_init)
            for i in range(self.num_agents)
        ]

        init_attempt = 0
        while True:
            init_attempt += 1
            try:
                port_to_use = udp_port_num(self.env_config)
                port = find_available_port(port_to_use, increment=1000)
                init_info = dict(port=port)
               
                for i, worker in enumerate(self.workers):
                    worker.task_queue.put((init_info, TaskType.INIT))
                    if self.safe_init:
                        time.sleep(1.0)  # just in case
                    else:
                        time.sleep(0.05)

                for i, worker in enumerate(self.workers):
                    worker.result_queue.get(timeout=20)

            except Exception:
                raise RuntimeError('Critical error: worker stuck on initialization. Abort!')
            else:
                break
        self.initialized = True
    def info(self):
        self._ensure_initialized()
        info = self.await_tasks(None, TaskType.INFO)[0]
        return info

    def reset(self):
        self._ensure_initialized()
        observation = self.await_tasks(None, TaskType.RESET, timeout=2.0)[0]
        return observation

    def step(self, actions):
        self._ensure_initialized()

        for frame in range(self.skip_frames - 1):
            self.await_tasks(actions, TaskType.STEP)

        obs, rew, dones, infos = self.await_tasks(actions, TaskType.STEP_UPDATE)
        for info in infos:
            info['num_frames'] = self.skip_frames

        if all(dones):
            obs = self.await_tasks(None, TaskType.RESET, timeout=2.0)[0]

        if self.enable_rendering:
            self.last_obs = obs

        return obs, rew, dones, infos

    # noinspection PyUnusedLocal
    def render(self, *args, **kwargs):
        self.enable_rendering = True

        if self.last_obs is None:
            return

        render_multiagent = True
        if render_multiagent:
            obs_display = [o['obs'] for o in self.last_obs]
            obs_grid = concat_grid(obs_display)
            cv2.imshow('vizdoom', obs_grid)
        else:
            obs_display = self.last_obs[0]['obs']
            cv2.imshow('vizdoom', cvt_doom_obs(obs_display))

        cv2.waitKey(1)

    def close(self):
        if self.workers is not None:
            # log.info('Stopping multiagent env %d...', self.env_config.worker_index)
            for worker in self.workers:
                worker.task_queue.put((None, TaskType.TERMINATE))
                time.sleep(0.1)
            for worker in self.workers:
                worker.process.join()

    def seed(self, seed=None):
        """Does not really make sense for the wrapper. Individual envs will be uniquely seeded on init."""
        pass

    def set_env_attr(self, agent_idx, attr_chain, value):
        data = (agent_idx, attr_chain, value)
        worker = self.workers[agent_idx]
        worker.task_queue.put((data, TaskType.SET_ATTR))

        result = safe_get(worker.result_queue, timeout=0.1)
        assert result is None
