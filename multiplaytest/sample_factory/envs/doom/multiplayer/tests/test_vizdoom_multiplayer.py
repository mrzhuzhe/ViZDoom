import time
import unittest
from multiprocessing import Process
from unittest import TestCase

from sample_factory.envs.env_utils import vizdoom_available
from sample_factory.utils.utils import log, AttrDict
import logging

#python3 -m sample_factory.envs.doom.multiplayer.tests.test_vizdoom_multiplayer

"""
 python3 -m sample_factory.algorithms.appo.train_appo --env=doom_duel --train_for_seconds=360000 --algo=APPO --gamma=0.995 \
     --env_frameskip=2 --use_rnn=True --num_workers=72 --num_envs_per_worker=16 --num_policies=8 --ppo_epochs=1 --rollout=32 \
          --recurrence=32 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --benchmark=False --pbt_replace_reward_gap=0.5 \
               --pbt_replace_reward_gap_absolute=0.35 --pbt_period_env_steps=5000000 --with_pbt=True --pbt_start_mutation=100000000 --experiment=doom_duel_full

 python3 -m sample_factory.algorithms.appo.train_appo --env=doom_duel --train_for_seconds=360000 --algo=APPO --gamma=0.995 \
    --env_frameskip=2 --use_rnn=True --num_workers=12 --num_envs_per_worker=2 --num_policies=1 --ppo_epochs=1 --rollout=32 \
    --recurrence=32 --batch_size=128 --res_w=128 --res_h=72 --wide_aspect_ratio=False --benchmark=False --pbt_replace_reward_gap=0.5 \
    --with_pbt=False --experiment=doom_duel_full
"""

@unittest.skipUnless(vizdoom_available(), 'Please install VizDoom to run a full test suite')
class TestDoom(TestCase):
    @staticmethod
    def make_standard_dm(env_config):
        from sample_factory.envs.doom.doom_utils import make_doom_env
        from sample_factory.envs.tests.test_envs import default_doom_cfg

        cfg = default_doom_cfg()
        cfg.env_frameskip = 2
        env = make_doom_env('doom_duel', cfg=cfg, env_config=env_config)
        env.skip_frames = cfg.env_frameskip
        return env

    @staticmethod
    def doom_multiagent(make_multi_env, worker_index, num_steps=1000):
        env_config = AttrDict({'worker_index': worker_index, 'vector_index': 0, 'safe_init': False})
        multi_env = make_multi_env(env_config)

        obs = multi_env.reset()

        visualize = True
        start = time.time()

        for i in range(num_steps):
            actions = [multi_env.action_space.sample()] * len(obs)
            obs, rew, dones, infos = multi_env.step(actions)
            if visualize:
                multi_env.render()

            if i % 100 == 0 or any(dones):
                log.info('Rew %r done %r info %r', rew, dones, infos)

            if all(dones):
                multi_env.reset()

        took = time.time() - start
        log.info('Took %.3f seconds for %d steps', took, num_steps)
        log.info('Server steps per second: %.1f', num_steps / took)
        log.info('Observations fps: %.1f', num_steps * multi_env.num_agents / took)
        log.info('Environment fps: %.1f', num_steps * multi_env.num_agents * multi_env.skip_frames / took)

        multi_env.close()

    def test_doom_multiagent(self):
        self.doom_multiagent(self.make_standard_dm, worker_index=0)

    def test_doom_multiagent_parallel(self):
        num_workers = 2
        workers = []

        for i in range(num_workers):
            log.info('Starting worker #%d', i)
            worker = Process(target=self.doom_multiagent, args=(self.make_standard_dm, i, 200))
            worker.start()
            workers.append(worker)
            time.sleep(0.01)

        for i in range(num_workers):
            workers[i].join()
        handlers = log.handlers[:]
        for handler in handlers:
            log.removeHandler(handler)
            handler.close()
            
a = TestDoom()
a.test_doom_multiagent()