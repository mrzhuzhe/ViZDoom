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

"""
todo
1. teacher kl
2. baseline training only

"""

import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from .core import environment
from .core import file_writer
from .core import prof
from .core import vtrace
from .core import td_lambda
from .core import upgo

from sample_factory.utils.utils import AttrDict
from sample_factory.envs.doom.doom_utils import make_doom_env
from sample_factory.envs.tests.test_envs import default_doom_cfg


from .action_distributions import calc_num_actions, calc_num_logits, get_action_distribution


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


#def compute_entropy_loss(logits):
#    """Return the entropy loss, i.e., the negative entropy of the policy."""
#    policy = F.softmax(logits, dim=-1)
#    log_policy = F.log_softmax(logits, dim=-1)
#    return torch.sum(policy * log_policy)

def compute_entropy_loss(entropy):
    return -torch.sum(entropy)

#def compute_policy_gradient_loss(logits, actions, advantages):
#    cross_entropy = F.nll_loss(
#        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
#        target=torch.flatten(actions, 0, 1),
#        reduction="none",
#    )
#    cross_entropy = cross_entropy.view_as(advantages)
#    return torch.sum(cross_entropy * advantages.detach())


def compute_policy_gradient_loss(
        action_log_probs: torch.Tensor,
        advantages: torch.Tensor
) -> torch.Tensor:
    cross_entropy = -action_log_probs.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())

def compute_teacher_kl_loss(
        learner_policy_logits: torch.Tensor,
        teacher_policy_logits: torch.Tensor
) -> torch.Tensor:
    learner_policy_log_probs = F.log_softmax(learner_policy_logits, dim=-1)
    teacher_policy = F.softmax(teacher_policy_logits, dim=-1)
    kl_div = F.kl_div(
        learner_policy_log_probs,
        teacher_policy.detach(),
        reduction="none",
        log_target=False
    ).sum(dim=-1)
    # Sum over y, x, and action_planes dimensions to combine kl divergences from different actions
    return kl_div.sum(dim=-1).sum(dim=-1)


def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    actor_model: torch.nn.Module,
    buffers: Buffers
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        gym_env = create_env(flags, actor_index)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        env = environment.Environment(gym_env, device=flags.actor_device)        
        env_output = env.initial()
        
        agent_output = actor_model(env_output)

        while True:
            #env.gym_env.render()
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]        

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output = actor_model(env_output)

                timings.time("actor_model")

                #print("env_output", env_output)
                env_output = env.step(torch.flatten(agent_output["action"], 0, 2))            
                
                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)
        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    timings.time("device")
    return batch


def learn(
    flags,
    actor_model,
    leaner_model,
    teacher_model,
    batch,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""

    """
    frame torch.Size([32, 16, 1, 100, 160])
    reward torch.Size([32, 16])
    done torch.Size([32, 16])
    episode_return torch.Size([32, 16])
    episode_step torch.Size([32, 16])
    policy_logits torch.Size([32, 16, 8])
    baseline torch.Size([32, 16])
    last_action torch.Size([32, 16])
    action torch.Size([32, 16])
    movement_reward torch.Size([32, 16])
    info torch.Size([32, 16, 2])
    """
    with lock:

        learner_outputs = leaner_model(batch)

        if flags.use_teacher:
            with torch.no_grad():
                teacher_model_outputs = teacher_model(batch)
                teacher_model_outputs = {key: tensor[:-1] for key, tensor in teacher_model_outputs.items()}
        else:
            teacher_model_outputs = None

        


        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        values = learner_outputs["baseline"]


        behavior_policy_action_distribution = get_action_distribution(leaner_model.action_space, torch.flatten(batch["policy_logits"], 0, 2))
        target_policy_action_distribution = get_action_distribution(leaner_model.action_space, torch.flatten(learner_outputs["policy_logits"], 0, 2))

        #print(behavior_policy_action_distribution.log_prob(torch.flatten(batch["action"], 0, 2)).shape, values.shape, clipped_rewards.shape, learner_outputs["baseline"].shape, bootstrap_value.shape)

        behavior_policy_log_props=behavior_policy_action_distribution.log_prob(torch.flatten(batch["action"], 0, 2)).view(32, 16,2)
        target_policy_log_props=target_policy_action_distribution.log_prob(torch.flatten(batch["action"], 0, 2)).view(32, 16,2)

        vtrace_returns = vtrace.from_log_props(
            behavior_policy_log_props=behavior_policy_log_props,
            target_policy_log_props=target_policy_log_props,
            #actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=values,
            bootstrap_value=bootstrap_value,
        )  

        #print("vtrace_returns.log_rhos", vtrace_returns.log_rhos.shape)

        if flags.use_teacher:
            teacher_kl_loss = flags.teacher_kl_cost * compute_teacher_kl_loss(
                learner_outputs["policy_logits"],
                teacher_model_outputs["policy_logits"]
            )
        else:
            teacher_kl_loss = torch.tensor(0)

        if flags.use_tdlamda:
            td_lambda_returns = td_lambda.td_lambda(
                    rewards=batch["reward"],
                    values=values,
                    bootstrap_value=bootstrap_value,
                    discounts=discounts,
                    lmb=flags.lmb
                )
            _adv = td_lambda_returns.vs - values
        else:
            _adv = vtrace_returns.vs - learner_outputs["baseline"]

            
        #"""
        if flags.use_upgo:
            upgo_returns = upgo.upgo(
                    rewards=batch["reward"],
                    values=values,
                    bootstrap_value=bootstrap_value,
                    discounts=discounts,
                    lmb=flags.lmb
                )

            upgo_clipped_importance = torch.minimum(
                    vtrace_returns.log_rhos.exp(),
                    torch.ones_like(vtrace_returns.log_rhos)
                ).detach()

            upgo_pg_loss = flags.upgo_cost * compute_policy_gradient_loss(
                    learner_outputs["policy_logits"],
                    batch["action"],
                    upgo_clipped_importance * upgo_returns.advantages
                )
        else: 
            upgo_pg_loss = torch.tensor(0)
        #"""



        #pg_loss = compute_policy_gradient_loss(
        #    learner_outputs["policy_logits"],
        #    batch["action"],
        #    vtrace_returns.pg_advantages,
        #)

        pg_loss = compute_policy_gradient_loss(
            target_policy_log_props,
            vtrace_returns.pg_advantages,
        )
        
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            _adv
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            target_policy_action_distribution.entropy()
        )

        total_loss = pg_loss + baseline_loss + entropy_loss + upgo_pg_loss + teacher_kl_loss
        #print(batch["episode_return"], [batch["done"]])
        episode_returns = batch["episode_return"][batch["done"]]
        #print('batch["episode_return"]', batch["episode_return"], 'batch["done"]', batch["done"] )
        #print('episode_returns', episode_returns.shape)
        #movement_reward = batch["movement_reward"][batch["done"]]

        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "upgo_pg_loss": upgo_pg_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "teacher_kl_loss": teacher_kl_loss.item(),
            #"movement_reward_return": torch.mean(movement_reward).item()
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(leaner_model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(leaner_model.state_dict())
        return stats


def create_buffers(flags, obs_shape, num_logit, num_actions, info_len) -> Buffers:
    T = flags.unroll_length
    #N = 1 
    P = 2
    specs = dict(
        frame=dict(size=(T + 1, P, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1, P), dtype=torch.float32),
        done=dict(size=(T + 1, P), dtype=torch.bool),
        episode_return=dict(size=(T + 1, P), dtype=torch.float32),
        episode_step=dict(size=(T + 1, P), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, P, num_logit), dtype=torch.float32),
        baseline=dict(size=(T + 1, P), dtype=torch.float32),
        #last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1, P, num_actions), dtype=torch.int64),
        #movement_reward=dict(size=(T + 1, P, 1), dtype=torch.float32),
        info=dict(size=(T + 1, P, info_len), dtype=torch.float32),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    #os.environ["OMP_NUM_THREADS"] = "1"
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    # only for actor model
    flags.actor_device = torch.device(flags.actor_device_str) 

    env = create_env(flags)
    _info_len = 23

    actor_model = Net(env.observation_space['obs'].shape, env.action_space, _info_len).to(flags.actor_device)
    
    if flags.checkPoint:
        actor_model.load_state_dict(torch.load(
                flags.checkPoint,
                map_location=torch.device("cpu")
            )["model_state_dict"])

    buffers = create_buffers(flags, env.observation_space['obs'].shape, actor_model.num_logit, actor_model.num_actions, _info_len)
    
    n_trainable_params = sum(p.numel() for p in actor_model.parameters() if p.requires_grad)
    logging.info(f'Training model with {n_trainable_params:,d} parameters.')

    #model.eval()
    actor_model.share_memory()

    actor_processes = []
    #ctx = mp.get_context("fork")
    ctx = mp
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                actor_model,
                buffers
            ),
        )
        actor.start()
        actor_processes.append(actor)

    # Load teacher model for KL loss
    if flags.use_teacher:
        _teacher_model_path = flags.teacher_model_path
        teacher_model = Net(env.observation_space['obs'].shape, env.action_space, _info_len).to(device=flags.device)
        teacher_model.load_state_dict(
            torch.load(
                _teacher_model_path,
                map_location=torch.device("cpu")
            )["model_state_dict"]
        )
        teacher_model.eval()
    else:
        teacher_model = None
        
    learner_model = Net(
        env.observation_space['obs'].shape, env.action_space, _info_len
    ).to(device=flags.device)

    if flags.checkPoint:
        learner_model.load_state_dict(torch.load(
                flags.checkPoint,
                map_location=torch.device("cpu")
            )["model_state_dict"])


    learner_model.share_memory()

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "upgo_pg_loss",
        "teacher_kl_loss",
        #"movement_reward_return"
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                timings,
            )
            stats = learn(
                flags, actor_model, learner_model, teacher_model, batch, optimizer, scheduler
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                _out = {k: stats[k] for k in stat_keys}
                to_log.update(_out)
                plogger.log(to_log)
                step += T * B
        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": actor_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()


class AtariNet(nn.Module):
    def __init__(self, observation_shape, action_space, info_shape, ):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape

        self.action_space = action_space
        self.num_logit= calc_num_logits(action_space)
        self.num_actions = calc_num_actions(action_space)

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.info_fc = nn.Sequential(
            nn.Linear(info_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
            )

        self.fc = nn.Linear(3840+128, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + 1
       
        self.policy = nn.Linear(core_output_size, self.num_logit)
        self.baseline = nn.Linear(core_output_size, 1)

    def forward(self, inputs):
        x = inputs["frame"]  # [T, B, P, C, H, W].
        T, B, P, *_ = x.shape

        _allView_ = T * B * P
        
        x = torch.flatten(x, 0, 2)  # Merge time and batch and oppotent.       
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(_allView_, -1)

        _info_feature = self.info_fc(inputs["info"])
        
        _info_feature = _info_feature.view(_allView_, -1)
        
        x = torch.cat([x, _info_feature], dim=-1)

        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"].float(), -1, 1).view(_allView_, 1)

        core_input = torch.cat([x, clipped_reward], dim=-1)
        
        core_output = core_input

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        action_distribution = get_action_distribution(self.action_space, policy_logits)
        if self.training:
            #action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)           
            #print(action_distribution.actions)
            #action, log_probs = 
            action = action_distribution.sample()
            #print("action.shape", action.shape)
            #print("log_probs.shape", log_probs.shape)
        else:
            # Don't sample when testing.
            #action = torch.argmax(policy_logits, dim=1)
            action = action_distribution.act()

        policy_logits = policy_logits.view(T, B, P, self.num_logit)
        baseline = baseline.view(T, B, P)

        action = action.view(T, B, P, self.num_actions)

        return dict(policy_logits=policy_logits, baseline=baseline, action=action)
        

Net = AtariNet

def make_standard_dm(env_config):
    cfg = default_doom_cfg()
    cfg.env_frameskip = 2
    env = make_doom_env('doom_duel', cfg=cfg, env_config=env_config)
    env.skip_frames = cfg.env_frameskip
    return env

def create_env(flags, worker_index = 0):
    env_config = AttrDict({'worker_index': worker_index, 'vector_index': 0, 'safe_init': False})
    multi_env = make_standard_dm(env_config)
    return multi_env
