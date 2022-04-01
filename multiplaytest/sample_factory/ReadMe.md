# Run

> python3 -m sample_factory.algorithms.appo.train_appo --env=doom_deathmatch_bots --train_for_seconds=3600 --algo=APPO --use_rnn=True --gamma=0.995 --env_frameskip=2 --rollout=32 --num_workers=16 --num_envs_per_worker=24 --num_policies=2 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=512 --res_w=128 --res_h=72 --wide_aspect_ratio=False --with_pbt=True --pbt_period_env_steps=5000000 --experiment=doom_deathmatch_bots

> python3 -m sample_factory.algorithms.appo.train_appo --env=doom_duel_bots --train_for_seconds=18000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --reward_scale=0.5 --num_workers=16 --num_envs_per_worker=32 --num_policies=2 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --benchmark=False --res_w=128 --res_h=72 --wide_aspect_ratio=False --pbt_replace_reward_gap=0.2 --pbt_replace_reward_gap_absolute=3.0 --pbt_period_env_steps=5000000 --save_milestones_sec=1800 --with_pbt=True --experiment=doom_duel_bots_2

> python3 -m sample_factory.algorithms.appo.enjoy_appo --env=doom_deathmatch_bots --algo=APPO --experiment=doom_deathmatch_bots

> python3 -m sample_factory.algorithms.appo.enjoy_appo --env=doom_duel_bots --algo=APPO --experiment=doom_duel_bots

> python3 -m sample_factory.algorithms.appo.enjoy_appo --env=doom_duel --algo=APPO --experiment=doom_duel_full