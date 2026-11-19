"""
PPO (Stable-Baselines3) on the Go2 self-recovery task (paper: "Self-Recovery of Quadrupedal Robot Using DRL").

Train:  python -m scripts.train_ppo
Resume: python -m scripts.train_ppo --resume checkpoints/ppo_go2_recovery_<N>_steps
Play:   python -m scripts.train_ppo --play checkpoints/ppo_go2_recovery
Run both from the repository root (assets use relative paths).
"""
import argparse
import copy
from functools import partial

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from control.action_adapter.joint_action_adapter import ActionAdapter
from env.env_factory import create_env
from env.go2_env import Go2Env
from env.tasks.recovery_task import RecoveryTask
from env.tasks.self_righting_scenario import SelfRightingScenario
from es_framework.core.env_config import EnvConfig
import matplotlib.pyplot as plotter
# ----------------------------------------
# NETWORK  (single place to change the NN)
# ----------------------------------------
# Paper: 3 hidden layers (256, 256, 16) with ReLU. For another architecture change this dict,
# or add features_extractor_class=<BaseFeaturesExtractor subclass> / features_extractor_kwargs.
POLICY_KWARGS = dict(
    activation_fn=nn.ReLU,
    net_arch=dict(pi=[256, 256, 16], vf=[256, 256, 16]),
)

# ----------------------------------------
# PPO hyperparameters (paper, Table IV)
# ----------------------------------------
PPO_KWARGS = dict(
    learning_rate=3.3e-4,
    n_steps=1024,
    batch_size=32,
    n_epochs=10,
    gamma=0.97,
    gae_lambda=0.95,
    ent_coef=2e-5,
    vf_coef=0.3,
    max_grad_norm=0.7,
)




MAX_STEPS = 1600  # 16 s at 100 Hz (paper)

estado_glob = {
    "tempo": [],
    "altura": []
}
ENV_CONFIG = EnvConfig(env_class=partial(Go2Env,shared_state=estado_glob),
                       controller_class=partial(ActionAdapter, kp=50.0, kd=2.0, rescale_to_limits=True),
                       task_class=partial(RecoveryTask, noise=True),
                       scene_path="sim/assets/unitree_go2/scene.xml",
                       urdf_path="sim/assets/unitree_go2/go2.urdf",
                       tpe_model_path=None,
                       normalizer_params={
                           "joint_limits": 1,
                           "torque_limits": 1
                       },
                       render=False)



class Timer():
    time_started = False 
    start_time = 0.0
    duracao = 0.0


def make_env(render=False):
    """Factory called inside each worker process (MuJoCo/Pinocchio objects are not picklable)."""
    cfg = copy.copy(ENV_CONFIG)
    cfg.render = render
    env, _ = create_env(cfg)
    env.max_step = MAX_STEPS
    # paper: dropped from 0.5 m with random orientation and joint positions
    env.scenario = SelfRightingScenario(random_only=True, drop_height=0.5)
    return env


def train(args):
    vec_env = make_vec_env(make_env,
                           n_envs=args.n_envs,
                           vec_env_cls=SubprocVecEnv,
                           vec_env_kwargs={"start_method": "spawn"})

    if args.resume:
        # restores policy, optimizer state and num_timesteps from the checkpoint zip
        model = PPO.load(args.resume, env=vec_env, device=args.device, tensorboard_log="runs/ppo")
    else:
        model = PPO("MlpPolicy",
                    vec_env,
                    policy_kwargs=POLICY_KWARGS,
                    tensorboard_log="runs/ppo",
                    verbose=1,
                    seed=args.seed,
                    device=args.device,
                    **{
                        **PPO_KWARGS, "batch_size": args.batch_size
                    })

    checkpoint = CheckpointCallback(save_freq=max(100_000 // args.n_envs, 1),
                                    save_path="checkpoints",
                                    name_prefix="ppo_go2_recovery")

    # with reset_num_timesteps=False SB3 counts total_timesteps as *additional* steps
    remaining = max(args.timesteps - model.num_timesteps, 0)
    model.learn(total_timesteps=remaining, callback=checkpoint, reset_num_timesteps=not args.resume)
    model.save("checkpoints/ppo_go2_recovery")

#h_altura
def play(path):
    env = make_env(render=True)
    model = PPO.load(path, device="cpu")
    

    obs, _ = env.reset()
    clock = Timer()
    while True:
        
        action, _ = model.predict(obs, deterministic=True)  # mean action only (paper)
        obs, reward, terminated, truncated, info = env.step(action, clock)
        if terminated or truncated:
            conj_tempo = list(env.estado["tempo"])
            conj_altura = list(env.estado["altura"])
            plotter.figure(figsize=(9,5))
            plotter.plot(conj_tempo,conj_altura,linestyle='-', color='tab:blue',linewidth=0.5)
            plotter.title("Mudança de alturas")
            plotter.xlabel('steps')
            plotter.ylabel('Altura')
            plotter.grid(True, linestyle='-',linewidth=0.5,color='gray',alpha=0.7)
            plotter.show()

            obs, _ = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", metavar="MODEL", help="load a saved model and visualize it in the MuJoCo viewer")
    parser.add_argument("--resume", metavar="MODEL", help="continue training from a checkpoint zip")
    parser.add_argument("--timesteps", type=int, default=4_000_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=PPO_KWARGS["batch_size"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.play:
        play(args.play)
    else:
        train(args)
