import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from genesis_lr.envs.minipupper_maze_env import MiniPupperMazeEnv


def make_env(env_name, headless=True):
    if env_name == 'minipupper_maze_env':
        return MiniPupperMazeEnv(headless=headless)
    else:
        raise ValueError(f'Unknown env: {env_name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='minipupper_maze_env')
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--headless', action='store_true', default=False)
    args = parser.parse_args()

    env = DummyVecEnv([lambda: make_env(args.env, headless=args.headless)])

    model = PPO('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(f'{args.env}_ppo')


if __name__ == '__main__':
    main()
