import sys
from multiprocessing import Queue, Process
import pyautogui
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import time
import numpy as np
from stable_baselines3.dqn import DQN
from stable_baselines3.ppo import PPO
from observer import Observer
from stable_baselines3.common.callbacks import BaseCallback

import pydirectinput

pydirectinput.FAILSAFE = False
pydirectinput.PAUSE = 0.0
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False


class Game(Env):
    def __init__(self, q):
        super().__init__()
        self.total_score = 0
        self.data_queue = q
        self.last_action_time = 0

        # 0 - go left
        # 1 - go right
        self.action_space = Discrete(2)

        # 1D vector with: [character_pos(0,1),branch_pos(0,1,2),branch_distance(0,inf),time]
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0.0]),
            high=np.array([1, 2, 1_000, 1.0]),
            shape=(4,),
            dtype=np.float64,
        )

    def _get_valid_observation(self):
        if not self.data_queue.empty():
            obs = self.data_queue.get()
            player_side, branch_side, branch_distance, is_game_over, time_percentage = obs

            if (player_side is not None and
                        branch_side is not None and
                        branch_distance is not None and
                        time_percentage is not None):
                return obs

        print("Warning: Observation timeout, returning default values")
        return [0, 2, 1000,False,0.0]

    def step(self, action):
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except:
                break
        # Execute action
        if action == 0:
            pydirectinput.press('left')
        else:
            pydirectinput.press('right')

        self.last_action_time = time.time()

        time.sleep(0.2)

        obs_data = self._get_valid_observation()
        player_side, branch_side, branch_distance, is_game_over, time_percentage = obs_data

        reward = 0

        if is_game_over:
            reward -= 15 #
            done = True
        else:
            reward += 2
           # reward -= 2 * (1 - time_percentage)
            done = False

        self.total_score += reward

        if player_side is None:
            player_side = 0
        if branch_side is None:
            branch_side = 2
        if branch_distance is None:
            branch_distance = 1000
        if time_percentage is None:
            time_percentage = 0.0

        observation = np.array([
            float(player_side),
            float(branch_side),
            float(min(branch_distance, 1000)),
            float(time_percentage)
        ], dtype=np.float64)

        info = {"score": self.total_score}
        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_score = 0
        self.last_action_time = 0

        # Clear the queue before reset
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except:
                break

        time.sleep(1)
        pydirectinput.press('space')

        time.sleep(0.5)

        obs_data = self._get_valid_observation()
        player_side, branch_side, branch_dist, is_game_over, time_perc = obs_data

        # Handle None values
        if player_side is None:
            player_side = 0
        if branch_side is None:
            branch_side = 2
        if branch_dist is None:
            branch_dist = 1000
        if time_perc is None:
            time_perc = 0.0

        observation = np.array([
            float(player_side),
            float(branch_side),
            float(min(branch_dist, 1000)),
            float(time_perc)
        ], dtype=np.float64)

        info = {}
        return observation, info

class ProgressCallback(BaseCallback):
    def __init__(self, check_freq: int = 10_000, verbose: int = 1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"{self.num_timesteps} steps")
        return True


if __name__ == '__main__':
    q1, data_queue = Queue(), Queue()
    obs = Observer(q1, data_queue)
    p = Process(target=obs.process_screen, args=())
    p.start()

    time.sleep(2)

    env = Game(data_queue)

    try:
        policy_kwargs = dict(
            net_arch=[128, 128],
        )
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="logs/"
        )
        #model.learn(total_timesteps=1_000_000, callback=ProgressCallback(check_freq=10_000))
        model.learn(total_timesteps=10_000, callback=ProgressCallback(check_freq=1_000))
        model.save(f"models/timber_{time.time()}")

    finally:
        p.terminate()
        p.join()
        sys.exit(1)

