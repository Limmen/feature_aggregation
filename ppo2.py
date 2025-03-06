import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from large_pomdp_parser import load_model, sample_next_state_and_obs
from pomdp_util import POMDPUtil
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from rocksample_simulator import RockSampleSimulator


class RockSampleEnv(gym.Env):
    """
    RockSample environment returning the belief distribution as observation.
    """

    def __init__(self):
        super().__init__()
        self.env = RockSampleSimulator(n=10, k=10, seed=999)
        self.b = self.env.initial_belief()
        self.x = self.env.init_state()
        self.action_space = spaces.Discrete(self.env.num_actions)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.env.num_states,),
            dtype=np.float32
        )
        self.t = 0

    def step(self, action):
        self.t += 1
        a = int(action)
        x_next, z, cost, done = self.env.step(state_id=self.x, action_id=a)
        self.x = x_next
        # Update belief
        # new_b = self.env.belief_operator(b=self.b, a_id=a, z_id=z)
        new_b = self.env.belief_operator_pf(b=self.b, a_id=a, z_id=z, num_particles=50)
        self.b = new_b
        done = False
        info = {}
        if self.t > 100:
            done = True
        return np.array(self.b, dtype=np.float32), -cost, done, done, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Re-initialize
        self.b = self.env.initial_belief()
        self.x = self.env.init_state()
        self.t = 0
        return np.array(self.b, dtype=np.float32), {}


class DiscountedRewardLoggerCallback(BaseCallback):
    """
    Logs the average DISCOUNTED episode returns every `log_freq` episodes,
    and also prints the elapsed time in seconds.
    """

    def __init__(self, gamma=0.99, log_freq=10, verbose=0):
        super().__init__(verbose)
        self.gamma = gamma  # discount factor for accumulating returns
        self.log_freq = log_freq
        self.episode_discounted_returns = []
        self.episode_count = 0
        self.current_discounted_return = 0.0
        self.episode_step = 0  # track step index in the current episode
        self.start_time = time.time()

    def _on_step(self) -> bool:
        done = self.locals["dones"][0]
        reward = self.locals["rewards"][0]

        # accumulate discounted return:
        # G += gamma^t * r_t
        self.current_discounted_return += (self.gamma ** self.episode_step) * reward
        self.episode_step += 1

        if done:
            self.episode_count += 1
            self.episode_discounted_returns.append(self.current_discounted_return)

            # print average over last 10 episodes every `log_freq` episodes
            if (self.episode_count % self.log_freq) == 0:
                avg_10 = np.mean(self.episode_discounted_returns[-100:])
                elapsed = time.time() - self.start_time
                print(f"Episode={self.episode_count}, "
                      f"DiscountedReturn={self.current_discounted_return:.2f}, "
                      f"AvgLast100={avg_10:.2f}, "
                      f"Elapsed={elapsed:.2f}s")

            # reset for next episode
            self.current_discounted_return = 0.0
            self.episode_step = 0

        return True


def main():
    # 1) Create your custom env
    env = RockSampleEnv()
    # 2) Wrap in Monitor
    env = Monitor(env)

    # 3) Create PPO model with explicit network size in 'policy_kwargs'
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        policy_kwargs={
            "net_arch": [64, 64]  # two hidden layers of size 64
        },
        learning_rate=1e-4,
        n_steps=5012,
        batch_size=128,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5
    )

    # 4) Logging callback, logs average reward every 5 episodes
    callback = DiscountedRewardLoggerCallback(log_freq=100, gamma=0.95)

    # 5) Train for some timesteps
    model.learn(total_timesteps=5000000, callback=callback)

    # # (Optional) Evaluate or test the final policy
    # obs, info = env.reset()
    # for _ in range(10):
    #     action, _ = model.predict(obs)
    #     obs, reward, done, truncated, _ = env.step(action)
    #     print("Test step => Reward:", reward)
    #     if done or truncated:
    #         obs, info = env.reset()


if __name__ == "__main__":
    main()
