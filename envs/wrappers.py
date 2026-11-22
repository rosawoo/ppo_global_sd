"""Environment construction shared by training, evaluation, analysis and demos.

Everything here is identical for both arms of the study (--std-mode global / state).

* `ParkingGoalRelInfo` / `PendulumGoalRelInfo` / `LunarLanderGoalRelInfo` put the
  goal-relative state into `info["goal_rel"]` and its Euclidean distance into
  `info["goal_dist"]` at *every* reset and step (CLAUDE.md §2) - this is the x-axis of
  the sigma-vs-state analysis.
* `ParkingFlatten` turns the parking Dict observation into a flat Box made of
  `observation` + `desired_goal` (`achieved_goal` duplicates `observation`).
* `VecNormalizer` is a CleanRL/gym `NormalizeObservation` + `NormalizeReward`
  equivalent that lives outside the env so its statistics are shared across the
  vector envs, can be checkpointed, and can be frozen for evaluation (DECISIONS.md D3).
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

PARKING_SCALES = np.array([100.0, 100.0, 5.0, 5.0, 1.0, 1.0])


@dataclass(frozen=True)
class EnvSpec:
    goal_rel_names: tuple[str, ...]
    dist_name: str  # what info["goal_dist"] measures
    action_names: tuple[str, ...]
    learned_metric: str  # eval column used for the H2 "learned" threshold
    learned_threshold: float


ENV_SPECS: dict[str, EnvSpec] = {
    "parking-v0": EnvSpec(
        goal_rel_names=("dx_m", "dy_m", "dvx_mps", "dvy_mps", "dcos_h", "dsin_h"),
        dist_name="distance to goal (m)",
        action_names=("acceleration", "steering"),
        learned_metric="stoch_success_rate",
        learned_threshold=0.5,
    ),
    "Pendulum-v1": EnvSpec(
        goal_rel_names=("theta_rad", "theta_dot"),
        dist_name="|theta| from upright (rad)",
        action_names=("torque",),
        learned_metric="stoch_return_mean",
        learned_threshold=-300.0,
    ),
    "LunarLanderContinuous-v3": EnvSpec(
        goal_rel_names=("x", "y", "vx", "vy", "angle", "angular_vel"),
        dist_name="distance to pad (obs units)",
        action_names=("main_engine", "lateral_engine"),
        learned_metric="stoch_return_mean",
        learned_threshold=100.0,
    ),
}


class _GoalRelInfo(gym.Wrapper):
    """Adds info['goal_rel'] (vector) and info['goal_dist'] (scalar) on reset and step."""

    def _goal_rel(self, obs) -> np.ndarray:
        raise NotImplementedError

    def _dist(self, rel: np.ndarray) -> float:
        raise NotImplementedError

    def _annotate(self, obs, info):
        rel = np.asarray(self._goal_rel(obs), dtype=np.float32)
        info = dict(info)
        info["goal_rel"] = rel
        info["goal_dist"] = np.float32(self._dist(rel))
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._annotate(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._annotate(obs, info)


class ParkingGoalRelInfo(_GoalRelInfo):
    """(achieved_goal - desired_goal), un-scaled back to metres / m/s. Must wrap the Dict env."""

    def _goal_rel(self, obs):
        return (obs["achieved_goal"] - obs["desired_goal"]) * PARKING_SCALES

    def _dist(self, rel):
        return np.hypot(rel[0], rel[1])


class PendulumGoalRelInfo(_GoalRelInfo):
    """[theta, theta_dot] with theta wrapped to (-pi, pi], 0 = upright (the goal)."""

    def _goal_rel(self, obs):
        theta = np.arctan2(obs[1], obs[0])
        return np.array([theta, obs[2]])

    def _dist(self, rel):
        return abs(rel[0])


class LunarLanderGoalRelInfo(_GoalRelInfo):
    """The landing pad is at the origin of the observation frame, at rest, upright."""

    def _goal_rel(self, obs):
        return obs[:6]

    def _dist(self, rel):
        return np.hypot(rel[0], rel[1])


class ParkingFlatten(gym.ObservationWrapper):
    """Dict(observation, achieved_goal, desired_goal) -> Box(concat(observation, desired_goal))."""

    def __init__(self, env):
        super().__init__(env)
        n = env.observation_space["observation"].shape[0] + env.observation_space["desired_goal"].shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)


class Float32Obs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        sp = env.observation_space
        self.observation_space = gym.spaces.Box(sp.low.astype(np.float32), sp.high.astype(np.float32), dtype=np.float32)

    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32)


def make_env(env_id: str, render_mode: str | None = None) -> gym.Env:
    """Default-config env + goal-relative info + flat float32 obs + episode stats + action clip.

    The policy's raw sample is clipped to the action bounds *here* (gym ClipAction), so
    the log-prob stored by PPO is that of the unclipped sample, as in CleanRL.
    Observation / reward normalization is NOT applied here; see `VecNormalizer`.
    """
    if env_id not in ENV_SPECS:
        raise ValueError(f"unknown env {env_id!r}; add it to envs.wrappers.ENV_SPECS")
    if env_id == "parking-v0":
        import highway_env

        gym.register_envs(highway_env)
        env = gym.make(env_id, render_mode=render_mode)  # default config, untouched
        env = ParkingGoalRelInfo(env)
        env = ParkingFlatten(env)
    elif env_id == "Pendulum-v1":
        env = gym.make(env_id, render_mode=render_mode)
        env = PendulumGoalRelInfo(env)
        env = Float32Obs(env)
    else:
        env = gym.make(env_id, render_mode=render_mode)
        env = LunarLanderGoalRelInfo(env)
        env = Float32Obs(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env


def make_env_thunk(env_id: str):
    return lambda: make_env(env_id)


class RunningMeanStd:
    """Same update rule as gymnasium.wrappers.utils.RunningMeanStd."""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        b_mean, b_var, b_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        new_mean = self.mean + delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + np.square(delta) * self.count * b_count / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot


class VecNormalizer:
    """Running obs normalization (clip +-obs_clip) and return-based reward scaling
    (clip +-rew_clip), shared across the envs of a vector env.

    Mirrors what CleanRL gets from NormalizeObservation -> clip -> NormalizeReward -> clip.
    """

    def __init__(self, obs_shape, num_envs: int, gamma: float, norm_obs=True, norm_reward=True,
                 obs_clip: float = 10.0, rew_clip: float = 10.0, epsilon: float = 1e-8):
        self.obs_rms = RunningMeanStd(obs_shape)
        self.ret_rms = RunningMeanStd(())
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.gamma, self.norm_obs, self.norm_reward = gamma, norm_obs, norm_reward
        self.obs_clip, self.rew_clip, self.epsilon = obs_clip, rew_clip, epsilon

    def obs(self, obs: np.ndarray, update: bool) -> np.ndarray:
        """Normalize a batch [N, obs_dim] (or a single obs when update=False)."""
        if not self.norm_obs:
            return np.asarray(obs, dtype=np.float32)
        if update:
            self.obs_rms.update(obs)
        out = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return np.clip(out, -self.obs_clip, self.obs_clip).astype(np.float32)

    def reward(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        """Training only: scale rewards by the running std of the discounted return."""
        if not self.norm_reward:
            return np.asarray(reward, dtype=np.float32)
        self.returns = self.returns * self.gamma + reward
        self.ret_rms.update(self.returns)
        out = reward / np.sqrt(self.ret_rms.var + self.epsilon)
        self.returns[np.asarray(done, dtype=bool)] = 0.0
        return np.clip(out, -self.rew_clip, self.rew_clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "obs_mean": self.obs_rms.mean, "obs_var": self.obs_rms.var, "obs_count": self.obs_rms.count,
            "ret_mean": self.ret_rms.mean, "ret_var": self.ret_rms.var, "ret_count": self.ret_rms.count,
            "norm_obs": self.norm_obs, "norm_reward": self.norm_reward, "gamma": self.gamma,
            "obs_clip": self.obs_clip, "rew_clip": self.rew_clip, "epsilon": self.epsilon,
        }

    @classmethod
    def from_state_dict(cls, sd: dict, num_envs: int = 1) -> "VecNormalizer":
        self = cls(np.shape(sd["obs_mean"]), num_envs, sd["gamma"], sd["norm_obs"], sd["norm_reward"],
                   sd["obs_clip"], sd["rew_clip"], sd["epsilon"])
        self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count = sd["obs_mean"], sd["obs_var"], sd["obs_count"]
        self.ret_rms.mean, self.ret_rms.var, self.ret_rms.count = sd["ret_mean"], sd["ret_var"], sd["ret_count"]
        return self
