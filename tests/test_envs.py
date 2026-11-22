"""Wrappers, normalizer and evaluation plumbing."""
import numpy as np
import pytest
import torch

from envs.wrappers import ENV_SPECS, PARKING_SCALES, RunningMeanStd, VecNormalizer, make_env
from ppo import Agent
from scripts.eval import evaluate


def test_parking_flat_obs_and_goal_rel():
    env = make_env("parking-v0")
    obs, info = env.reset(seed=3)
    assert obs.shape == (12,) and obs.dtype == np.float32
    raw = env.unwrapped.observation_type.observe()
    np.testing.assert_allclose(obs, np.concatenate([raw["observation"], raw["desired_goal"]]), rtol=1e-6)
    np.testing.assert_allclose(obs[:6], raw["achieved_goal"], rtol=1e-6)  # why achieved_goal is dropped
    for _ in range(5):
        obs, r, term, trunc, info = env.step(np.array([0.3, 0.1], dtype=np.float32))
        rel = (obs[:6] - obs[6:]) * PARKING_SCALES
        np.testing.assert_allclose(info["goal_rel"], rel, rtol=1e-4, atol=1e-5)
        assert np.isclose(info["goal_dist"], np.hypot(rel[0], rel[1]), rtol=1e-4)
        assert "is_success" in info


def test_parking_config_is_default():
    cfg = make_env("parking-v0").unwrapped.config
    assert cfg["reward_weights"] == [1, 0.3, 0, 0, 0.02, 0.02]
    assert cfg["success_goal_reward"] == 0.12 and cfg["collision_reward"] == -5
    assert cfg["policy_frequency"] == 5 and cfg["duration"] == 100 and cfg["add_walls"]


def test_pendulum_goal_rel_is_angle_from_upright():
    env = make_env("Pendulum-v1")
    obs, info = env.reset(seed=0)
    th, thdot = env.unwrapped.state
    wrapped = ((th + np.pi) % (2 * np.pi)) - np.pi
    np.testing.assert_allclose(info["goal_rel"], [wrapped, thdot], atol=1e-5)
    assert np.isclose(info["goal_dist"], abs(wrapped), atol=1e-5)


@pytest.mark.parametrize("env_id", list(ENV_SPECS))
def test_actions_are_clipped_in_the_wrapper(env_id):
    env = make_env(env_id)
    obs, info = env.reset(seed=0)
    assert len(info["goal_rel"]) == len(ENV_SPECS[env_id].goal_rel_names)
    big = np.full(env.unwrapped.action_space.shape, 1e3, dtype=np.float32)
    obs, r, term, trunc, info = env.step(big)  # must not raise / produce NaN
    assert np.isfinite(obs).all() and np.isfinite(r)
    assert len(ENV_SPECS[env_id].action_names) == big.shape[0]


def test_running_mean_std_matches_batch_statistics():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, size=(5000, 4))
    rms = RunningMeanStd((4,))
    for chunk in np.split(x, 50):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, x.mean(0), atol=1e-3)
    np.testing.assert_allclose(rms.var, x.var(0), rtol=1e-3)


def test_normalizer_freeze_clip_and_roundtrip():
    rng = np.random.default_rng(0)
    nz = VecNormalizer((3,), num_envs=4, gamma=0.99)
    for _ in range(200):
        nz.obs(rng.normal(5, 2, size=(4, 3)), update=True)
        nz.reward(rng.normal(size=4), np.zeros(4, dtype=bool))
    before = nz.state_dict()["obs_mean"].copy()
    out = nz.obs(np.full((1, 3), 1e6), update=False)
    assert np.all(out == 10.0)  # clipped
    np.testing.assert_array_equal(before, nz.obs_rms.mean)  # frozen
    nz2 = VecNormalizer.from_state_dict(nz.state_dict())
    x = rng.normal(5, 2, size=(2, 3))
    np.testing.assert_array_equal(nz.obs(x, False), nz2.obs(x, False))


def test_evaluate_is_reproducible_and_leaves_global_rng_alone(tmp_path):
    env = make_env("Pendulum-v1")
    torch.manual_seed(0)
    agent = Agent(3, 1, "state")
    nz = VecNormalizer((3,), 1, 0.99)
    torch.manual_seed(7)
    expected_next = torch.randn(3)
    torch.manual_seed(7)
    s1 = evaluate(agent, nz, env, 2, seed_base=100, noise_seed=0, out_path=str(tmp_path / "a.npz"))
    assert torch.equal(torch.randn(3), expected_next)
    s2 = evaluate(agent, nz, env, 2, seed_base=100, noise_seed=0)
    np.testing.assert_equal(s1, s2)  # NaN-aware (success/crash rates are NaN on Pendulum)
    d = np.load(tmp_path / "a.npz")
    n = len(d["t"])
    assert n == 4 * 200 and d["ep_return"].shape == (4,)
    for k in ("obs", "goal_rel", "goal_dist", "mean", "std", "action", "reward", "episode_id", "deterministic"):
        assert len(d[k]) == n
    det = d["deterministic"]
    np.testing.assert_array_equal(d["action"][det], d["mean"][det])
    assert not np.allclose(d["action"][~det], d["mean"][~det])


SG = "simglucose-adolescent1-v0"


def test_simglucose_fast_params_are_bit_identical():
    """envs.simglucose_env._FastParams only changes how parameters are read, not the dynamics."""
    import warnings
    from datetime import datetime

    from envs.simglucose_env import _FastParams, _import_simglucose

    Pump, Action, Patient, Sensor, SimEnv, Scenario = _import_simglucose()

    def run(fast):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = Patient.withName("adolescent#001", random_init_bg=True, seed=3)
            if fast:
                p._params = _FastParams(p._params)
            env = SimEnv(p, Sensor.withName("Dexcom", seed=4), Pump.withName("Insulet"),
                         Scenario(start_time=datetime(2018, 1, 1, 6, 0, 0), seed=5))
            env.reset()
            return np.array([env.step(Action(basal=0.02 + 0.01 * np.sin(t / 9), bolus=0)).info["bg"]
                             for t in range(120)])

    np.testing.assert_array_equal(run(False), run(True))


def test_simglucose_seeding_reward_and_info():
    from envs.simglucose_env import magni_risk

    def roll(seed, a=-0.6, n=40):
        env = make_env(SG)
        obs, info = env.reset(seed=seed)
        assert obs.shape == (42,) and obs.dtype == np.float32
        out = []
        for _ in range(n):
            obs, r, term, trunc, info = env.step(np.array([a], dtype=np.float32))
            # reward = alive bonus - Magni risk of the true BG / 100
            assert np.isclose(r, 1.0 - magni_risk(info["bg"]) / 100.0, atol=1e-6)
            out.append(info["bg"])
        assert len(info["goal_rel"]) == len(ENV_SPECS[SG].goal_rel_names)
        assert {"is_success", "crashed", "tir", "tbr"} <= set(info)
        return np.array(out)

    np.testing.assert_array_equal(roll(5), roll(5))
    assert not np.array_equal(roll(5), roll(6))
    assert np.isclose(magni_risk(112.5), 0.0, atol=1e-3)
    assert magni_risk(50) > magni_risk(250) > magni_risk(180)  # the asymmetry the study relies on


def test_simglucose_action_maps():
    from envs.simglucose_env import SimglucoseEnv

    lin, log = SimglucoseEnv(action_map="linear"), SimglucoseEnv(action_map="log")
    np.testing.assert_allclose(lin.rate_multiple([-1.0, 0.0, 1.0, 5.0]), [0.0, 5.0, 10.0, 10.0])
    np.testing.assert_allclose(log.rate_multiple([-1.0, 0.0, 1.0]), [0.1, 1.0, 10.0])


def test_simglucose_env_can_be_cloned_for_the_noise_probe():
    import copy

    env = make_env(SG)
    env.reset(seed=7)
    for _ in range(20):
        env.step(np.array([-0.6], dtype=np.float32))
    clone = copy.deepcopy(env)
    a = [env.step(np.array([0.2], dtype=np.float32))[4]["bg"] for _ in range(15)]
    b = [clone.step(np.array([0.2], dtype=np.float32))[4]["bg"] for _ in range(15)]
    np.testing.assert_array_equal(a, b)
