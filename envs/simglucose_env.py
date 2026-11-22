"""Closed-loop basal-insulin control of a simulated type-1-diabetes patient.

Built on `simglucose` (the UVA/Padova 2008 simulator, Jinyu Xie's Python implementation),
using its own components: `T1DPatient`, `CGMSensor('Dexcom')`, `InsulinPump('Insulet')`,
`RandomScenario` (random daily meals) and `simulation.env.T1DSimEnv`. What follows the
package defaults and what is ours is listed in DECISIONS.md (D27-D33). In short:

* default: patient `adolescent#001` (the simglucose gym default), Dexcom CGM with 3-min
  sampling (one env step = 3 simulated minutes), random initial BG, random start hour,
  random meals, basal-only control (no meal bolus), termination at BG < 10 or > 600 mg/dL.
* ours: a [-1, 1] action that maps to an insulin rate in units of the patient's own
  steady-state basal (the native action box is 0-30 U/min, ~2000x a typical basal);
  a reward from the Magni risk index plus an alive bonus (the default `risk_diff` reward
  telescopes to risk(BG_0) - risk(BG_T)); an observation with 1 h of CGM and insulin
  history and the time of day (CGM alone is far from Markov); a 24 h time limit; every
  random component re-drawn from the reset seed (the package's gymnasium wrapper replays
  the same day after every reset).

The precision gradient the study is about: an insulin overdose at low BG drives the
patient into hypoglycaemia, where the risk (and, below 10 mg/dL, termination) rises
steeply, while the same overdose at high BG only brings BG back towards range.
"""
from __future__ import annotations

import warnings
from collections import deque
from datetime import datetime

import gymnasium as gym
import numpy as np

TARGET_BG = 112.5  # mg/dL; minimiser of the Magni risk function
EPISODE_METRIC_KEYS = ("tir", "tbr", "tbr54", "tar", "mean_bg", "mean_risk")


def magni_risk(bg):
    """Magni/Kovatchev blood-glucose risk index (0 at 112.5 mg/dL, ~100 at 20 and 600 mg/dL)."""
    bg = np.maximum(np.asarray(bg, dtype=np.float64), 1.0)
    f = 1.509 * (np.log(bg) ** 1.084 - 5.381)
    return 10.0 * f**2


def _import_simglucose():
    # simglucose imports the legacy `gym` package and pkg_resources at import time.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from simglucose.actuator.pump import InsulinPump
        from simglucose.controller.base import Action
        from simglucose.patient.t1dpatient import T1DPatient
        from simglucose.sensor.cgm import CGMSensor
        from simglucose.simulation.env import T1DSimEnv
        from simglucose.simulation.scenario_gen import RandomScenario
    return InsulinPump, Action, T1DPatient, CGMSensor, T1DSimEnv, RandomScenario


class _FastParams:
    """The patient's parameters as plain attributes (identical values to the pandas Series).

    simglucose's ODE right-hand side reads ~40 parameters per call through pandas
    attribute access, which dominates its run time. This makes the simulator ~8x faster
    with bit-identical trajectories (tests/test_envs.py checks this)."""

    def __init__(self, series):
        self._series = series
        for k, v in series.items():
            setattr(self, k, v)

    @property
    def iloc(self):
        return self._series.iloc


class SimglucoseEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, patient_name: str = "adolescent#001", action_map: str = "linear", max_rate_mult: float = 10.0,
                 history: int = 20, max_steps: int = 480, render_mode: str | None = None):
        """
        action_map: 'linear' -> rate = (a + 1) / 2 * max_rate_mult * basal   (a in [-1, 1])
                    'log'    -> rate = basal * max_rate_mult ** a            (multiplicative noise)
        history:    CGM readings / insulin rates in the observation (20 x 3 min = 1 h)
        max_steps:  time limit (480 x 3 min = 24 h)
        """
        assert action_map in ("linear", "log"), action_map
        (self._Pump, self._Action, self._Patient, self._Sensor, self._SimEnv,
         self._Scenario) = _import_simglucose()
        self.patient_name, self.action_map, self.max_rate_mult = patient_name, action_map, float(max_rate_mult)
        self.history, self.max_steps, self.render_mode = history, max_steps, render_mode
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2 * history + 2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.sim = None
        self._fig = None

    # ---- action map -------------------------------------------------------------------
    def rate_multiple(self, a):
        """Insulin rate in multiples of the patient's steady-state basal, for action(s) a."""
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        if self.action_map == "linear":
            return (a + 1.0) / 2.0 * self.max_rate_mult
        return self.max_rate_mult**a

    # ---- gym API ----------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        hour = int(rng.integers(0, 24))
        s_patient, s_sensor, s_scenario = (int(s) for s in rng.integers(0, 2**31 - 1, size=3))
        patient = self._Patient.withName(self.patient_name, random_init_bg=True, seed=s_patient)
        patient._params = _FastParams(patient._params)
        sensor = self._Sensor.withName("Dexcom", seed=s_sensor)
        pump = self._Pump.withName("Insulet")
        scenario = self._Scenario(start_time=datetime(2018, 1, 1, hour, 0, 0), seed=s_scenario)
        self.sim = self._SimEnv(patient, sensor, pump, scenario)
        step = self.sim.reset()
        self.basal = float(patient._params.u2ss * patient._params.BW / 6000.0)  # U/min
        cgm = float(step.observation.CGM)
        self.cgm_hist = deque([cgm] * self.history, maxlen=self.history)
        self.ins_hist = deque([1.0] * self.history, maxlen=self.history)  # x basal; init state is basal steady state
        self.t = 0
        self.bg_trace, self.cgm_trace, self.rate_trace, self.meal_trace = [float(step.info["bg"])], [cgm], [], []
        return self._obs(), self._info(float(step.info["bg"]), meal=0.0)

    def step(self, action):
        assert self.sim is not None, "call reset() first"
        a = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        mult = float(self.rate_multiple(a))
        step = self.sim.step(self._Action(basal=mult * self.basal, bolus=0.0), reward_fun=_no_reward)
        bg, cgm = float(step.info["bg"]), float(step.observation.CGM)
        risk = float(step.info["risk"])
        reward = 1.0 - risk / 100.0
        terminated = bool(step.done)  # simglucose default: BG < 10 or BG > 600 mg/dL
        self.t += 1
        truncated = (self.t >= self.max_steps) and not terminated
        self.cgm_hist.append(cgm)
        self.ins_hist.append(mult)
        self.bg_trace.append(bg)
        self.cgm_trace.append(cgm)
        self.rate_trace.append(mult)
        self.meal_trace.append(float(step.info["meal"]))
        return self._obs(), reward, terminated, truncated, self._info(bg, meal=float(step.info["meal"]))

    # ---- helpers ----------------------------------------------------------------------
    def hour_of_day(self):
        t = self.sim.time
        return t.hour + t.minute / 60.0

    def _obs(self):
        h = 2 * np.pi * self.hour_of_day() / 24.0
        return np.concatenate([np.asarray(self.cgm_hist), np.asarray(self.ins_hist), [np.sin(h), np.cos(h)]]).astype(np.float32)

    def _info(self, bg, meal):
        cgm = self.cgm_hist[-1]
        trend = (self.cgm_hist[-1] - self.cgm_hist[-6]) / 15.0  # mg/dL per min over 15 min
        bgs = np.asarray(self.bg_trace)
        tir = float(np.mean((bgs >= 70) & (bgs <= 180)))
        tbr = float(np.mean(bgs < 70))
        info = {
            # goal-relative state for the sigma analysis (envs/wrappers.py ENV_SPECS has the names)
            "goal_rel": np.array([cgm, trend, bg, self.hour_of_day(), float(np.mean(self.ins_hist))], dtype=np.float32),
            "goal_dist": np.float32(magni_risk(bg)),
            "bg": bg, "meal": meal, "insulin_rate_x_basal": self.rate_trace[-1] if self.rate_trace else 1.0,
            # running episode metrics (final step = episode values)
            "tir": tir, "tbr": tbr, "tbr54": float(np.mean(bgs < 54)), "tar": float(np.mean(bgs > 180)),
            "mean_bg": float(bgs.mean()), "mean_risk": float(np.mean(magni_risk(bgs))),
            # consensus targets (Battelino et al. 2019): >70% time in 70-180, <4% below 70
            "is_success": bool(tir >= 0.70 and tbr < 0.04),
            # safety failure: any level-2 hypoglycaemia (< 54 mg/dL) so far
            "crashed": bool(np.any(bgs < 54)),
        }
        return info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self._fig is None:
            self._fig, self._axes = plt.subplots(2, 1, figsize=(6.4, 4.0), dpi=80, sharex=True,
                                                 gridspec_kw={"height_ratios": [2, 1]})
        ax0, ax1 = self._axes
        ax0.clear()
        ax1.clear()
        hrs = np.arange(len(self.bg_trace)) * 3 / 60.0
        ax0.axhspan(70, 180, color="#e1f0e1", lw=0)
        ax0.plot(hrs, self.bg_trace, color="#0b0b0b", lw=1.5, label="BG")
        ax0.plot(hrs, self.cgm_trace, color="#898781", lw=0.8, label="CGM")
        ax0.set_ylim(30, 400)
        ax0.set_ylabel("mg/dL")
        ax0.legend(loc="upper right", fontsize=7, frameon=False)
        if self.rate_trace:
            ax1.step(hrs[1:], self.rate_trace, where="pre", color="#2a78d6", lw=1.2)
        ax1.set_ylim(0, self.max_rate_mult * 1.05)
        ax1.set_ylabel("insulin (x basal)")
        ax1.set_xlabel("hours")
        ax1.set_xlim(0, self.max_steps * 3 / 60.0)
        self._fig.tight_layout()
        self._fig.canvas.draw()
        return np.asarray(self._fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None


def _no_reward(_bg_last_hour):
    return 0.0  # reward is computed from the risk index in SimglucoseEnv.step
