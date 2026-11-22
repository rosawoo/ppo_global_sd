# Derived from CleanRL ppo_continuous_action.py (vwxyzjn/cleanrl, master).
# docs for the base implementation: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
#
# Study: global vs. state-dependent sigma in PPO. The ONLY difference between the two
# arms is --std-mode {global,state}. See CLAUDE.md (spec) and DECISIONS.md (deviations).
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from envs.wrappers import VecNormalizer, make_env, make_env_thunk
from scripts.eval import evaluate

STD_MODES = ("global", "state", "state_separate")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.use_deterministic_algorithms(True)`"""
    device: str = "cpu"
    """torch device (the networks are tiny; cpu is fastest)"""
    torch_threads: int = 1
    """intra-op threads per process (1 so that the sweep can run one process per core)"""
    runs_root: str = "runs"
    """runs are written to {runs_root}/{env_id}/{std_mode}/seed{seed}/"""
    save_model: bool = True
    """whether to save the final agent + normalizer to {run_dir}/checkpoint.pt"""

    # Policy parameterisation (the object of the study)
    std_mode: str = "global"
    """how sigma is computed: 'global' (state-independent parameter), 'state' (head on the
    shared actor trunk) or 'state_separate' (own 64-64 trunk; optional ablation)"""
    log_std_init: float = -0.6931471805599453
    """initial log sigma for every mode (log 0.5)"""
    logstd_min: float = -5.0
    """lower clamp on log sigma (both modes)"""
    logstd_max: float = 2.0
    """upper clamp on log sigma (both modes)"""
    logstd_head_init_std: float = 0.01
    """orthogonal-init gain of the state-dependent log-sigma head weights"""

    # Algorithm specific arguments
    env_id: str = "Pendulum-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    adam_beta1: float = 0.9
    """Adam beta1"""
    adam_eps: float = 1e-5
    """Adam epsilon (CleanRL default)"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization (per minibatch)"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    norm_obs: bool = True
    """running mean/std observation normalization"""
    obs_clip: float = 10.0
    """clip normalized observations to +-obs_clip"""
    norm_reward: bool = True
    """CleanRL NormalizeReward-style reward scaling"""
    reward_clip: float = 10.0
    """clip normalized rewards to +-reward_clip"""
    bootstrap_truncated: bool = True
    """bootstrap the value of the final observation when an episode is truncated by the time limit"""
    actor_hidden: int = 64
    """width of the two actor trunk layers"""
    critic_hidden: int = 256
    """width of the two critic layers"""

    # Evaluation
    eval_freq: int = 25000
    """evaluate every this many env steps (at the next update boundary); 0 disables"""
    eval_episodes: int = 20
    """episodes per policy type (stochastic, deterministic) per evaluation"""
    eval_seed_base: int = 100000
    """eval episode i always resets with seed eval_seed_base + i (same for all runs)"""
    eval_noise_seed: int = 0
    """seed of the action-noise generator used by the stochastic eval episodes"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class NonFiniteError(RuntimeError):
    """Raised when a NaN/inf shows up; the run is recorded as status='nan', never restarted."""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim, hidden):
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden, hidden)),
        nn.Tanh(),
    )


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, std_mode="global", log_std_init=np.log(0.5), logstd_min=-5.0, logstd_max=2.0,
                 logstd_head_init_std=0.01, actor_hidden=64, critic_hidden=256):
        super().__init__()
        assert std_mode in STD_MODES, std_mode
        self.std_mode, self.logstd_min, self.logstd_max = std_mode, logstd_min, logstd_max
        # NOTE: construction order matters. Everything shared by the two arms is created
        # first so that, for the same seed, it gets bit-identical initial weights.
        self.critic = nn.Sequential(mlp(obs_dim, critic_hidden), layer_init(nn.Linear(critic_hidden, 1), std=1.0))
        self.actor_trunk = mlp(obs_dim, actor_hidden)
        self.actor_mean = layer_init(nn.Linear(actor_hidden, act_dim), std=0.01)
        if std_mode == "global":
            self.actor_logstd = nn.Parameter(torch.full((1, act_dim), float(log_std_init)))
        else:
            if std_mode == "state_separate":
                self.actor_logstd_trunk = mlp(obs_dim, actor_hidden)
            # bias initialised so that logstd_head(h) ~= log_std_init at init
            self.actor_logstd_head = layer_init(
                nn.Linear(actor_hidden, act_dim), std=logstd_head_init_std, bias_const=float(log_std_init)
            )

    def mean_parameters(self):
        return list(self.actor_mean.parameters())

    def sigma_parameters(self):
        if self.std_mode == "global":
            return [self.actor_logstd]
        params = list(self.actor_logstd_head.parameters())
        if self.std_mode == "state_separate":
            params += list(self.actor_logstd_trunk.parameters())
        return params

    def get_value(self, x):
        return self.critic(x)

    def dist(self, x):
        h = self.actor_trunk(x)
        mean = self.actor_mean(h)
        if self.std_mode == "global":
            logstd = self.actor_logstd.expand_as(mean)
        elif self.std_mode == "state":
            logstd = self.actor_logstd_head(h)
        else:
            logstd = self.actor_logstd_head(self.actor_logstd_trunk(x))
        if not (torch.isfinite(mean).all() and torch.isfinite(logstd).all()):
            raise NonFiniteError("non-finite policy output")
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        return Normal(mean, logstd.exp()), logstd

    def get_action_and_value(self, x, action=None):
        probs, logstd = self.dist(x)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), logstd


def grad_norm(params):
    sq = sum(float(p.grad.pow(2).sum()) for p in params if p.grad is not None)
    return sq**0.5


class CsvLog:
    def __init__(self, path):
        self.f, self.writer = open(path, "w", newline=""), None

    def write(self, row: dict):
        if self.writer is None:
            self.writer = csv.DictWriter(self.f, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.f.flush()


def write_status(run_dir, **kw):
    tmp = os.path.join(run_dir, "status.json.tmp")
    with open(tmp, "w") as f:
        json.dump(kw, f, indent=2)
    os.replace(tmp, os.path.join(run_dir, "status.json"))


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.std_mode in STD_MODES, f"--std-mode must be one of {STD_MODES}"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_dir = os.path.join(args.runs_root, args.env_id, args.std_mode, f"seed{args.seed}")
    os.makedirs(os.path.join(run_dir, "evals"), exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(asdict(args), f, indent=2)
    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    update_log = CsvLog(os.path.join(run_dir, "updates.csv"))
    eval_log = CsvLog(os.path.join(run_dir, "evals.csv"))

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    if args.torch_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)

    # env setup. SAME_STEP autoreset = the pre-gymnasium-1.0 behaviour CleanRL was written
    # for: the obs returned on a `done` step is already the reset obs of the next episode.
    envs = gym.vector.SyncVectorEnv(
        [make_env_thunk(args.env_id) for _ in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    normalizer = VecNormalizer(
        envs.single_observation_space.shape, args.num_envs, args.gamma, args.norm_obs, args.norm_reward,
        args.obs_clip, args.reward_clip,
    )
    eval_env = make_env(args.env_id)

    agent = Agent(obs_dim, act_dim, args.std_mode, args.log_std_init, args.logstd_min, args.logstd_max,
                  args.logstd_head_init_std, args.actor_hidden, args.critic_hidden).to(device)
    # Building the sigma head consumes RNG in 'state' mode only; re-seed so both arms
    # draw the same exploration-noise stream from here on.
    torch.manual_seed(args.seed)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=args.adam_eps, betas=(args.adam_beta1, 0.999))

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    def run_eval(global_step, wall):
        out = os.path.join(run_dir, "evals", f"step_{global_step:09d}.npz")
        summary = evaluate(agent, normalizer, eval_env, args.eval_episodes, args.eval_seed_base, args.eval_noise_seed,
                           out, device)
        eval_log.write({"global_step": global_step, "wall_time_s": round(wall, 2), **summary})
        for k, v in summary.items():
            writer.add_scalar(f"eval/{k}", v, global_step)
        print(f"[eval] step={global_step} stoch_return={summary['stoch_return_mean']:.2f} "
              f"det_return={summary['det_return_mean']:.2f} stoch_success={summary['stoch_success_rate']:.2f} "
              f"sigma={summary['stoch_sigma_mean']:.3f}")

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    eval_time = 0.0
    status = "done"
    write_status(run_dir, status="running", global_step=0)
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(normalizer.obs(next_obs, update=True)).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_eval = 0
    ep_returns, ep_lengths, ep_success, ep_final_dist = [], [], [], []

    try:
        for iteration in range(1, args.num_iterations + 1):
            if args.eval_freq > 0 and global_step >= next_eval:
                t0 = time.time()
                run_eval(global_step, t0 - start_time)
                eval_time += time.time() - t0
                while next_eval <= global_step:
                    next_eval += args.eval_freq

            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            rollout_goal_dist = []
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                raw_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                done_np = np.logical_or(terminations, truncations)
                reward = normalizer.reward(reward, done_np)
                next_obs = torch.Tensor(normalizer.obs(raw_obs, update=True)).to(device)
                rollout_goal_dist.append(infos["goal_dist"])  # goal-relative state is in info every step

                if args.bootstrap_truncated and truncations.any():
                    # time-limit truncation is not a real termination: add gamma * V(s_final)
                    for i in np.where(np.logical_and(truncations, ~terminations))[0]:
                        final_obs = normalizer.obs(infos["final_obs"][i][None], update=False)
                        with torch.no_grad():
                            reward[i] += args.gamma * agent.get_value(torch.Tensor(final_obs).to(device)).item()

                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_done = torch.Tensor(done_np).to(device)

                if "final_info" in infos:
                    fin = infos["final_info"]
                    for i in np.where(infos["_final_info"])[0]:
                        ep_returns.append(float(fin["episode"]["r"][i]))
                        ep_lengths.append(int(fin["episode"]["l"][i]))
                        ep_final_dist.append(float(fin["goal_dist"][i]))
                        if "is_success" in fin:
                            ep_success.append(float(fin["is_success"][i]))
                        writer.add_scalar("charts/episodic_return", ep_returns[-1], global_step)
                        writer.add_scalar("charts/episodic_length", ep_lengths[-1], global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs, kls, gn_mean, gn_sigma, gn_trunk, gn_critic, gn_total = [], [], [], [], [], [], []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                        kls += [approx_kl.item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    if not torch.isfinite(loss):
                        raise NonFiniteError("non-finite loss")

                    optimizer.zero_grad()
                    loss.backward()
                    # H2 diagnostics: pre-clip gradient norms of the sigma vs. mean parameters
                    gn_mean += [grad_norm(agent.mean_parameters())]
                    gn_sigma += [grad_norm(agent.sigma_parameters())]
                    gn_trunk += [grad_norm(agent.actor_trunk.parameters())]
                    gn_critic += [grad_norm(agent.critic.parameters())]
                    gn_total += [float(nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm))]
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # sigma diagnostics on the rollout states, after the update
            with torch.no_grad():
                dist_b, logstd_b = agent.dist(b_obs)
                at_min = (logstd_b <= args.logstd_min).float().mean().item()
                at_max = (logstd_b >= args.logstd_max).float().mean().item()
                entropy_b = dist_b.entropy().sum(1).mean().item()
            if not all(torch.isfinite(p).all() for p in agent.parameters()):
                raise NonFiniteError("non-finite parameters")

            sps = int(global_step / max(time.time() - start_time - eval_time, 1e-9))
            row = {
                "global_step": global_step,
                "iteration": iteration,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy": entropy_b,
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": float(np.mean(kls)),
                "approx_kl_last": approx_kl.item(),
                "approx_kl_max": float(np.max(kls)),
                "clipfrac": float(np.mean(clipfracs)),
                "explained_variance": float(explained_var),
                "logstd_mean": logstd_b.mean().item(),
                "logstd_min": logstd_b.min().item(),
                "logstd_max": logstd_b.max().item(),
                "logstd_state_std": logstd_b.std(dim=0).mean().item(),
                "logstd_frac_at_min": at_min,
                "logstd_frac_at_max": at_max,
                "grad_norm_mean_head": float(np.mean(gn_mean)),
                "grad_norm_sigma": float(np.mean(gn_sigma)),
                "grad_norm_actor_trunk": float(np.mean(gn_trunk)),
                "grad_norm_critic": float(np.mean(gn_critic)),
                "grad_norm_total": float(np.mean(gn_total)),
                "train_return_mean": float(np.mean(ep_returns)) if ep_returns else np.nan,
                "train_length_mean": float(np.mean(ep_lengths)) if ep_lengths else np.nan,
                "train_success_rate": float(np.mean(ep_success)) if ep_success else np.nan,
                "train_final_goal_dist": float(np.mean(ep_final_dist)) if ep_final_dist else np.nan,
                "train_episodes": len(ep_returns),
                "rollout_goal_dist_mean": float(np.mean(rollout_goal_dist)),
                "SPS": sps,
            }
            for d in range(act_dim):
                row[f"logstd_mean_dim{d}"] = logstd_b[:, d].mean().item()
            ep_returns, ep_lengths, ep_success, ep_final_dist = [], [], [], []
            update_log.write(row)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for k, v in row.items():
                if k not in ("global_step", "iteration"):
                    group = "charts" if k in ("learning_rate", "SPS") or k.startswith(("train_", "rollout_")) else "losses"
                    writer.add_scalar(f"{group}/{k}", v, global_step)
            if iteration % 10 == 0 or iteration == args.num_iterations:
                print(f"it={iteration}/{args.num_iterations} step={global_step} SPS={sps} "
                      f"train_return={row['train_return_mean']:.2f} logstd_mean={row['logstd_mean']:.3f} "
                      f"kl={row['approx_kl']:.4f}")

        if args.eval_freq > 0:
            run_eval(global_step, time.time() - start_time)
    except NonFiniteError as e:
        # Do NOT restart with another seed: NaN frequency is itself an H2 result.
        status = "nan"
        print(f"[NaN] {e} at global_step={global_step}; recording and stopping.")

    wall = time.time() - start_time
    if args.save_model:
        torch.save(
            {"agent": agent.state_dict(), "normalizer": normalizer.state_dict(), "args": asdict(args),
             "obs_dim": obs_dim, "act_dim": act_dim, "global_step": global_step},
            os.path.join(run_dir, "checkpoint.pt"),
        )
    write_status(run_dir, status=status, global_step=global_step, wall_clock_s=round(wall, 1),
                 eval_wall_clock_s=round(eval_time, 1), device=str(device), torch_threads=args.torch_threads,
                 train_sps=int(global_step / max(wall - eval_time, 1e-9)))
    envs.close()
    eval_env.close()
    writer.close()
