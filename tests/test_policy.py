"""Init equivalence, shapes, log-prob correctness, clamp (CLAUDE.md §3.1 invariants)."""
import math

import numpy as np
import pytest
import torch

from ppo import Agent, NonFiniteError

OBS, ACT, LOG_STD_INIT = 12, 2, math.log(0.5)


def build(mode, seed=0, **kw):
    torch.manual_seed(seed)
    return Agent(OBS, ACT, mode, LOG_STD_INIT, **kw)


def states(n=4096, scale=3.0):
    return torch.randn(n, OBS, generator=torch.Generator().manual_seed(123)) * scale


@pytest.mark.parametrize("mode", ["state", "state_separate"])
def test_shared_parts_have_identical_init(mode):
    g, s = build("global"), build(mode)
    for name in ("critic", "actor_trunk", "actor_mean"):
        for pg, ps in zip(getattr(g, name).parameters(), getattr(s, name).parameters()):
            assert torch.equal(pg, ps), name
    x = states()
    assert torch.equal(g.dist(x)[0].mean, s.dist(x)[0].mean)
    assert torch.equal(g.get_value(x), s.get_value(x))


@pytest.mark.parametrize("mode", ["state", "state_separate"])
def test_sigma_equal_at_init(mode):
    """At step 0 both policies give (numerically) the same sigma for every state.

    With the spec'd head init (orthogonal, gain 0.01) each row of W has norm 0.01 and the
    tanh features satisfy ||h|| <= sqrt(64) = 8, so |log sigma(s) - log_std_init| <= 0.08
    for EVERY state; in practice it is far smaller. Both bounds are asserted.
    """
    g, s = build("global"), build(mode)
    x = states()
    sg, ss = g.dist(x)[0].stddev, s.dist(x)[0].stddev
    assert torch.allclose(sg, torch.full_like(sg, 0.5), atol=1e-7)
    dlog = (ss.log() - LOG_STD_INIT).abs()
    assert dlog.max() <= 0.01 * 8 + 1e-6  # hard bound, any state
    assert dlog.mean() < 0.02  # typical deviation ~1%
    assert torch.allclose(ss, sg, rtol=0.09)


def test_sigma_exactly_equal_with_zero_head_init():
    g, s = build("global"), build("state", logstd_head_init_std=0.0)
    x = states()
    assert torch.allclose(g.dist(x)[0].stddev, s.dist(x)[0].stddev, atol=1e-7)


def test_parameter_count_differs_by_one_head():
    n = lambda m: sum(p.numel() for p in m.parameters())
    g, s = build("global"), build("state")
    # global: act_dim log-std params; state: Linear(64, act_dim) = 64*act_dim + act_dim
    assert n(s) - n(g) == 64 * ACT
    assert sum(p.numel() for p in g.sigma_parameters()) == ACT
    assert sum(p.numel() for p in s.sigma_parameters()) == 64 * ACT + ACT


@pytest.mark.parametrize("mode", ["global", "state", "state_separate"])
def test_shapes(mode):
    a = build(mode)
    x = states(7)
    action, logprob, entropy, value, logstd = a.get_action_and_value(x)
    assert action.shape == (7, ACT) and logstd.shape == (7, ACT)
    assert logprob.shape == (7,) and entropy.shape == (7,) and value.shape == (7, 1)


@pytest.mark.parametrize("mode", ["global", "state", "state_separate"])
def test_log_prob_and_entropy_match_closed_form(mode):
    a = build(mode)
    with torch.no_grad():  # make sigma genuinely non-trivial
        for p in a.sigma_parameters():
            p.add_(torch.randn_like(p) * 0.3)
    x = states(64)
    act = torch.randn(64, ACT, generator=torch.Generator().manual_seed(1)) * 2  # incl. out-of-bounds actions
    _, logprob, entropy, _, logstd = a.get_action_and_value(x, act)
    dist, _ = a.dist(x)
    mu, sigma = dist.mean, logstd.exp()
    manual_lp = (-0.5 * ((act - mu) / sigma) ** 2 - logstd - 0.5 * math.log(2 * math.pi)).sum(1)
    manual_ent = (0.5 + 0.5 * math.log(2 * math.pi) + logstd).sum(1)
    assert torch.allclose(logprob, manual_lp, atol=1e-5)
    assert torch.allclose(entropy, manual_ent, atol=1e-5)
    # passing an action must return that action untouched (log-prob of the unclipped sample)
    assert torch.equal(a.get_action_and_value(x, act)[0], act)


@pytest.mark.parametrize("mode", ["global", "state", "state_separate"])
@pytest.mark.parametrize("value,bound", [(50.0, 2.0), (-50.0, -5.0)])
def test_logstd_clamp(mode, value, bound):
    a = build(mode)
    with torch.no_grad():
        if mode == "global":
            a.actor_logstd.fill_(value)
        else:
            a.actor_logstd_head.bias.fill_(value)
    dist, logstd = a.dist(states(32))
    assert torch.all(logstd == bound)
    assert torch.allclose(dist.stddev, torch.full_like(logstd, math.exp(bound)))
    assert torch.isfinite(dist.log_prob(torch.zeros(32, ACT))).all()


@pytest.mark.parametrize("mode", ["global", "state"])
def test_non_finite_output_raises(mode):
    a = build(mode)
    with torch.no_grad():
        a.actor_mean.bias.fill_(float("nan"))
    with pytest.raises(NonFiniteError):
        a.dist(states(4))


def test_state_mode_sigma_can_depend_on_state_but_global_cannot():
    g, s = build("global"), build("state")
    with torch.no_grad():
        s.actor_logstd_head.weight.normal_(0, 0.5)
        g.actor_logstd.add_(0.3)
    x = states(256)
    assert g.dist(x)[1].std(dim=0).max() == 0
    assert s.dist(x)[1].std(dim=0).min() > 0.1


def test_default_log_std_init_is_log_half():
    from ppo import Args

    assert np.isclose(Args().log_std_init, math.log(0.5))
    assert Args().logstd_min == -5.0 and Args().logstd_max == 2.0


def test_state_detach_blocks_sigma_gradient_into_trunk():
    """'state_detach' has the same parameters and init as 'state', but a loss on log sigma
    alone leaves the shared trunk untouched (isolates gradient interference)."""
    s, d = build("state"), build("state_detach")
    for (n1, p1), (n2, p2) in zip(s.named_parameters(), d.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)
    x = states(64)
    for agent, expect_trunk_grad in ((s, True), (d, False)):
        agent.zero_grad()
        _, logstd = agent.mean_and_logstd(x)
        logstd.sum().backward()
        trunk_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor_trunk.parameters())
        assert trunk_has_grad == expect_trunk_grad
        assert agent.actor_logstd_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["global", "state", "state_detach", "state_separate"])
def test_gradient_diagnostics_are_read_only(mode):
    from types import SimpleNamespace

    from ppo import gradient_diagnostics

    a = build(mode)
    n = 512
    x = states(n)
    with torch.no_grad():
        act, logp, _, _, _ = a.get_action_and_value(x)
    adv = torch.randn(n, generator=torch.Generator().manual_seed(3))
    args = SimpleNamespace(grad_diag_minibatches=8, minibatch_size=32, norm_adv=True, logstd_min=-5.0,
                           logstd_max=2.0, ent_coef=0.0)
    before = {k: v.clone() for k, v in a.state_dict().items()}
    torch.manual_seed(11)
    expected = torch.rand(3)
    torch.manual_seed(11)
    out = gradient_diagnostics(a, x, act, logp, adv, args, np.random.default_rng(0))
    assert torch.equal(torch.rand(3), expected)  # torch RNG untouched
    assert all(p.grad is None for p in a.parameters())  # nothing written to .grad
    assert all(torch.equal(before[k], v) for k, v in a.state_dict().items())
    assert np.isfinite(out["gsnr_mean_head"]) and np.isfinite(out["gsnr_sigma"])
    if mode == "state":
        assert -1.0 <= out["trunk_cos_sigma_vs_mean"] <= 1.0
    else:
        assert np.isnan(out["trunk_cos_sigma_vs_mean"])


def test_gradient_snr_is_about_one_for_pure_noise_and_large_for_signal():
    from ppo import gradient_snr

    g = torch.Generator().manual_seed(0)
    noise = [torch.randn(50, generator=g) for _ in range(16)]
    signal = [torch.ones(50) * 3 + torch.randn(50, generator=g) for _ in range(16)]
    assert 0.3 < gradient_snr(noise) < 3.0
    assert gradient_snr(signal) > 100


def test_tuning_helpers():
    """scripts/tune.py: sampled params map onto ppo.py Args, known-good configs are valid, scores pad."""
    import math as _m

    from ppo import Args
    from scripts.tune import GAMMAS, NUM_STEPS, final_score, known_good, to_args

    for env in ("parking-v0", "Pendulum-v1", "simglucose-adolescent1-v0"):
        for p in known_good(env):
            args = to_args(env, p)
            assert set(args) <= set(vars(Args())), set(args) - set(vars(Args()))
            assert _m.isclose(args["log_std_init"], _m.log(p["sigma_init"]))
            assert p["gamma"] in GAMMAS[env] and p["num_steps"] in NUM_STEPS[env]
    # spec default reproduces the current shared config exactly
    spec = to_args("parking-v0", known_good("parking-v0")[0])
    assert spec["num_minibatches"] == Args().num_minibatches and spec["learning_rate"] == Args().learning_rate
    assert final_score(np.array([0.0, 1.0, 2.0, 3.0]), 4) == 3.0
    assert final_score(np.array([0.0, 2.0]), 8) == 2.0  # stopped early -> padded with its last value
