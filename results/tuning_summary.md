# Hyper-parameter tuning (paired: both σ arms per trial)

Scores: mean over the last 25% of evaluations, 2 tuning seeds per arm, held-out eval seeds. Difference = score(state) − score(global) at the same hyper-parameters. 'random configs' = the TPE start-up trials (uniform samples of the search space); 'all' also includes TPE's model-guided trials, which concentrate where the mean of the two arms is high.

## Pendulum-v1

trials: 64 (25 complete, 39 pruned, 0 failed, 0 running)

| configs | n | P(state > global) | mean diff | median diff | Wilcoxon p |
|---|---|---|---|---|---|
| all complete | 25 | 0.56 | -15.21 | +2.915 | 0.692 |
| random configs | 7 | 0.57 | -16.43 | +2.536 | 0.812 |
| top quartile by shared objective | 7 | 0.57 | -2.666 | +2.536 | 0.578 |

| config | trial | objective | score global | score state | learning_rate | gamma | gae_lambda | clip_coef | ent_coef | update_epochs | num_steps | num_minibatches | log_std_init |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shared best | 53 | -210.8 | -205 | -216.6 | 0.0014 | 0.98 | 0.99 | 0.3 | 0.01 | 10 | 1024 | 64 | 0 |
| best for global | 53 | -210.8 | -205 | -216.6 | 0.0014 | 0.98 | 0.99 | 0.3 | 0.01 | 10 | 1024 | 64 | 0 |
| best for state | 51 | -216.7 | -219.5 | -213.9 | 0.0016 | 0.95 | 0.9 | 0.3 | 0.001 | 10 | 1024 | 32 | 0 |

Hyper-parameter importance (fANOVA) for each arm's score:

| param | global | state |
|---|---|---|
| learning_rate | 0.55 | 0.55 |
| num_steps | 0.11 | 0.11 |
| update_epochs | 0.11 | 0.11 |
| sigma_init | 0.08 | 0.08 |
| ent_coef | 0.04 | 0.04 |
| gamma | 0.04 | 0.04 |
| minibatch_size | 0.03 | 0.03 |
| clip_coef | 0.03 | 0.03 |
| gae_lambda | 0.01 | 0.01 |

## parking-v0

trials: 32 (21 complete, 11 pruned, 0 failed, 0 running)

| configs | n | P(state > global) | mean diff | median diff | Wilcoxon p |
|---|---|---|---|---|---|
| all complete | 21 | 0.57 | +0.09591 | +0.02267 | 0.338 |
| random configs | 7 | 0.43 | +0.01692 | -7.864e-05 | 0.938 |
| top quartile by shared objective | 6 | 1.00 | +0.1935 | +0.09366 | 0.0312 |

| config | trial | objective | score global | score state | learning_rate | gamma | gae_lambda | clip_coef | ent_coef | update_epochs | num_steps | num_minibatches | log_std_init |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shared best | 30 | 0.9218 | 0.9105 | 0.9331 | 0.00243 | 0.98 | 0.99 | 0.3 | 0.01 | 5 | 512 | 16 | 0 |
| best for global | 30 | 0.9218 | 0.9105 | 0.9331 | 0.00243 | 0.98 | 0.99 | 0.3 | 0.01 | 5 | 512 | 16 | 0 |
| best for state | 30 | 0.9218 | 0.9105 | 0.9331 | 0.00243 | 0.98 | 0.99 | 0.3 | 0.01 | 5 | 512 | 16 | 0 |

Hyper-parameter importance (fANOVA) for each arm's score:

| param | global | state |
|---|---|---|
| learning_rate | 0.29 | 0.43 |
| gamma | 0.14 | 0.18 |
| num_steps | 0.13 | 0.03 |
| ent_coef | 0.12 | 0.03 |
| gae_lambda | 0.11 | 0.06 |
| clip_coef | 0.10 | 0.10 |
| update_epochs | 0.06 | 0.09 |
| minibatch_size | 0.03 | 0.05 |
| sigma_init | 0.01 | 0.03 |

## simglucose-adolescent1-v0

trials: 40 (26 complete, 14 pruned, 0 failed, 0 running)

| configs | n | P(state > global) | mean diff | median diff | Wilcoxon p |
|---|---|---|---|---|---|
| all complete | 26 | 0.50 | -15.06 | -0.6362 | 0.617 |
| random configs | 8 | 0.50 | -21.17 | -6.398 | 0.641 |
| top quartile by shared objective | 7 | 0.57 | -0.2159 | +0.04853 | 0.812 |

| config | trial | objective | score global | score state | learning_rate | gamma | gae_lambda | clip_coef | ent_coef | update_epochs | num_steps | num_minibatches | log_std_init |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shared best | 33 | 473.8 | 473.8 | 473.9 | 0.000318 | 0.99 | 0.99 | 0.2 | 0.001 | 5 | 256 | 32 | 0 |
| best for global | 19 | 473.3 | 474.5 | 472 | 0.000529 | 0.99 | 0.99 | 0.2 | 0.001 | 3 | 256 | 32 | 0 |
| best for state | 31 | 473.3 | 472.4 | 474.3 | 0.00274 | 0.99 | 0.99 | 0.2 | 0.001 | 3 | 512 | 64 | 0 |

Hyper-parameter importance (fANOVA) for each arm's score:

| param | global | state |
|---|---|---|
| sigma_init | 0.24 | 0.06 |
| gamma | 0.23 | 0.05 |
| minibatch_size | 0.15 | 0.03 |
| update_epochs | 0.12 | 0.24 |
| clip_coef | 0.11 | 0.12 |
| learning_rate | 0.07 | 0.38 |
| gae_lambda | 0.05 | 0.10 |
| ent_coef | 0.02 | 0.03 |
| num_steps | 0.00 | 0.00 |

