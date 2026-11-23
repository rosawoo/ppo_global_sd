### Pendulum-v1

IQM across seeds [95% stratified-bootstrap CI]; 'm − global' = IQM difference with a 95% bootstrap CI (seeds resampled per arm) and P(m > global) over bootstrap draws

| metric | global | state | state − global [95% CI] | P(state > global) |
|---|---|---|---|---|
| stoch_return / auc | -359.589 [-371.443, -348.261] | -651.620 [-768.116, -481.322] | -292.030 [-407.176, -128.881] | 0.00 |
| stoch_return / final | -193.626 [-203.473, -186.051] | -422.090 [-613.822, -245.014] | -228.465 [-420.938, -56.047] | 0.00 |
| det_return / auc | -351.715 [-365.003, -343.693] | -634.417 [-750.300, -483.736] | -282.702 [-397.470, -127.887] | 0.00 |
| det_return / final | -190.286 [-199.362, -180.532] | -414.847 [-605.925, -254.111] | -224.561 [-417.405, -59.598] | 0.00 |
| stoch_final_goal_dist / auc | 0.296 [0.277, 0.345] | 0.712 [0.499, 0.879] | +0.416 [+0.206, +0.583] | 1.00 |
| stoch_final_goal_dist / final | 0.065 [0.022, 0.128] | 0.333 [0.097, 0.576] | +0.268 [+0.024, +0.505] | 0.99 |
| noise cost (det − stoch return) / auc | 6.878 [2.618, 9.748] | 15.149 [5.188, 21.318] | +8.271 [-1.876, +15.553] | 0.95 |
| noise cost (det − stoch return) / final | 5.485 [0.411, 8.174] | 2.029 [-0.190, 14.254] | -3.456 [-7.261, +9.779] | 0.30 |

Optimization difficulty (H2)

| diagnostic | global | state |
|---|---|---|
| n_seeds | 6 | 6 |
| n_nan_runs | 0 | 0 |
| final_return_std_across_seeds | 9.806 | 196.7 |
| final_return_iqr_across_seeds | 14.85 | 333.1 |
| auc_std_across_seeds | 13.78 | 153.9 |
| frac_seeds_never_learned (stoch_return_mean >= -300.0) | 0 | 0.5 |
| approx_kl_mean | 0.005375 | 0.01283 |
| approx_kl_max_minibatch_p99 | 0.05576 | 0.1842 |
| frac_updates_kl_gt_0.02 | 0 | 0.08538 |
| clipfrac_mean | 0.06163 | 0.1345 |
| explained_variance_final | 0.9885 | 0.9849 |
| logstd_mean_final | -1.545 | -3.366 |
| logstd_min_final | -1.545 | -3.959 |
| frac_seeds_sigma_hit_floor | 0 | 0 |
| logstd_frac_at_min_final | 0 | 0 |
| grad_norm_sigma_over_mean_head | 0.0537 | 0.09312 |
| gsnr_mean_head_mean_last20pct | 1.017 | 1.405 |
| gsnr_sigma_bias_mean_last20pct | 1.418 | 1.078 |
| gsnr_sigma_weight_mean_last20pct | nan | 1.108 |
| gsnr_actor_trunk_mean_last20pct | 1.049 | 1.578 |
| trunk_cos_sigma_vs_mean_median_last20pct | nan | -0.003907 |
| trunk_gnorm_sigma_over_mean_median_last20pct | nan | 0.05688 |
| mean_outside_bounds_frac_median_last20pct | 0 | 0 |
| action_clipped_frac_median_last20pct | 0 | 0 |
| train_wall_clock_min_mean | 12.3 | 12.6 |
