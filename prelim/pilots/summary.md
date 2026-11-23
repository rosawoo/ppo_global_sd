### parking-v0

IQM across seeds [95% stratified-bootstrap CI]; 'm − global' = IQM difference with a 95% bootstrap CI (seeds resampled per arm) and P(m > global) over bootstrap draws

| metric | global | state | state − global [95% CI] | P(state > global) |
|---|---|---|---|---|
| stoch_return / auc | -14.175 [-14.175, -14.175] | -13.937 [-13.937, -13.937] | +0.238 [+0.238, +0.238] | 1.00 |
| stoch_return / final | -11.443 [-11.443, -11.443] | -11.513 [-11.513, -11.513] | -0.069 [-0.069, -0.069] | 0.00 |
| det_return / auc | -18.512 [-18.512, -18.512] | -18.739 [-18.739, -18.739] | -0.227 [-0.227, -0.227] | 0.00 |
| det_return / final | -11.361 [-11.361, -11.361] | -11.460 [-11.460, -11.460] | -0.098 [-0.098, -0.098] | 0.00 |
| stoch_success_rate / auc | 0.000 [0.000, 0.000] | 0.008 [0.008, 0.008] | +0.008 [+0.008, +0.008] | 1.00 |
| stoch_success_rate / final | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] | 0.00 |
| det_success_rate / auc | 0.000 [0.000, 0.000] | 0.004 [0.004, 0.004] | +0.004 [+0.004, +0.004] | 1.00 |
| det_success_rate / final | 0.000 [0.000, 0.000] | 0.017 [0.017, 0.017] | +0.017 [+0.017, +0.017] | 1.00 |
| stoch_crash_rate / auc | 1.000 [1.000, 1.000] | 0.992 [0.992, 0.992] | -0.008 [-0.008, -0.008] | 0.00 |
| stoch_crash_rate / final | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | +0.000 [+0.000, +0.000] | 0.00 |
| stoch_final_goal_dist / auc | 20.055 [20.055, 20.055] | 20.883 [20.883, 20.883] | +0.828 [+0.828, +0.828] | 1.00 |
| stoch_final_goal_dist / final | 18.426 [18.426, 18.426] | 18.027 [18.027, 18.027] | -0.399 [-0.399, -0.399] | 0.00 |
| noise cost (det − stoch return) / auc | -4.337 [-4.337, -4.337] | -4.801 [-4.801, -4.801] | -0.465 [-0.465, -0.465] | 0.00 |
| noise cost (det − stoch return) / final | 0.082 [0.082, 0.082] | 0.053 [0.053, 0.053] | -0.029 [-0.029, -0.029] | 0.00 |

Optimization difficulty (H2)

| diagnostic | global | state |
|---|---|---|
| n_seeds | 1 | 1 |
| n_nan_runs | 0 | 0 |
| final_return_std_across_seeds | nan | nan |
| final_return_iqr_across_seeds | 0 | 0 |
| auc_std_across_seeds | nan | nan |
| frac_seeds_never_learned (stoch_success_rate >= 0.5) | 1 | 1 |
| approx_kl_mean | 0.00737 | 0.01189 |
| approx_kl_max_minibatch_p99 | 0.03117 | 0.08173 |
| frac_updates_kl_gt_0.02 | 0 | 0.006849 |
| clipfrac_mean | 0.09538 | 0.139 |
| explained_variance_final | 0.9739 | 0.9762 |
| logstd_mean_final | -1.571 | -2.221 |
| logstd_min_final | -1.605 | -3.244 |
| frac_seeds_sigma_hit_floor | 0 | 0 |
| logstd_frac_at_min_final | 0 | 0 |
| grad_norm_sigma_over_mean_head | 0.05728 | 0.2065 |
| gsnr_mean_head_mean_last20pct | 0.9101 | 1.014 |
| gsnr_sigma_bias_mean_last20pct | 1.046 | 0.9538 |
| gsnr_sigma_weight_mean_last20pct | nan | 1.054 |
| gsnr_actor_trunk_mean_last20pct | 1.129 | 1.023 |
| trunk_cos_sigma_vs_mean_median_last20pct | nan | 0.00149 |
| trunk_gnorm_sigma_over_mean_median_last20pct | nan | 0.1799 |
| mean_outside_bounds_frac_median_last20pct | 0.5571 | 0.45 |
| action_clipped_frac_median_last20pct | 0.536 | 0.4459 |
| train_wall_clock_min_mean | 35.41 | 35.58 |

### simglucose-adolescent1-v0

IQM across seeds [95% stratified-bootstrap CI]; 'm − global' = IQM difference with a 95% bootstrap CI (seeds resampled per arm) and P(m > global) over bootstrap draws

| metric | global | state | state − global [95% CI] | P(state > global) |
|---|---|---|---|---|
| stoch_return / auc | 368.722 [368.722, 368.722] | 219.771 [219.771, 219.771] | -148.952 [-148.952, -148.952] | 0.00 |
| stoch_return / final | 444.525 [444.525, 444.525] | 264.239 [264.239, 264.239] | -180.287 [-180.287, -180.287] | 0.00 |
| det_return / auc | 386.739 [386.739, 386.739] | 226.279 [226.279, 226.279] | -160.460 [-160.460, -160.460] | 0.00 |
| det_return / final | 448.834 [448.834, 448.834] | 279.519 [279.519, 279.519] | -169.314 [-169.314, -169.314] | 0.00 |
| stoch_success_rate / auc | 0.004 [0.004, 0.004] | 0.000 [0.000, 0.000] | -0.004 [-0.004, -0.004] | 0.00 |
| stoch_success_rate / final | 0.017 [0.017, 0.017] | 0.000 [0.000, 0.000] | -0.017 [-0.017, -0.017] | 0.00 |
| det_success_rate / auc | 0.053 [0.053, 0.053] | 0.000 [0.000, 0.000] | -0.053 [-0.053, -0.053] | 0.00 |
| det_success_rate / final | 0.150 [0.150, 0.150] | 0.000 [0.000, 0.000] | -0.150 [-0.150, -0.150] | 0.00 |
| stoch_crash_rate / auc | 0.910 [0.910, 0.910] | 1.000 [1.000, 1.000] | +0.090 [+0.090, +0.090] | 1.00 |
| stoch_crash_rate / final | 0.817 [0.817, 0.817] | 1.000 [1.000, 1.000] | +0.183 [+0.183, +0.183] | 1.00 |
| stoch_final_goal_dist / auc | 68.518 [68.518, 68.518] | 175.093 [175.093, 175.093] | +106.575 [+106.575, +106.575] | 1.00 |
| stoch_final_goal_dist / final | 10.929 [10.929, 10.929] | 140.180 [140.180, 140.180] | +129.251 [+129.251, +129.251] | 1.00 |
| stoch_tir / auc | 0.552 [0.552, 0.552] | 0.415 [0.415, 0.415] | -0.137 [-0.137, -0.137] | 0.00 |
| stoch_tir / final | 0.686 [0.686, 0.686] | 0.379 [0.379, 0.379] | -0.307 [-0.307, -0.307] | 0.00 |
| stoch_tbr / auc | 0.415 [0.415, 0.415] | 0.585 [0.585, 0.585] | +0.171 [+0.171, +0.171] | 1.00 |
| stoch_tbr / final | 0.294 [0.294, 0.294] | 0.621 [0.621, 0.621] | +0.327 [+0.327, +0.327] | 1.00 |
| noise cost (det − stoch return) / auc | 18.016 [18.016, 18.016] | 6.508 [6.508, 6.508] | -11.508 [-11.508, -11.508] | 0.00 |
| noise cost (det − stoch return) / final | 4.308 [4.308, 4.308] | 15.281 [15.281, 15.281] | +10.972 [+10.972, +10.972] | 1.00 |

Optimization difficulty (H2)

| diagnostic | global | state |
|---|---|---|
| n_seeds | 1 | 1 |
| n_nan_runs | 0 | 0 |
| final_return_std_across_seeds | nan | nan |
| final_return_iqr_across_seeds | 0 | 0 |
| auc_std_across_seeds | nan | nan |
| frac_seeds_never_learned (stoch_success_rate >= 0.5) | 1 | 1 |
| approx_kl_mean | 0.0072 | 0.02493 |
| approx_kl_max_minibatch_p99 | 0.05741 | 0.8306 |
| frac_updates_kl_gt_0.02 | 0 | 0.5026 |
| clipfrac_mean | 0.08959 | 0.12 |
| explained_variance_final | 0.9077 | 0.9657 |
| logstd_mean_final | -2.076 | -1.989 |
| logstd_min_final | -2.076 | -5 |
| frac_seeds_sigma_hit_floor | 0 | 1 |
| logstd_frac_at_min_final | 0 | 0.08789 |
| grad_norm_sigma_over_mean_head | 0.03457 | 0.08116 |
| gsnr_mean_head_mean_last20pct | 0.9739 | 1.292 |
| gsnr_sigma_bias_mean_last20pct | 0.6971 | 0.9196 |
| gsnr_sigma_weight_mean_last20pct | nan | 0.9231 |
| gsnr_actor_trunk_mean_last20pct | 1.022 | 1.154 |
| trunk_cos_sigma_vs_mean_median_last20pct | nan | 0.003822 |
| trunk_gnorm_sigma_over_mean_median_last20pct | nan | 0.1489 |
| mean_outside_bounds_frac_median_last20pct | 0 | 0 |
| action_clipped_frac_median_last20pct | 0 | 0 |
| train_wall_clock_min_mean | 54.2 | 52.12 |
