### Pendulum-v1

IQM across seeds [95% stratified-bootstrap CI]; 'm − global' = IQM difference with a 95% bootstrap CI (seeds resampled per arm) and P(m > global) over bootstrap draws

| metric | global | state | state − global [95% CI] | P(state > global) |
|---|---|---|---|---|
| stoch_return / auc | -306.503 [-340.829, -288.019] | -284.553 [-300.868, -270.234] | +21.950 [-3.591, +58.771] | 0.95 |
| stoch_return / final | -184.878 [-194.759, -177.771] | -180.761 [-183.992, -177.604] | +4.117 [-3.931, +14.414] | 0.81 |
| det_return / auc | -294.488 [-327.358, -276.074] | -282.940 [-296.325, -270.808] | +11.549 [-11.460, +46.478] | 0.82 |
| det_return / final | -181.887 [-191.101, -176.295] | -181.315 [-185.117, -178.651] | +0.572 [-6.380, +10.233] | 0.54 |
| stoch_final_goal_dist / auc | 0.261 [0.203, 0.377] | 0.238 [0.198, 0.278] | -0.023 [-0.149, +0.046] | 0.26 |
| stoch_final_goal_dist / final | 0.050 [0.023, 0.127] | 0.037 [0.021, 0.068] | -0.014 [-0.090, +0.030] | 0.33 |
| noise cost (det − stoch return) / auc | 12.075 [8.390, 16.827] | 4.528 [-5.869, 12.521] | -7.547 [-19.101, +1.212] | 0.05 |
| noise cost (det − stoch return) / final | 2.366 [0.617, 4.002] | -0.999 [-3.444, 1.118] | -3.365 [-6.364, -0.676] | 0.01 |

Optimization difficulty (H2)

| diagnostic | global | state |
|---|---|---|
| n_seeds | 10 | 10 |
| n_nan_runs | 0 | 0 |
| final_return_std_across_seeds | 12.66 | 4.343 |
| final_return_iqr_across_seeds | 18.05 | 6.508 |
| auc_std_across_seeds | 42.4 | 19.22 |
| frac_seeds_never_learned (stoch_return_mean >= -300.0) | 0 | 0 |
| approx_kl_mean | 0.01925 | 0.06503 |
| approx_kl_max_minibatch_p99 | 1.025 | 5.915 |
| frac_updates_kl_gt_0.02 | 0.3098 | 0.8066 |
| clipfrac_mean | 0.07547 | 0.17 |
| explained_variance_final | 0.9906 | 0.9993 |
| logstd_mean_final | -0.6168 | -1.737 |
| logstd_min_final | -0.6168 | -2.602 |
| frac_seeds_sigma_hit_floor | 0 | 0 |
| logstd_frac_at_min_final | 0 | 0 |
| grad_norm_sigma_over_mean_head | 0.1378 | 0.7263 |
| gsnr_mean_head_mean_last20pct | 1.008 | 1.307 |
| gsnr_sigma_bias_mean_last20pct | 1.164 | 1.349 |
| gsnr_sigma_weight_mean_last20pct | nan | 1.159 |
| gsnr_actor_trunk_mean_last20pct | 1.053 | 1.209 |
| trunk_cos_sigma_vs_mean_median_last20pct | nan | 0.004278 |
| trunk_gnorm_sigma_over_mean_median_last20pct | nan | 0.459 |
| mean_outside_bounds_frac_median_last20pct | 0.09592 | 0.1272 |
| action_clipped_frac_median_last20pct | 0.1336 | 0.1332 |
| train_wall_clock_min_mean | 8.621 | 6.484 |

### parking-v0

IQM across seeds [95% stratified-bootstrap CI]; 'm − global' = IQM difference with a 95% bootstrap CI (seeds resampled per arm) and P(m > global) over bootstrap draws

| metric | global | state | state − global [95% CI] | P(state > global) |
|---|---|---|---|---|
| stoch_return / auc | -10.142 [-11.035, -9.185] | -10.844 [-11.469, -9.554] | -0.702 [-1.895, +0.891] | 0.19 |
| stoch_return / final | -8.660 [-10.232, -7.165] | -9.504 [-10.739, -7.782] | -0.844 [-2.885, +1.521] | 0.25 |
| det_return / auc | -10.783 [-11.846, -9.703] | -11.238 [-12.003, -9.972] | -0.455 [-1.797, +1.222] | 0.31 |
| det_return / final | -8.553 [-10.113, -6.943] | -9.423 [-10.525, -7.735] | -0.869 [-2.850, +1.450] | 0.24 |
| stoch_success_rate / auc | 0.274 [0.117, 0.428] | 0.210 [0.093, 0.422] | -0.064 [-0.259, +0.203] | 0.34 |
| stoch_success_rate / final | 0.483 [0.200, 0.750] | 0.364 [0.161, 0.653] | -0.119 [-0.472, +0.289] | 0.29 |
| det_success_rate / auc | 0.293 [0.129, 0.451] | 0.220 [0.099, 0.430] | -0.073 [-0.277, +0.199] | 0.32 |
| det_success_rate / final | 0.500 [0.219, 0.781] | 0.356 [0.169, 0.636] | -0.144 [-0.486, +0.256] | 0.25 |
| stoch_crash_rate / auc | 0.730 [0.578, 0.880] | 0.792 [0.582, 0.908] | +0.062 [-0.202, +0.254] | 0.66 |
| stoch_crash_rate / final | 0.522 [0.264, 0.803] | 0.636 [0.347, 0.839] | +0.114 [-0.289, +0.458] | 0.70 |
| stoch_final_goal_dist / auc | 12.900 [8.457, 18.094] | 14.681 [9.320, 19.708] | +1.782 [-5.799, +8.774] | 0.68 |
| stoch_final_goal_dist / final | 9.538 [3.837, 16.308] | 12.714 [6.672, 18.468] | +3.176 [-5.998, +11.360] | 0.74 |
| noise cost (det − stoch return) / auc | -0.700 [-0.895, -0.539] | -0.459 [-0.639, -0.319] | +0.240 [-0.002, +0.478] | 0.97 |
| noise cost (det − stoch return) / final | 0.137 [-0.005, 0.300] | 0.171 [0.010, 0.246] | +0.034 [-0.188, +0.199] | 0.61 |

Optimization difficulty (H2)

| diagnostic | global | state |
|---|---|---|
| n_seeds | 10 | 10 |
| n_nan_runs | 0 | 0 |
| final_return_std_across_seeds | 1.82 | 1.845 |
| final_return_iqr_across_seeds | 3.694 | 3.093 |
| auc_std_across_seeds | 1.163 | 1.29 |
| frac_seeds_never_learned (stoch_success_rate >= 0.5) | 0.4 | 0.6 |
| approx_kl_mean | 0.03505 | 0.03639 |
| approx_kl_max_minibatch_p99 | 0.2491 | 0.4017 |
| frac_updates_kl_gt_0.02 | 0.8115 | 0.8111 |
| clipfrac_mean | 0.1565 | 0.1349 |
| explained_variance_final | 0.9392 | 0.9479 |
| logstd_mean_final | -1.076 | 0.4579 |
| logstd_min_final | -1.722 | -3.036 |
| frac_seeds_sigma_hit_floor | 0 | 0 |
| logstd_frac_at_min_final | 0 | 0 |
| grad_norm_sigma_over_mean_head | 0.05939 | 0.7387 |
| gsnr_mean_head_mean_last20pct | 1.126 | 1.153 |
| gsnr_sigma_bias_mean_last20pct | 1.273 | 1.147 |
| gsnr_sigma_weight_mean_last20pct | nan | 1.039 |
| gsnr_actor_trunk_mean_last20pct | 1.237 | 1.152 |
| trunk_cos_sigma_vs_mean_median_last20pct | nan | 0.002726 |
| trunk_gnorm_sigma_over_mean_median_last20pct | nan | 0.1746 |
| mean_outside_bounds_frac_median_last20pct | 0.6492 | 0.8331 |
| action_clipped_frac_median_last20pct | 0.6473 | 0.8398 |
| train_wall_clock_min_mean | 279.7 | 271.9 |

### simglucose-adolescent1-v0

IQM across seeds [95% stratified-bootstrap CI]; 'm − global' = IQM difference with a 95% bootstrap CI (seeds resampled per arm) and P(m > global) over bootstrap draws

| metric | global | state | state − global [95% CI] | P(state > global) |
|---|---|---|---|---|
| stoch_return / auc | 454.379 [448.585, 459.114] | 440.856 [428.794, 448.853] | -13.523 [-25.998, -3.046] | 0.00 |
| stoch_return / final | 475.125 [474.658, 475.426] | 474.357 [473.860, 474.705] | -0.769 [-1.351, -0.173] | 0.01 |
| det_return / auc | 445.074 [438.157, 453.005] | 303.471 [256.596, 341.428] | -141.603 [-189.099, -103.467] | 0.00 |
| det_return / final | 474.883 [474.352, 475.360] | 434.096 [336.116, 469.544] | -40.786 [-139.076, -5.391] | 0.00 |
| stoch_success_rate / auc | 0.828 [0.812, 0.847] | 0.699 [0.612, 0.731] | -0.129 [-0.218, -0.091] | 0.00 |
| stoch_success_rate / final | 0.964 [0.953, 0.983] | 0.989 [0.964, 1.000] | +0.025 [-0.006, +0.044] | 0.93 |
| det_success_rate / auc | 0.804 [0.772, 0.842] | 0.216 [0.045, 0.399] | -0.588 [-0.760, -0.406] | 0.00 |
| det_success_rate / final | 0.989 [0.961, 1.000] | 0.697 [0.261, 1.000] | -0.292 [-0.731, +0.011] | 0.04 |
| stoch_crash_rate / auc | 0.100 [0.077, 0.126] | 0.218 [0.175, 0.288] | +0.117 [+0.067, +0.191] | 1.00 |
| stoch_crash_rate / final | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] | 0.00 |
| stoch_final_goal_dist / auc | 13.663 [10.810, 17.962] | 19.597 [16.121, 31.807] | +5.934 [+0.558, +18.165] | 0.98 |
| stoch_final_goal_dist / final | 0.737 [0.623, 0.902] | 0.655 [0.554, 0.850] | -0.081 [-0.287, +0.143] | 0.22 |
| stoch_tir / auc | 0.935 [0.919, 0.948] | 0.873 [0.838, 0.900] | -0.062 [-0.099, -0.030] | 0.00 |
| stoch_tir / final | 0.997 [0.995, 0.998] | 0.997 [0.995, 0.999] | +0.001 [-0.002, +0.003] | 0.65 |
| stoch_tbr / auc | 0.059 [0.047, 0.073] | 0.121 [0.096, 0.153] | +0.061 [+0.033, +0.096] | 1.00 |
| stoch_tbr / final | 0.003 [0.002, 0.004] | 0.001 [0.000, 0.003] | -0.002 [-0.004, +0.001] | 0.10 |
| noise cost (det − stoch return) / auc | -6.975 [-13.217, -1.563] | -142.098 [-181.055, -100.665] | -135.123 [-173.948, -93.945] | 0.00 |
| noise cost (det − stoch return) / final | -0.073 [-0.501, 0.211] | -40.054 [-138.705, -4.911] | -39.981 [-138.270, -4.839] | 0.00 |

Optimization difficulty (H2)

| diagnostic | global | state |
|---|---|---|
| n_seeds | 10 | 10 |
| n_nan_runs | 0 | 0 |
| final_return_std_across_seeds | 0.5388 | 0.6241 |
| final_return_iqr_across_seeds | 0.6578 | 0.6889 |
| auc_std_across_seeds | 7.382 | 13.16 |
| frac_seeds_never_learned (stoch_success_rate >= 0.5) | 0 | 0 |
| approx_kl_mean | 0.003151 | 0.03425 |
| approx_kl_max_minibatch_p99 | 0.02522 | 5.12 |
| frac_updates_kl_gt_0.02 | 0 | 0.7348 |
| clipfrac_mean | 0.03547 | 0.1364 |
| explained_variance_final | 0.4101 | 0.3387 |
| logstd_mean_final | -0.4185 | 1.495 |
| logstd_min_final | -0.4185 | -3.151 |
| frac_seeds_sigma_hit_floor | 0 | 0.1 |
| logstd_frac_at_min_final | 0 | 0 |
| grad_norm_sigma_over_mean_head | 0.1291 | 1.583 |
| gsnr_mean_head_mean_last20pct | 0.9959 | 0.9791 |
| gsnr_sigma_bias_mean_last20pct | 1.042 | 0.9589 |
| gsnr_sigma_weight_mean_last20pct | nan | 0.9969 |
| gsnr_actor_trunk_mean_last20pct | 1.013 | 1.004 |
| trunk_cos_sigma_vs_mean_median_last20pct | nan | -0.002358 |
| trunk_gnorm_sigma_over_mean_median_last20pct | nan | 0.9666 |
| mean_outside_bounds_frac_median_last20pct | 0.6128 | 0.935 |
| action_clipped_frac_median_last20pct | 0.6159 | 0.9182 |
| train_wall_clock_min_mean | 74.95 | 71.64 |
