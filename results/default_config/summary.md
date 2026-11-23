### Pendulum-v1

IQM across seeds [95% stratified-bootstrap CI]; 'm − global' = IQM difference with a 95% bootstrap CI (seeds resampled per arm) and P(m > global) over bootstrap draws

| metric | global | state | state − global [95% CI] | P(state > global) |
|---|---|---|---|---|
| stoch_return / auc | -366.716 [-387.439, -356.063] | -628.186 [-724.400, -509.635] | -261.470 [-358.221, -137.522] | 0.00 |
| stoch_return / final | -193.176 [-201.630, -186.018] | -377.956 [-539.868, -243.667] | -184.780 [-342.354, -47.612] | 0.00 |
| det_return / auc | -361.522 [-381.909, -350.348] | -612.017 [-709.141, -501.084] | -250.495 [-344.134, -135.979] | 0.00 |
| det_return / final | -191.820 [-199.474, -183.170] | -369.429 [-528.701, -241.091] | -177.609 [-334.516, -46.815] | 0.00 |
| stoch_final_goal_dist / auc | 0.307 [0.279, 0.348] | 0.695 [0.514, 0.842] | +0.388 [+0.198, +0.534] | 1.00 |
| stoch_final_goal_dist / final | 0.057 [0.024, 0.102] | 0.294 [0.096, 0.497] | +0.236 [+0.033, +0.441] | 0.99 |
| noise cost (det − stoch return) / auc | 5.112 [2.369, 7.774] | 13.721 [5.583, 19.916] | +8.609 [+0.191, +15.590] | 0.98 |
| noise cost (det − stoch return) / final | 2.507 [-0.988, 6.414] | 3.916 [1.605, 11.692] | +1.408 [-3.129, +10.124] | 0.77 |

Optimization difficulty (H2)

| diagnostic | global | state |
|---|---|---|
| n_seeds | 10 | 10 |
| n_nan_runs | 0 | 0 |
| final_return_std_across_seeds | 9.868 | 179.9 |
| final_return_iqr_across_seeds | 15.23 | 287.2 |
| auc_std_across_seeds | 28.59 | 142.2 |
| frac_seeds_never_learned (stoch_return_mean >= -300.0) | 0 | 0.5 |
| approx_kl_mean | 0.005332 | 0.01254 |
| approx_kl_max_minibatch_p99 | 0.05578 | 0.1849 |
| frac_updates_kl_gt_0.02 | 0 | 0.07541 |
| clipfrac_mean | 0.0612 | 0.1323 |
| explained_variance_final | 0.9882 | 0.9853 |
| logstd_mean_final | -1.498 | -3.351 |
| logstd_min_final | -1.498 | -3.899 |
| frac_seeds_sigma_hit_floor | 0 | 0 |
| logstd_frac_at_min_final | 0 | 0 |
| grad_norm_sigma_over_mean_head | 0.05558 | 0.09749 |
| gsnr_mean_head_mean_last20pct | 1.078 | 1.5 |
| gsnr_sigma_bias_mean_last20pct | 1.317 | 1.096 |
| gsnr_sigma_weight_mean_last20pct | nan | 1.081 |
| gsnr_actor_trunk_mean_last20pct | 1.093 | 1.65 |
| trunk_cos_sigma_vs_mean_median_last20pct | nan | -0.0009718 |
| trunk_gnorm_sigma_over_mean_median_last20pct | nan | 0.05508 |
| mean_outside_bounds_frac_median_last20pct | 0.04912 | 0.04004 |
| action_clipped_frac_median_last20pct | 0.0499 | 0.04111 |
| train_wall_clock_min_mean | 9.452 | 10.06 |

### parking-v0

IQM across seeds [95% stratified-bootstrap CI]; 'm − global' = IQM difference with a 95% bootstrap CI (seeds resampled per arm) and P(m > global) over bootstrap draws

| metric | global | state | state − global [95% CI] | P(state > global) |
|---|---|---|---|---|
| stoch_return / auc | -11.454 [-11.652, -11.297] | -11.475 [-11.645, -11.274] | -0.021 [-0.259, +0.261] | 0.46 |
| stoch_return / final | -11.004 [-11.250, -10.843] | -11.053 [-11.218, -10.891] | -0.049 [-0.286, +0.249] | 0.38 |
| det_return / auc | -12.269 [-12.552, -12.073] | -12.188 [-12.412, -12.027] | +0.081 [-0.213, +0.418] | 0.71 |
| det_return / final | -11.068 [-11.314, -10.857] | -11.046 [-11.209, -10.882] | +0.022 [-0.245, +0.313] | 0.57 |
| stoch_success_rate / auc | 0.000 [0.000, 0.005] | 0.000 [0.000, 0.001] | +0.000 [-0.004, +0.001] | 0.22 |
| stoch_success_rate / final | 0.000 [0.000, 0.003] | 0.000 [0.000, 0.000] | +0.000 [-0.003, +0.000] | 0.00 |
| det_success_rate / auc | 0.001 [0.000, 0.005] | 0.000 [0.000, 0.000] | -0.001 [-0.005, +0.000] | 0.01 |
| det_success_rate / final | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] | 0.00 |
| stoch_crash_rate / auc | 1.000 [0.996, 1.000] | 1.000 [0.999, 1.000] | +0.000 [-0.001, +0.004] | 0.32 |
| stoch_crash_rate / final | 1.000 [0.997, 1.000] | 1.000 [1.000, 1.000] | +0.000 [+0.000, +0.003] | 0.07 |
| stoch_final_goal_dist / auc | 18.309 [15.470, 22.386] | 20.239 [17.043, 22.717] | +1.930 [-3.185, +5.806] | 0.76 |
| stoch_final_goal_dist / final | 17.750 [14.697, 21.937] | 19.442 [15.947, 22.485] | +1.692 [-3.677, +6.027] | 0.73 |
| noise cost (det − stoch return) / auc | -0.797 [-0.981, -0.678] | -0.762 [-0.851, -0.658] | +0.035 [-0.112, +0.239] | 0.69 |
| noise cost (det − stoch return) / final | -0.000 [-0.068, 0.002] | 0.003 [0.000, 0.009] | +0.004 [+0.000, +0.073] | 0.98 |

Optimization difficulty (H2)

| diagnostic | global | state |
|---|---|---|
| n_seeds | 10 | 10 |
| n_nan_runs | 0 | 0 |
| final_return_std_across_seeds | 0.2834 | 0.2312 |
| final_return_iqr_across_seeds | 0.4127 | 0.2883 |
| auc_std_across_seeds | 0.2371 | 0.2585 |
| frac_seeds_never_learned (stoch_success_rate >= 0.5) | 1 | 1 |
| approx_kl_mean | 0.07052 | 0.02262 |
| approx_kl_max_minibatch_p99 | 1.56 | 0.3798 |
| frac_updates_kl_gt_0.02 | 0.5301 | 0.6966 |
| clipfrac_mean | 0.2212 | 0.2156 |
| explained_variance_final | 0.9823 | 0.9863 |
| logstd_mean_final | -3.361 | -3.427 |
| logstd_min_final | -3.426 | -4.941 |
| frac_seeds_sigma_hit_floor | 0 | 0.8 |
| logstd_frac_at_min_final | 0 | 0.008081 |
| grad_norm_sigma_over_mean_head | 0.01098 | 0.08077 |
| gsnr_mean_head_mean_last20pct | 1.023 | 1.018 |
| gsnr_sigma_bias_mean_last20pct | 1.06 | 1.096 |
| gsnr_sigma_weight_mean_last20pct | nan | 1.023 |
| gsnr_actor_trunk_mean_last20pct | 1.024 | 1.018 |
| trunk_cos_sigma_vs_mean_median_last20pct | nan | 0.001204 |
| trunk_gnorm_sigma_over_mean_median_last20pct | nan | 0.06648 |
| mean_outside_bounds_frac_median_last20pct | 0.5664 | 0.6099 |
| action_clipped_frac_median_last20pct | 0.5656 | 0.6071 |
| train_wall_clock_min_mean | 303.5 | 302.5 |

### simglucose-adolescent1-v0

IQM across seeds [95% stratified-bootstrap CI]; 'm − global' = IQM difference with a 95% bootstrap CI (seeds resampled per arm) and P(m > global) over bootstrap draws

| metric | global | state | state − global [95% CI] | P(state > global) |
|---|---|---|---|---|
| stoch_return / auc | 307.932 [253.898, 367.277] | 276.398 [228.387, 340.355] | -31.534 [-108.066, +51.985] | 0.23 |
| stoch_return / final | 362.973 [288.084, 423.163] | 296.857 [249.096, 373.292] | -66.116 [-146.098, +38.934] | 0.11 |
| det_return / auc | 319.830 [262.862, 374.175] | 306.767 [251.784, 362.709] | -13.064 [-94.075, +66.486] | 0.34 |
| det_return / final | 370.495 [296.197, 425.438] | 320.943 [261.354, 391.225] | -49.553 [-131.187, +52.604] | 0.18 |
| stoch_success_rate / auc | 0.035 [0.001, 0.154] | 0.005 [0.001, 0.141] | -0.030 [-0.133, +0.130] | 0.33 |
| stoch_success_rate / final | 0.039 [0.000, 0.192] | 0.000 [0.000, 0.175] | -0.039 [-0.186, +0.131] | 0.24 |
| det_success_rate / auc | 0.037 [0.001, 0.161] | 0.006 [0.001, 0.131] | -0.031 [-0.135, +0.106] | 0.35 |
| det_success_rate / final | 0.047 [0.000, 0.206] | 0.006 [0.000, 0.097] | -0.042 [-0.197, +0.069] | 0.16 |
| stoch_crash_rate / auc | 0.807 [0.620, 0.948] | 0.927 [0.745, 0.984] | +0.119 [-0.109, +0.313] | 0.85 |
| stoch_crash_rate / final | 0.786 [0.486, 0.978] | 0.936 [0.744, 1.000] | +0.150 [-0.119, +0.453] | 0.85 |
| stoch_final_goal_dist / auc | 111.223 [69.068, 158.785] | 140.773 [84.198, 179.946] | +29.550 [-42.233, +91.151] | 0.79 |
| stoch_final_goal_dist / final | 73.013 [25.229, 139.175] | 139.390 [68.897, 179.212] | +66.377 [-28.329, +132.746] | 0.92 |
| stoch_tir / auc | 0.516 [0.446, 0.616] | 0.505 [0.447, 0.608] | -0.010 [-0.127, +0.114] | 0.46 |
| stoch_tir / final | 0.538 [0.433, 0.701] | 0.556 [0.485, 0.666] | +0.018 [-0.161, +0.174] | 0.60 |
| stoch_tbr / auc | 0.360 [0.241, 0.490] | 0.430 [0.315, 0.535] | +0.071 [-0.104, +0.230] | 0.79 |
| stoch_tbr / final | 0.302 [0.167, 0.473] | 0.376 [0.270, 0.483] | +0.075 [-0.126, +0.249] | 0.76 |
| noise cost (det − stoch return) / auc | 6.977 [1.642, 13.248] | 20.499 [5.787, 37.360] | +13.522 [-3.049, +30.889] | 0.94 |
| noise cost (det − stoch return) / final | 2.775 [-0.900, 10.055] | 6.122 [-2.332, 32.773] | +3.347 [-7.797, +29.546] | 0.75 |

Optimization difficulty (H2)

| diagnostic | global | state |
|---|---|---|
| n_seeds | 10 | 10 |
| n_nan_runs | 0 | 0 |
| final_return_std_across_seeds | 88.21 | 84.37 |
| final_return_iqr_across_seeds | 135.9 | 124.5 |
| auc_std_across_seeds | 72.96 | 74.52 |
| frac_seeds_never_learned (stoch_success_rate >= 0.5) | 0.7 | 0.8 |
| approx_kl_mean | 0.02212 | 0.09085 |
| approx_kl_max_minibatch_p99 | 0.2705 | 7.386 |
| frac_updates_kl_gt_0.02 | 0.3566 | 0.8076 |
| clipfrac_mean | 0.1857 | 0.1688 |
| explained_variance_final | 0.9229 | 0.9566 |
| logstd_mean_final | -3.51 | -2.634 |
| logstd_min_final | -3.51 | -5 |
| frac_seeds_sigma_hit_floor | 0 | 1 |
| logstd_frac_at_min_final | 0 | 0.1836 |
| grad_norm_sigma_over_mean_head | 0.008629 | 0.03667 |
| gsnr_mean_head_mean_last20pct | 1.008 | 1.014 |
| gsnr_sigma_bias_mean_last20pct | 1.166 | 1.238 |
| gsnr_sigma_weight_mean_last20pct | nan | 1.06 |
| gsnr_actor_trunk_mean_last20pct | 0.9979 | 0.9955 |
| trunk_cos_sigma_vs_mean_median_last20pct | nan | 0.000473 |
| trunk_gnorm_sigma_over_mean_median_last20pct | nan | 0.1047 |
| mean_outside_bounds_frac_median_last20pct | 0.1085 | 0.2294 |
| action_clipped_frac_median_last20pct | 0.1104 | 0.3151 |
| train_wall_clock_min_mean | 80.34 | 80.81 |
