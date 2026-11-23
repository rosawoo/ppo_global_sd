### Pendulum-v1

IQM across seeds [95% stratified-bootstrap CI]; 'm − global' = IQM difference with a 95% bootstrap CI (seeds resampled per arm) and P(m > global) over bootstrap draws

| metric | global | state | state − global [95% CI] | P(state > global) |
|---|---|---|---|---|
| stoch_return / auc | -306.503 [-340.949, -287.866] | -326.613 [-346.547, -315.462] | -20.110 [-46.874, +15.638] | 0.11 |
| stoch_return / final | -184.878 [-194.521, -177.684] | -176.461 [-180.355, -173.647] | +8.417 [+0.209, +18.646] | 0.98 |
| det_return / auc | -294.488 [-327.542, -276.040] | -273.917 [-306.471, -260.928] | +20.572 [-15.818, +55.496] | 0.88 |
| det_return / final | -181.887 [-191.132, -176.263] | -173.553 [-176.318, -170.778] | +8.334 [+1.940, +17.881] | 1.00 |
| stoch_final_goal_dist / auc | 0.261 [0.204, 0.380] | 0.271 [0.245, 0.307] | +0.010 [-0.109, +0.078] | 0.57 |
| stoch_final_goal_dist / final | 0.050 [0.023, 0.127] | 0.025 [0.022, 0.027] | -0.026 [-0.102, +0.001] | 0.04 |
| noise cost (det − stoch return) / auc | 12.075 [8.390, 16.913] | 47.537 [33.958, 60.612] | +35.462 [+21.269, +49.346] | 1.00 |
| noise cost (det − stoch return) / final | 2.366 [0.617, 4.002] | 1.457 [-1.593, 7.957] | -0.910 [-4.474, +5.678] | 0.40 |

Optimization difficulty (H2)

| diagnostic | global | state |
|---|---|---|
| n_seeds | 10 | 10 |
| n_nan_runs | 0 | 0 |
| final_return_std_across_seeds | 12.66 | 4.678 |
| final_return_iqr_across_seeds | 18.05 | 6.337 |
| auc_std_across_seeds | 42.4 | 23.97 |
| frac_seeds_never_learned (stoch_return_mean >= -300.0) | 0 | 0 |
| approx_kl_mean | 0.01925 | 0.02604 |
| approx_kl_max_minibatch_p99 | 1.025 | 2.326 |
| frac_updates_kl_gt_0.02 | 0.3098 | 0.4066 |
| clipfrac_mean | 0.07547 | 0.08389 |
| explained_variance_final | 0.9906 | 0.99 |
| logstd_mean_final | -0.6168 | 1.94 |
| logstd_min_final | -0.6168 | -0.4895 |
| frac_seeds_sigma_hit_floor | 0 | 0 |
| logstd_frac_at_min_final | 0 | 0 |
| grad_norm_sigma_over_mean_head | 0.1378 | 2.034 |
| gsnr_mean_head_mean_last20pct | 1.008 | 1.116 |
| gsnr_sigma_bias_mean_last20pct | 1.164 | 1.204 |
| gsnr_sigma_weight_mean_last20pct | nan | 1.12 |
| gsnr_actor_trunk_mean_last20pct | 1.053 | 1.038 |
| trunk_cos_sigma_vs_mean_median_last20pct | nan | -0.003347 |
| trunk_gnorm_sigma_over_mean_median_last20pct | nan | 0.4838 |
| mean_outside_bounds_frac_median_last20pct | 0.09592 | 0.8885 |
| action_clipped_frac_median_last20pct | 0.1336 | 0.8887 |
| train_wall_clock_min_mean | 8.815 | 8.804 |

### parking-v0

IQM across seeds [95% stratified-bootstrap CI]; 'm − global' = IQM difference with a 95% bootstrap CI (seeds resampled per arm) and P(m > global) over bootstrap draws

| metric | global | state | state − global [95% CI] | P(state > global) |
|---|---|---|---|---|
| stoch_return / auc | -10.142 [-11.040, -9.177] | -10.844 [-11.467, -9.553] | -0.702 [-1.895, +0.891] | 0.19 |
| stoch_return / final | -8.660 [-10.239, -7.162] | -9.504 [-10.739, -7.773] | -0.844 [-2.885, +1.521] | 0.25 |
| det_return / auc | -10.783 [-11.854, -9.706] | -11.238 [-12.003, -10.025] | -0.455 [-1.797, +1.222] | 0.31 |
| det_return / final | -8.553 [-10.105, -6.952] | -9.423 [-10.523, -7.745] | -0.869 [-2.850, +1.450] | 0.24 |
| stoch_success_rate / auc | 0.274 [0.113, 0.426] | 0.210 [0.093, 0.422] | -0.064 [-0.259, +0.203] | 0.34 |
| stoch_success_rate / final | 0.483 [0.197, 0.758] | 0.364 [0.161, 0.664] | -0.119 [-0.472, +0.289] | 0.29 |
| det_success_rate / auc | 0.293 [0.125, 0.451] | 0.220 [0.100, 0.431] | -0.073 [-0.277, +0.199] | 0.32 |
| det_success_rate / final | 0.500 [0.219, 0.781] | 0.356 [0.167, 0.642] | -0.144 [-0.486, +0.256] | 0.25 |
| stoch_crash_rate / auc | 0.730 [0.578, 0.882] | 0.792 [0.579, 0.908] | +0.062 [-0.202, +0.254] | 0.66 |
| stoch_crash_rate / final | 0.522 [0.264, 0.803] | 0.636 [0.347, 0.839] | +0.114 [-0.289, +0.458] | 0.70 |
| stoch_final_goal_dist / auc | 12.900 [8.421, 18.229] | 14.681 [9.356, 19.866] | +1.782 [-5.799, +8.774] | 0.68 |
| stoch_final_goal_dist / final | 9.538 [3.821, 16.268] | 12.714 [6.704, 18.448] | +3.176 [-5.998, +11.360] | 0.74 |
| noise cost (det − stoch return) / auc | -0.700 [-0.894, -0.538] | -0.459 [-0.641, -0.319] | +0.240 [-0.002, +0.478] | 0.97 |
| noise cost (det − stoch return) / final | 0.137 [-0.006, 0.298] | 0.171 [0.005, 0.246] | +0.034 [-0.188, +0.199] | 0.61 |

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
| train_wall_clock_min_mean | 262.9 | 261.8 |

### simglucose-adolescent1-v0

IQM across seeds [95% stratified-bootstrap CI]; 'm − global' = IQM difference with a 95% bootstrap CI (seeds resampled per arm) and P(m > global) over bootstrap draws

| metric | global | state | state − global [95% CI] | P(state > global) |
|---|---|---|---|---|
| stoch_return / auc | 449.559 [443.226, 453.551] | 431.310 [395.530, 448.919] | -18.249 [-53.281, +0.913] | 0.03 |
| stoch_return / final | 474.758 [474.083, 475.017] | 465.441 [440.470, 471.359] | -9.317 [-34.241, -3.246] | 0.00 |
| det_return / auc | 439.129 [428.509, 448.829] | 362.579 [295.080, 415.648] | -76.550 [-145.010, -22.354] | 0.00 |
| det_return / final | 474.320 [473.722, 474.796] | 428.360 [320.811, 469.385] | -45.961 [-153.578, -4.935] | 0.00 |
| stoch_success_rate / auc | 0.770 [0.733, 0.819] | 0.454 [0.218, 0.640] | -0.316 [-0.559, -0.127] | 0.00 |
| stoch_success_rate / final | 0.958 [0.950, 0.972] | 0.611 [0.350, 0.842] | -0.347 [-0.614, -0.119] | 0.00 |
| det_success_rate / auc | 0.738 [0.683, 0.797] | 0.245 [0.039, 0.537] | -0.492 [-0.706, -0.202] | 0.00 |
| det_success_rate / final | 0.967 [0.947, 0.989] | 0.506 [0.042, 0.947] | -0.461 [-0.922, -0.022] | 0.01 |
| stoch_crash_rate / auc | 0.159 [0.114, 0.199] | 0.362 [0.186, 0.636] | +0.203 [+0.022, +0.478] | 0.99 |
| stoch_crash_rate / final | 0.000 [0.000, 0.000] | 0.136 [0.017, 0.400] | +0.136 [+0.017, +0.392] | 1.00 |
| stoch_final_goal_dist / auc | 16.512 [12.526, 21.605] | 26.856 [14.807, 63.654] | +10.344 [-3.185, +46.946] | 0.90 |
| stoch_final_goal_dist / final | 0.864 [0.710, 1.010] | 6.136 [2.066, 35.437] | +5.272 [+1.205, +34.931] | 1.00 |
| stoch_tir / auc | 0.903 [0.884, 0.929] | 0.801 [0.692, 0.886] | -0.102 [-0.218, -0.016] | 0.00 |
| stoch_tir / final | 0.995 [0.992, 0.996] | 0.915 [0.797, 0.965] | -0.080 [-0.199, -0.030] | 0.00 |
| stoch_tbr / auc | 0.088 [0.062, 0.106] | 0.171 [0.105, 0.276] | +0.084 [+0.014, +0.193] | 1.00 |
| stoch_tbr / final | 0.004 [0.002, 0.007] | 0.074 [0.026, 0.153] | +0.071 [+0.022, +0.149] | 1.00 |
| noise cost (det − stoch return) / auc | -8.844 [-16.222, -4.381] | -52.027 [-108.883, -17.456] | -43.183 [-99.289, -7.977] | 0.01 |
| noise cost (det − stoch return) / final | -0.343 [-0.687, 0.062] | -30.864 [-114.012, 0.559] | -30.522 [-115.706, +0.913] | 0.04 |

Optimization difficulty (H2)

| diagnostic | global | state |
|---|---|---|
| n_seeds | 10 | 10 |
| n_nan_runs | 0 | 0 |
| final_return_std_across_seeds | 0.6851 | 26.46 |
| final_return_iqr_across_seeds | 0.642 | 18.85 |
| auc_std_across_seeds | 8.028 | 41.47 |
| frac_seeds_never_learned (stoch_success_rate >= 0.5) | 0 | 0.3 |
| approx_kl_mean | 0.003384 | 0.01532 |
| approx_kl_max_minibatch_p99 | 0.02267 | 0.8753 |
| frac_updates_kl_gt_0.02 | 0 | 0.2463 |
| clipfrac_mean | 0.03865 | 0.08271 |
| explained_variance_final | 0.412 | 0.5454 |
| logstd_mean_final | -0.5828 | -0.957 |
| logstd_min_final | -0.5828 | -4.906 |
| frac_seeds_sigma_hit_floor | 0 | 1 |
| logstd_frac_at_min_final | 0 | 0.005273 |
| grad_norm_sigma_over_mean_head | 0.119 | 0.1845 |
| gsnr_mean_head_mean_last20pct | 1.03 | 1.028 |
| gsnr_sigma_bias_mean_last20pct | 1.154 | 1.215 |
| gsnr_sigma_weight_mean_last20pct | nan | 1.066 |
| gsnr_actor_trunk_mean_last20pct | 1.013 | 1.001 |
| trunk_cos_sigma_vs_mean_median_last20pct | nan | 0.0006583 |
| trunk_gnorm_sigma_over_mean_median_last20pct | nan | 0.1804 |
| mean_outside_bounds_frac_median_last20pct | 0.6411 | 0.6414 |
| action_clipped_frac_median_last20pct | 0.6463 | 0.6608 |
| train_wall_clock_min_mean | 79.32 | 78.67 |
