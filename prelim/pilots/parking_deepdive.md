### parking-v0 deep dive

IQM over seeds [95% bootstrap CI of the mean]; final = mean of the last 3 evals; σ / executed noise from the final eval's stoch episodes

| quantity | global σ | state-dependent σ(s) |
|---|---|---|
| stoch_success_final | 0 [0, 0] (n=1) | 0 [0, 0] (n=1) |
| stoch_crash_final | 1 [1, 1] (n=1) | 1 [1, 1] (n=1) |
| stoch_timeout_final | 0 [0, 0] (n=1) | 0 [0, 0] (n=1) |
| stoch_first_step_success_ge_0.5 | – | – |
| det_success_final | 0 [0, 0] (n=1) | 0.0167 [0.0167, 0.0167] (n=1) |
| det_crash_final | 1 [1, 1] (n=1) | 0.983 [0.983, 0.983] (n=1) |
| det_timeout_final | 0 [0, 0] (n=1) | 0 [0, 0] (n=1) |
| det_first_step_success_ge_0.5 | – | – |
| frac_steps_within_3m | 0.0098 [0.0098, 0.0098] (n=1) | 0.0129 [0.0129, 0.0129] (n=1) |
| steps_to_park_successful_eps | – | – |
| speed_within_3m | 11.8 [11.8, 11.8] (n=1) | 12.7 [12.7, 12.7] (n=1) |
| sigma_acceleration_within_3m | 0.215 [0.215, 0.215] (n=1) | 0.102 [0.102, 0.102] (n=1) |
| sigma_acceleration_beyond_3m | 0.215 [0.215, 0.215] (n=1) | 0.102 [0.102, 0.102] (n=1) |
| exec_noise_acceleration_within_3m | 0.00666 [0.00666, 0.00666] (n=1) | 0.0359 [0.0359, 0.0359] (n=1) |
| exec_noise_acceleration_beyond_3m | 0.042 [0.042, 0.042] (n=1) | 0.02 [0.02, 0.02] (n=1) |
| sigma_steering_within_3m | 0.201 [0.201, 0.201] (n=1) | 0.106 [0.106, 0.106] (n=1) |
| sigma_steering_beyond_3m | 0.201 [0.201, 0.201] (n=1) | 0.128 [0.128, 0.128] (n=1) |
| exec_noise_steering_within_3m | 0.201 [0.201, 0.201] (n=1) | 0.106 [0.106, 0.106] (n=1) |
| exec_noise_steering_beyond_3m | 0.151 [0.151, 0.151] (n=1) | 0.105 [0.105, 0.105] (n=1) |
