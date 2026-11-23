### parking-v0 deep dive

IQM over seeds [95% bootstrap CI of the mean]; final = mean of the last 3 evals; σ / executed noise from the final eval's stoch episodes

| quantity | global σ | state-dependent σ(s) |
|---|---|---|
| stoch_success_final | 0 [0, 0.005] (n=10) | 0 [0, 0] (n=10) |
| stoch_crash_final | 1 [0.995, 1] (n=10) | 1 [1, 1] (n=10) |
| stoch_timeout_final | 0 [0, 0] (n=10) | 0 [0, 0] (n=10) |
| stoch_first_step_success_ge_0.5 | – | – |
| det_success_final | 0 [0, 0] (n=10) | 0 [0, 0] (n=10) |
| det_crash_final | 1 [1, 1] (n=10) | 1 [1, 1] (n=10) |
| det_timeout_final | 0 [0, 0] (n=10) | 0 [0, 0] (n=10) |
| det_first_step_success_ge_0.5 | – | – |
| frac_steps_within_3m | 0.0193 [0.0133, 0.0274] (n=10) | 0.0209 [0.0123, 0.0268] (n=10) |
| steps_to_park_successful_eps | 17 [17, 17] (n=1) | – |
| speed_within_3m | 13 [12.4, 13.4] (n=10) | 13 [12.6, 13.3] (n=9) |
| sigma_acceleration_within_3m | 0.0397 [0.0316, 0.0417] (n=10) | 0.0751 [0.0615, 0.362] (n=9) |
| sigma_acceleration_beyond_3m | 0.0397 [0.0316, 0.0417] (n=10) | 0.054 [0.0457, 0.0843] (n=10) |
| exec_noise_acceleration_within_3m | 0.0128 [0.00715, 0.0243] (n=10) | 0.0188 [0.00887, 0.0974] (n=9) |
| exec_noise_acceleration_beyond_3m | 0.00316 [0.00193, 0.00417] (n=10) | 0.00475 [0.00316, 0.0124] (n=10) |
| sigma_steering_within_3m | 0.0369 [0.0287, 0.0403] (n=10) | 0.0314 [0.0272, 0.0399] (n=9) |
| sigma_steering_beyond_3m | 0.0369 [0.0287, 0.0403] (n=10) | 0.0317 [0.0276, 0.042] (n=10) |
| exec_noise_steering_within_3m | 0.0369 [0.0287, 0.0403] (n=10) | 0.0314 [0.0272, 0.0399] (n=9) |
| exec_noise_steering_beyond_3m | 0.025 [0.0197, 0.028] (n=10) | 0.0205 [0.0174, 0.0241] (n=10) |
