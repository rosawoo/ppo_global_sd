### parking-v0 deep dive

IQM over seeds [95% bootstrap CI of the mean]; final = mean of the last 3 evals; σ / executed noise from the final eval's stoch episodes

| quantity | global σ | state-dependent σ(s) |
|---|---|---|
| stoch_success_final | 0.483 [0.288, 0.675] (n=10) | 0.364 [0.238, 0.608] (n=10) |
| stoch_crash_final | 0.522 [0.337, 0.713] (n=10) | 0.636 [0.402, 0.762] (n=10) |
| stoch_timeout_final | -0.00278 [-0.0133, -1.94e-17] (n=10) | 0 [-0.01, 1.39e-17] (n=10) |
| stoch_first_step_success_ge_0.5 | 8.2e+05 [6.68e+05, 1.37e+06] (n=6) | 6.88e+05 [4.51e+05, 9.89e+05] (n=4) |
| det_success_final | 0.5 [0.303, 0.702] (n=10) | 0.356 [0.242, 0.592] (n=10) |
| det_crash_final | 0.506 [0.312, 0.698] (n=10) | 0.647 [0.413, 0.76] (n=10) |
| det_timeout_final | -0.00278 [-0.015, -2.04e-17] (n=10) | 0 [-0.00833, 3.7e-18] (n=10) |
| det_first_step_success_ge_0.5 | 8.46e+05 [6.42e+05, 1.34e+06] (n=6) | 7.54e+05 [4.65e+05, 1.09e+06] (n=4) |
| frac_steps_within_3m | 0.0467 [0.035, 0.0505] (n=10) | 0.0286 [0.0235, 0.0361] (n=10) |
| steps_to_park_successful_eps | 16.6 [15.6, 17.2] (n=10) | 17.6 [17, 17.9] (n=10) |
| speed_within_3m | 13.1 [12, 13.8] (n=10) | 13.3 [12.3, 13.8] (n=10) |
| sigma_acceleration_within_3m | 0.633 [0.578, 0.769] (n=10) | 6.2 [5.06, 6.66] (n=10) |
| sigma_acceleration_beyond_3m | 0.633 [0.578, 0.769] (n=10) | 4.14 [3.29, 4.77] (n=10) |
| exec_noise_acceleration_within_3m | 0.274 [0.147, 0.384] (n=10) | 0.655 [0.522, 0.736] (n=10) |
| exec_noise_acceleration_beyond_3m | 0.0813 [0.0536, 0.107] (n=10) | 0.19 [0.169, 0.211] (n=10) |
| sigma_steering_within_3m | 0.177 [0.168, 0.192] (n=10) | 0.84 [0.495, 1.33] (n=10) |
| sigma_steering_beyond_3m | 0.177 [0.168, 0.192] (n=10) | 1.51 [1.17, 2.01] (n=10) |
| exec_noise_steering_within_3m | 0.145 [0.118, 0.173] (n=10) | 0.24 [0.148, 0.338] (n=10) |
| exec_noise_steering_beyond_3m | 0.102 [0.0857, 0.119] (n=10) | 0.311 [0.25, 0.375] (n=10) |
