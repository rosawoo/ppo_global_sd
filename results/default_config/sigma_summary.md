### Pendulum-v1

**global σ** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (|θ| from upright (rad) ≷ 0.3) | ρ: |θ| from upright (rad) | ρ: |θ̇| (rad/s) | η²: |θ| from upright (rad) | η²: |θ̇| (rad/s) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | torque | 0.229 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 2 | torque | 0.201 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 3 | torque | 0.240 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 4 | torque | 0.188 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 5 | torque | 0.204 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 6 | torque | 0.223 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 7 | torque | 0.291 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 8 | torque | 0.212 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 9 | torque | 0.280 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 10 | torque | 0.191 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
- torque: median std_s(log σ) = 0.000; median σ far/near = 1.00; median executed-noise far/near = 0.69; median frac. of states with mean outside the action box = 0.06; within-seed (state) share of pooled Var[log σ] = 0.00

**state-dependent σ(s)** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (|θ| from upright (rad) ≷ 0.3) | ρ: |θ| from upright (rad) | ρ: |θ̇| (rad/s) | η²: |θ| from upright (rad) | η²: |θ̇| (rad/s) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | torque | 0.157 | 0.938 | 19.27 | 12.79 | +0.58 | +0.59 | 0.85 | 0.63 | 0.80 |
| 2 | torque | 0.038 | 0.637 | 6.23 | 3.09 | +0.74 | +0.74 | 0.77 | 0.58 | 0.69 |
| 3 | torque | 0.036 | 0.546 | 5.74 | 2.40 | +0.80 | +0.86 | 0.53 | 0.56 | 0.51 |
| 4 | torque | 0.062 | 0.736 | 7.94 | 5.62 | +0.10 | +0.59 | 0.89 | 0.66 | 0.86 |
| 5 | torque | 0.054 | 0.877 | 11.07 | 8.23 | +0.36 | +0.55 | 0.93 | 0.66 | 0.90 |
| 6 | torque | 0.036 | 0.658 | 6.24 | 3.14 | +0.82 | +0.78 | 0.75 | 0.66 | 0.65 |
| 7 | torque | 0.035 | 0.501 | 4.06 | 2.33 | +0.72 | +0.78 | 0.72 | 0.66 | 0.66 |
| 8 | torque | 0.056 | 0.786 | 10.26 | 7.93 | +0.09 | +0.42 | 0.92 | 0.60 | 0.87 |
| 9 | torque | 0.115 | 0.859 | 21.09 | 11.61 | +0.47 | +0.49 | 0.80 | 0.54 | 0.78 |
| 10 | torque | 0.033 | 0.467 | 3.98 | 2.31 | +0.87 | +0.77 | 0.71 | 0.55 | 0.59 |
- torque: median std_s(log σ) = 0.697; median σ far/near = 4.38; median executed-noise far/near = 3.83; median frac. of states with mean outside the action box = 0.04; within-seed (state) share of pooled Var[log σ] = 0.85; ρ(σ, |θ| from upright (rad)): median +0.65, 8/10 seeds with ρ > +0.1, 0/10 with ρ < −0.1; ρ(σ, |θ̇| (rad/s)): median +0.67, 10/10 seeds with ρ > +0.1, 0/10 with ρ < −0.1

### parking-v0

**global σ** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (distance to goal (m) ≷ 3.0) | ρ: distance to goal (m) | ρ: |heading error| (rad) | ρ: speed (m/s) | η²: distance to goal (m) | η²: |heading error| (rad) | η²: speed (m/s) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | acceleration | 0.045 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | steering | 0.039 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | acceleration | 0.015 | 0.000 | 1.00 | nan | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | steering | 0.012 | 0.000 | 1.00 | nan | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | acceleration | 0.042 | 0.000 | 1.00 | nan | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | steering | 0.044 | 0.000 | 1.00 | nan | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | acceleration | 0.039 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | steering | 0.034 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | acceleration | 0.039 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | steering | 0.039 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | acceleration | 0.038 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | steering | 0.043 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | acceleration | 0.043 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | steering | 0.033 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | acceleration | 0.044 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | steering | 0.044 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | acceleration | 0.038 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | steering | 0.031 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | acceleration | 0.031 | 0.000 | 1.00 | nan | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | steering | 0.033 | 0.000 | 1.00 | nan | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
- acceleration: median std_s(log σ) = 0.000; median σ far/near = 1.00; median executed-noise far/near = 0.38; median frac. of states with mean outside the action box = 0.93; within-seed (state) share of pooled Var[log σ] = 0.00
- steering: median std_s(log σ) = 0.000; median σ far/near = 1.00; median executed-noise far/near = 0.70; median frac. of states with mean outside the action box = 0.29; within-seed (state) share of pooled Var[log σ] = 0.00

**state-dependent σ(s)** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (distance to goal (m) ≷ 3.0) | ρ: distance to goal (m) | ρ: |heading error| (rad) | ρ: speed (m/s) | η²: distance to goal (m) | η²: |heading error| (rad) | η²: speed (m/s) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | acceleration | 0.054 | 0.689 | 10.16 | nan | -0.05 | +0.09 | +0.13 | 0.08 | 0.04 | 0.05 | 0.01 |
| 1 | steering | 0.031 | 0.752 | 12.08 | nan | +0.10 | +0.11 | -0.01 | 0.09 | 0.15 | 0.32 | 0.05 |
| 2 | acceleration | 0.037 | 0.594 | 7.46 | nan | +0.09 | +0.00 | +0.65 | 0.15 | 0.16 | 0.39 | 0.52 |
| 2 | steering | 0.034 | 0.513 | 4.91 | nan | -0.11 | +0.17 | +0.25 | 0.13 | 0.09 | 0.05 | 0.16 |
| 3 | acceleration | 0.138 | 1.412 | 84.47 | 0.14 | -0.02 | -0.06 | +0.61 | 0.38 | 0.34 | 0.51 | 0.48 |
| 3 | steering | 0.034 | 0.547 | 6.14 | 0.63 | +0.10 | -0.26 | +0.03 | 0.12 | 0.14 | 0.11 | 0.10 |
| 4 | acceleration | 0.062 | 0.667 | 9.21 | 0.62 | +0.06 | -0.22 | -0.15 | 0.23 | 0.11 | 0.04 | 0.16 |
| 4 | steering | 0.064 | 0.782 | 13.76 | 1.97 | +0.24 | -0.20 | -0.48 | 0.21 | 0.30 | 0.25 | 0.26 |
| 5 | acceleration | 0.034 | 0.729 | 10.47 | 0.38 | -0.21 | -0.05 | +0.62 | 0.36 | 0.24 | 0.37 | 0.38 |
| 5 | steering | 0.036 | 0.536 | 5.10 | 0.79 | -0.06 | -0.19 | -0.11 | 0.04 | 0.32 | 0.09 | 0.14 |
| 6 | acceleration | 0.124 | 1.172 | 46.89 | 0.43 | -0.34 | -0.49 | +0.60 | 0.25 | 0.25 | 0.43 | 0.44 |
| 6 | steering | 0.026 | 0.591 | 6.55 | 0.83 | +0.12 | +0.26 | +0.03 | 0.05 | 0.17 | 0.12 | 0.13 |
| 7 | acceleration | 0.034 | 0.550 | 6.67 | 0.61 | -0.32 | +0.18 | +0.61 | 0.26 | 0.14 | 0.33 | 0.40 |
| 7 | steering | 0.029 | 0.395 | 3.58 | 0.82 | +0.04 | +0.28 | +0.52 | 0.14 | 0.22 | 0.21 | 0.39 |
| 8 | acceleration | 0.048 | 0.694 | 9.02 | 1.07 | +0.45 | -0.14 | +0.18 | 0.24 | 0.15 | 0.08 | 0.25 |
| 8 | steering | 0.028 | 0.541 | 5.30 | 1.46 | +0.36 | -0.19 | -0.07 | 0.10 | 0.22 | 0.01 | 0.08 |
| 9 | acceleration | 0.074 | 0.591 | 6.39 | nan | +0.56 | -0.19 | +0.53 | 0.48 | 0.24 | 0.30 | 0.55 |
| 9 | steering | 0.019 | 0.494 | 4.95 | nan | +0.01 | +0.19 | -0.04 | 0.03 | 0.26 | 0.03 | 0.10 |
| 10 | acceleration | 0.050 | 0.693 | 9.28 | nan | -0.07 | -0.15 | +0.21 | 0.04 | 0.11 | 0.05 | 0.07 |
| 10 | steering | 0.039 | 0.435 | 3.87 | nan | -0.12 | -0.29 | -0.13 | 0.05 | 0.26 | 0.08 | 0.19 |
- acceleration: median std_s(log σ) = 0.691; median σ far/near = 0.52; median executed-noise far/near = 0.13; median frac. of states with mean outside the action box = 0.96; within-seed (state) share of pooled Var[log σ] = 0.90; ρ(σ, distance to goal (m)): median -0.03, 2/10 seeds with ρ > +0.1, 3/10 with ρ < −0.1; ρ(σ, |heading error| (rad)): median -0.10, 1/10 seeds with ρ > +0.1, 5/10 with ρ < −0.1; ρ(σ, speed (m/s)): median +0.56, 9/10 seeds with ρ > +0.1, 1/10 with ρ < −0.1
- steering: median std_s(log σ) = 0.538; median σ far/near = 0.82; median executed-noise far/near = 0.56; median frac. of states with mean outside the action box = 0.33; within-seed (state) share of pooled Var[log σ] = 0.83; ρ(σ, distance to goal (m)): median +0.07, 3/10 seeds with ρ > +0.1, 2/10 with ρ < −0.1; ρ(σ, |heading error| (rad)): median -0.04, 5/10 seeds with ρ > +0.1, 5/10 with ρ < −0.1; ρ(σ, speed (m/s)): median -0.03, 2/10 seeds with ρ > +0.1, 3/10 with ρ < −0.1

### simglucose-adolescent1-v0

**global σ** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (CGM (mg/dL) ≷ 100.0) | ρ: CGM (mg/dL) | ρ: CGM trend (mg/dL/min) | η²: CGM (mg/dL) | η²: CGM trend (mg/dL/min) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | basal_insulin | 0.030 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 2 | basal_insulin | 0.013 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 3 | basal_insulin | 0.013 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 4 | basal_insulin | 0.037 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 5 | basal_insulin | 0.053 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 6 | basal_insulin | 0.056 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 7 | basal_insulin | 0.054 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 8 | basal_insulin | 0.057 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 9 | basal_insulin | 0.013 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 10 | basal_insulin | 0.026 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
- basal_insulin: median std_s(log σ) = 0.000; median σ far/near = 1.00; median executed-noise far/near = 1.01; median frac. of states with mean outside the action box = 0.08; within-seed (state) share of pooled Var[log σ] = 0.00

**state-dependent σ(s)** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (CGM (mg/dL) ≷ 100.0) | ρ: CGM (mg/dL) | ρ: CGM trend (mg/dL/min) | η²: CGM (mg/dL) | η²: CGM trend (mg/dL/min) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | basal_insulin | 1.894 | 2.711 | 1096.63 | 0.64 | -0.04 | -0.19 | 0.01 | 0.04 | 0.04 |
| 2 | basal_insulin | 0.163 | 1.368 | 92.05 | 8.16 | +0.67 | -0.04 | 0.40 | 0.01 | 0.34 |
| 3 | basal_insulin | 0.565 | 1.948 | 781.67 | 0.25 | -0.08 | -0.35 | 0.02 | 0.09 | 0.10 |
| 4 | basal_insulin | 0.369 | 1.793 | 237.65 | 4.69 | +0.25 | -0.11 | 0.18 | 0.02 | 0.12 |
| 5 | basal_insulin | 0.231 | 1.653 | 156.94 | 0.17 | -0.24 | -0.11 | 0.29 | 0.01 | 0.18 |
| 6 | basal_insulin | 0.344 | 1.731 | 232.62 | 0.80 | -0.02 | -0.15 | 0.15 | 0.02 | 0.03 |
| 7 | basal_insulin | 1.371 | 2.340 | 1096.63 | 0.26 | -0.06 | -0.13 | 0.05 | 0.02 | 0.04 |
| 8 | basal_insulin | 1.455 | 2.593 | 1096.63 | 0.72 | +0.06 | -0.10 | 0.01 | 0.01 | 0.01 |
| 9 | basal_insulin | 0.777 | 2.188 | 1096.63 | 0.52 | -0.19 | -0.08 | 0.14 | 0.02 | 0.05 |
| 10 | basal_insulin | 1.072 | 2.332 | 1096.63 | 1.19 | +0.17 | -0.18 | 0.05 | 0.03 | 0.07 |
- basal_insulin: median std_s(log σ) = 2.068; median σ far/near = 0.68; median executed-noise far/near = 0.78; median frac. of states with mean outside the action box = 0.20; within-seed (state) share of pooled Var[log σ] = 0.95; ρ(σ, CGM (mg/dL)): median -0.03, 3/10 seeds with ρ > +0.1, 2/10 with ρ < −0.1; ρ(σ, CGM trend (mg/dL/min)): median -0.12, 0/10 seeds with ρ > +0.1, 8/10 with ρ < −0.1
