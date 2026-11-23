### Pendulum-v1

**global σ** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (|θ| from upright (rad) ≷ 0.3) | ρ: |θ| from upright (rad) | ρ: |θ̇| (rad/s) | η²: |θ| from upright (rad) | η²: |θ̇| (rad/s) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | torque | 0.529 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 2 | torque | 0.654 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 3 | torque | 0.346 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 4 | torque | 0.362 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 5 | torque | 1.001 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 6 | torque | 0.704 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 7 | torque | 0.955 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 8 | torque | 0.668 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 9 | torque | 0.451 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 10 | torque | 0.239 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
- torque: median std_s(log σ) = 0.000; median σ far/near = 1.00; median executed-noise far/near = 0.54; median frac. of states with mean outside the action box = 0.10; within-seed (state) share of pooled Var[log σ] = 0.00

**state-dependent σ(s)** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (|θ| from upright (rad) ≷ 0.3) | ρ: |θ| from upright (rad) | ρ: |θ̇| (rad/s) | η²: |θ| from upright (rad) | η²: |θ̇| (rad/s) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | torque | 7.250 | 0.275 | 1.00 | 0.91 | -0.22 | -0.25 | 0.17 | 0.15 | 0.10 |
| 2 | torque | 7.090 | 0.254 | 1.77 | 0.77 | -0.41 | -0.42 | 0.32 | 0.28 | 0.22 |
| 3 | torque | 7.193 | 0.220 | 1.18 | 0.85 | -0.36 | -0.37 | 0.23 | 0.29 | 0.27 |
| 4 | torque | 7.011 | 0.340 | 1.99 | 0.73 | -0.45 | -0.48 | 0.42 | 0.33 | 0.33 |
| 5 | torque | 7.208 | 0.248 | 1.00 | 0.86 | -0.29 | -0.31 | 0.17 | 0.28 | 0.27 |
| 6 | torque | 7.008 | 0.270 | 1.76 | 0.79 | -0.47 | -0.48 | 0.33 | 0.28 | 0.22 |
| 7 | torque | 6.738 | 0.349 | 2.18 | 0.79 | -0.29 | -0.58 | 0.23 | 0.18 | 0.14 |
| 8 | torque | 7.132 | 0.201 | 1.44 | 0.85 | -0.34 | -0.46 | 0.28 | 0.26 | 0.21 |
| 9 | torque | 6.932 | 0.294 | 2.29 | 0.65 | -0.54 | -0.55 | 0.54 | 0.61 | 0.58 |
| 10 | torque | 7.314 | 0.099 | 1.00 | 0.94 | -0.25 | -0.23 | 0.24 | 0.07 | 0.04 |
- torque: median std_s(log σ) = 0.262; median σ far/near = 0.82; median executed-noise far/near = 0.76; median frac. of states with mean outside the action box = 0.89; within-seed (state) share of pooled Var[log σ] = 0.98; ρ(σ, |θ| from upright (rad)): median -0.35, 0/10 seeds with ρ > +0.1, 10/10 with ρ < −0.1; ρ(σ, |θ̇| (rad/s)): median -0.44, 0/10 seeds with ρ > +0.1, 10/10 with ρ < −0.1

### parking-v0

**global σ** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (distance to goal (m) ≷ 3.0) | ρ: distance to goal (m) | ρ: |heading error| (rad) | ρ: speed (m/s) | η²: distance to goal (m) | η²: |heading error| (rad) | η²: speed (m/s) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | acceleration | 0.924 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | steering | 0.181 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | acceleration | 0.656 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | steering | 0.201 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | acceleration | 0.551 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | steering | 0.147 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | acceleration | 0.576 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | steering | 0.174 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | acceleration | 0.951 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | steering | 0.221 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | acceleration | 0.696 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | steering | 0.165 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | acceleration | 0.519 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | steering | 0.172 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | acceleration | 0.777 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | steering | 0.173 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | acceleration | 0.542 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | steering | 0.169 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | acceleration | 0.486 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | steering | 0.195 | 0.000 | 1.00 | 1.00 | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
- acceleration: median std_s(log σ) = 0.000; median σ far/near = 1.00; median executed-noise far/near = 0.37; median frac. of states with mean outside the action box = 0.86; within-seed (state) share of pooled Var[log σ] = 0.00
- steering: median std_s(log σ) = 0.000; median σ far/near = 1.00; median executed-noise far/near = 0.69; median frac. of states with mean outside the action box = 0.43; within-seed (state) share of pooled Var[log σ] = 0.00

**state-dependent σ(s)** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (distance to goal (m) ≷ 3.0) | ρ: distance to goal (m) | ρ: |heading error| (rad) | ρ: speed (m/s) | η²: distance to goal (m) | η²: |heading error| (rad) | η²: speed (m/s) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | acceleration | 2.871 | 0.822 | 14.71 | 0.46 | -0.18 | -0.19 | +0.63 | 0.24 | 0.20 | 0.37 | 0.39 |
| 1 | steering | 0.535 | 0.751 | 11.52 | 1.61 | +0.40 | +0.37 | -0.52 | 0.20 | 0.18 | 0.29 | 0.37 |
| 2 | acceleration | 2.559 | 0.743 | 14.39 | 0.48 | -0.40 | -0.26 | +0.56 | 0.28 | 0.15 | 0.29 | 0.40 |
| 2 | steering | 1.599 | 1.195 | 50.23 | 1.30 | -0.01 | +0.49 | -0.44 | 0.12 | 0.26 | 0.19 | 0.42 |
| 3 | acceleration | 5.669 | 0.544 | 4.91 | 0.78 | -0.46 | -0.37 | +0.32 | 0.23 | 0.24 | 0.08 | 0.19 |
| 3 | steering | 0.909 | 0.944 | 25.72 | 8.36 | +0.67 | +0.33 | -0.63 | 0.58 | 0.18 | 0.38 | 0.62 |
| 4 | acceleration | 4.245 | 0.622 | 6.64 | 0.56 | -0.66 | -0.44 | +0.67 | 0.39 | 0.16 | 0.41 | 0.51 |
| 4 | steering | 0.903 | 1.009 | 25.97 | 7.39 | +0.64 | +0.34 | -0.56 | 0.53 | 0.15 | 0.28 | 0.54 |
| 5 | acceleration | 4.330 | 0.686 | 7.81 | 0.96 | +0.23 | -0.44 | +0.32 | 0.07 | 0.20 | 0.08 | 0.10 |
| 5 | steering | 2.080 | 1.029 | 29.83 | 1.35 | -0.34 | +0.43 | -0.53 | 0.32 | 0.40 | 0.32 | 0.48 |
| 6 | acceleration | 2.201 | 0.738 | 11.29 | 0.70 | -0.09 | +0.27 | +0.15 | 0.06 | 0.13 | 0.01 | 0.14 |
| 6 | steering | 1.522 | 1.177 | 58.56 | 1.47 | +0.56 | +0.66 | -0.41 | 0.29 | 0.45 | 0.15 | 0.60 |
| 7 | acceleration | 4.815 | 0.456 | 5.02 | 0.80 | -0.24 | +0.13 | +0.48 | 0.12 | 0.20 | 0.17 | 0.19 |
| 7 | steering | 1.523 | 0.853 | 18.00 | 1.39 | -0.05 | +0.26 | -0.25 | 0.07 | 0.15 | 0.08 | 0.16 |
| 8 | acceleration | 5.095 | 0.342 | 2.84 | 0.83 | -0.04 | -0.29 | +0.22 | 0.07 | 0.12 | 0.07 | 0.11 |
| 8 | steering | 1.669 | 0.983 | 26.41 | 1.46 | -0.18 | +0.41 | -0.56 | 0.17 | 0.38 | 0.31 | 0.50 |
| 9 | acceleration | 3.811 | 0.594 | 6.59 | nan | +0.12 | -0.22 | +0.69 | 0.13 | 0.32 | 0.45 | 0.47 |
| 9 | steering | 1.736 | 1.395 | 62.54 | nan | -0.20 | -0.11 | -0.67 | 0.28 | 0.17 | 0.46 | 0.53 |
| 10 | acceleration | 5.585 | 0.636 | 7.56 | 0.80 | -0.00 | -0.42 | +0.45 | 0.15 | 0.13 | 0.10 | 0.31 |
| 10 | steering | 3.036 | 1.120 | 29.01 | 1.38 | -0.37 | +0.48 | -0.42 | 0.24 | 0.25 | 0.23 | 0.43 |
- acceleration: median std_s(log σ) = 0.629; median σ far/near = 0.78; median executed-noise far/near = 0.31; median frac. of states with mean outside the action box = 0.98; within-seed (state) share of pooled Var[log σ] = 0.73; ρ(σ, distance to goal (m)): median -0.14, 2/10 seeds with ρ > +0.1, 5/10 with ρ < −0.1; ρ(σ, |heading error| (rad)): median -0.28, 2/10 seeds with ρ > +0.1, 8/10 with ρ < −0.1; ρ(σ, speed (m/s)): median +0.47, 10/10 seeds with ρ > +0.1, 0/10 with ρ < −0.1
- steering: median std_s(log σ) = 1.019; median σ far/near = 1.46; median executed-noise far/near = 1.25; median frac. of states with mean outside the action box = 0.74; within-seed (state) share of pooled Var[log σ] = 0.86; ρ(σ, distance to goal (m)): median -0.03, 4/10 seeds with ρ > +0.1, 4/10 with ρ < −0.1; ρ(σ, |heading error| (rad)): median +0.39, 9/10 seeds with ρ > +0.1, 1/10 with ρ < −0.1; ρ(σ, speed (m/s)): median -0.53, 0/10 seeds with ρ > +0.1, 10/10 with ρ < −0.1

### simglucose-adolescent1-v0

**global σ** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (CGM (mg/dL) ≷ 100.0) | ρ: CGM (mg/dL) | ρ: CGM trend (mg/dL/min) | η²: CGM (mg/dL) | η²: CGM trend (mg/dL/min) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | basal_insulin | 0.571 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 2 | basal_insulin | 0.474 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 3 | basal_insulin | 0.528 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 4 | basal_insulin | 0.527 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 5 | basal_insulin | 0.544 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 6 | basal_insulin | 0.629 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 7 | basal_insulin | 0.518 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 8 | basal_insulin | 0.513 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 9 | basal_insulin | 0.606 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 10 | basal_insulin | 0.707 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
- basal_insulin: median std_s(log σ) = 0.000; median σ far/near = 1.00; median executed-noise far/near = 1.15; median frac. of states with mean outside the action box = 0.64; within-seed (state) share of pooled Var[log σ] = 0.00

**state-dependent σ(s)** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (CGM (mg/dL) ≷ 100.0) | ρ: CGM (mg/dL) | ρ: CGM trend (mg/dL/min) | η²: CGM (mg/dL) | η²: CGM trend (mg/dL/min) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | basal_insulin | 2.125 | 1.881 | 231.36 | 0.24 | -0.48 | +0.00 | 0.25 | 0.00 | 0.24 |
| 2 | basal_insulin | 1.465 | 2.023 | 689.55 | 1.55 | +0.13 | +0.02 | 0.05 | 0.00 | 0.01 |
| 3 | basal_insulin | 1.632 | 1.700 | 274.94 | 0.43 | -0.06 | -0.02 | 0.12 | 0.00 | 0.00 |
| 4 | basal_insulin | 1.961 | 1.951 | 433.51 | 1.17 | +0.11 | -0.09 | 0.02 | 0.01 | 0.02 |
| 5 | basal_insulin | 1.028 | 1.573 | 180.74 | 0.39 | -0.52 | +0.17 | 0.25 | 0.03 | 0.30 |
| 6 | basal_insulin | 1.506 | 1.768 | 345.87 | 0.87 | +0.02 | +0.00 | 0.06 | 0.00 | 0.00 |
| 7 | basal_insulin | 0.801 | 1.629 | 259.46 | 0.27 | -0.49 | +0.02 | 0.27 | 0.00 | 0.23 |
| 8 | basal_insulin | 1.549 | 1.760 | 316.25 | 0.98 | -0.05 | +0.01 | 0.02 | 0.00 | 0.00 |
| 9 | basal_insulin | 1.193 | 1.486 | 154.30 | 0.26 | -0.52 | +0.06 | 0.29 | 0.01 | 0.28 |
| 10 | basal_insulin | 1.156 | 1.819 | 512.07 | 1.37 | +0.09 | +0.19 | 0.07 | 0.03 | 0.04 |
- basal_insulin: median std_s(log σ) = 1.764; median σ far/near = 0.65; median executed-noise far/near = 0.75; median frac. of states with mean outside the action box = 0.64; within-seed (state) share of pooled Var[log σ] = 0.97; ρ(σ, CGM (mg/dL)): median -0.05, 2/10 seeds with ρ > +0.1, 4/10 with ρ < −0.1; ρ(σ, CGM trend (mg/dL/min)): median +0.01, 2/10 seeds with ρ > +0.1, 0/10 with ρ < −0.1
