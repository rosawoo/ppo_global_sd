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
| 1 | torque | 0.835 | 1.272 | 62.17 | 29.76 | +0.70 | +0.55 | 0.89 | 0.69 | 0.81 |
| 2 | torque | 0.606 | 1.130 | 29.13 | 21.53 | +0.42 | +0.55 | 0.94 | 0.67 | 0.90 |
| 3 | torque | 0.404 | 0.997 | 18.43 | 11.81 | +0.09 | +0.64 | 0.78 | 0.70 | 0.76 |
| 4 | torque | 0.349 | 1.258 | 36.71 | 31.31 | +0.67 | +0.58 | 0.91 | 0.71 | 0.83 |
| 5 | torque | 0.381 | 0.986 | 20.51 | 13.94 | +0.22 | +0.54 | 0.89 | 0.67 | 0.84 |
| 6 | torque | 0.455 | 0.700 | 8.53 | 5.90 | +0.72 | +0.56 | 0.84 | 0.63 | 0.73 |
| 7 | torque | 0.457 | 1.463 | 71.34 | 37.86 | +0.61 | +0.67 | 0.91 | 0.62 | 0.83 |
| 8 | torque | 0.466 | 1.043 | 22.43 | 15.86 | +0.45 | +0.39 | 0.90 | 0.57 | 0.84 |
| 9 | torque | 0.490 | 0.854 | 13.71 | 8.97 | +0.36 | +0.46 | 0.74 | 0.60 | 0.71 |
| 10 | torque | 0.566 | 0.676 | 9.98 | 4.57 | +0.29 | +0.30 | 0.54 | 0.34 | 0.52 |
- torque: median std_s(log σ) = 1.020; median σ far/near = 14.90; median executed-noise far/near = 3.58; median frac. of states with mean outside the action box = 0.14; within-seed (state) share of pooled Var[log σ] = 0.85; ρ(σ, |θ| from upright (rad)): median +0.44, 9/10 seeds with ρ > +0.1, 0/10 with ρ < −0.1; ρ(σ, |θ̇| (rad/s)): median +0.55, 10/10 seeds with ρ > +0.1, 0/10 with ρ < −0.1

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
| 1 | basal_insulin | 0.610 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 2 | basal_insulin | 0.613 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 3 | basal_insulin | 0.572 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 4 | basal_insulin | 0.648 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 5 | basal_insulin | 0.709 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 6 | basal_insulin | 0.656 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 7 | basal_insulin | 0.817 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 8 | basal_insulin | 0.675 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 9 | basal_insulin | 0.781 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
| 10 | basal_insulin | 0.549 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
- basal_insulin: median std_s(log σ) = 0.000; median σ far/near = 1.00; median executed-noise far/near = 1.15; median frac. of states with mean outside the action box = 0.62; within-seed (state) share of pooled Var[log σ] = 0.00

**state-dependent σ(s)** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (CGM (mg/dL) ≷ 100.0) | ρ: CGM (mg/dL) | ρ: CGM trend (mg/dL/min) | η²: CGM (mg/dL) | η²: CGM trend (mg/dL/min) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | basal_insulin | 5.754 | 0.786 | 9.55 | 1.08 | +0.22 | -0.04 | 0.04 | 0.00 | 0.04 |
| 2 | basal_insulin | 5.598 | 0.884 | 12.10 | 1.00 | +0.08 | -0.02 | 0.03 | 0.00 | 0.01 |
| 3 | basal_insulin | 6.086 | 0.629 | 5.84 | 0.93 | +0.00 | +0.01 | 0.03 | 0.00 | 0.00 |
| 4 | basal_insulin | 5.741 | 0.946 | 14.75 | 0.81 | -0.11 | +0.10 | 0.08 | 0.01 | 0.01 |
| 5 | basal_insulin | 5.396 | 0.979 | 15.46 | 0.82 | -0.19 | +0.06 | 0.03 | 0.00 | 0.02 |
| 6 | basal_insulin | 5.393 | 0.994 | 16.35 | 0.88 | -0.13 | -0.02 | 0.02 | 0.00 | 0.01 |
| 7 | basal_insulin | 5.589 | 0.896 | 12.97 | 0.93 | +0.00 | +0.08 | 0.03 | 0.01 | 0.01 |
| 8 | basal_insulin | 5.773 | 0.754 | 8.43 | 0.92 | -0.09 | -0.01 | 0.01 | 0.00 | 0.00 |
| 9 | basal_insulin | 5.199 | 0.906 | 12.90 | 1.28 | +0.33 | -0.06 | 0.09 | 0.00 | 0.08 |
| 10 | basal_insulin | 5.500 | 0.822 | 9.51 | 0.78 | -0.28 | +0.04 | 0.06 | 0.00 | 0.03 |
- basal_insulin: median std_s(log σ) = 0.890; median σ far/near = 0.93; median executed-noise far/near = 1.09; median frac. of states with mean outside the action box = 0.94; within-seed (state) share of pooled Var[log σ] = 0.99; ρ(σ, CGM (mg/dL)): median -0.04, 2/10 seeds with ρ > +0.1, 4/10 with ρ < −0.1; ρ(σ, CGM trend (mg/dL/min)): median +0.00, 0/10 seeds with ρ > +0.1, 0/10 with ρ < −0.1
