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
- torque: median std_s(log σ) = 0.000; median σ far/near = 1.00; median executed-noise far/near = 0.70; median frac. of states with mean outside the action box = 0.06; within-seed (state) share of pooled Var[log σ] = 0.00

**state-dependent σ(s)** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (|θ| from upright (rad) ≷ 0.3) | ρ: |θ| from upright (rad) | ρ: |θ̇| (rad/s) | η²: |θ| from upright (rad) | η²: |θ̇| (rad/s) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | torque | 0.157 | 0.938 | 19.27 | 12.79 | +0.58 | +0.59 | 0.85 | 0.63 | 0.80 |
| 2 | torque | 0.038 | 0.637 | 6.23 | 3.09 | +0.74 | +0.74 | 0.77 | 0.58 | 0.69 |
| 3 | torque | 0.036 | 0.546 | 5.74 | 2.40 | +0.80 | +0.86 | 0.53 | 0.56 | 0.51 |
| 4 | torque | 0.062 | 0.736 | 7.94 | 5.62 | +0.10 | +0.59 | 0.89 | 0.66 | 0.86 |
| 5 | torque | 0.054 | 0.877 | 11.07 | 8.23 | +0.36 | +0.55 | 0.93 | 0.66 | 0.90 |
| 6 | torque | 0.036 | 0.658 | 6.24 | 3.14 | +0.82 | +0.78 | 0.75 | 0.66 | 0.65 |
- torque: median std_s(log σ) = 0.697; median σ far/near = 4.38; median executed-noise far/near = 3.83; median frac. of states with mean outside the action box = 0.03; within-seed (state) share of pooled Var[log σ] = 0.84; ρ(σ, |θ| from upright (rad)): median +0.66, 5/6 seeds with ρ > +0.1, 0/6 with ρ < −0.1; ρ(σ, |θ̇| (rad/s)): median +0.67, 6/6 seeds with ρ > +0.1, 0/6 with ρ < −0.1
