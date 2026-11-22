### parking-v0

**global σ** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (distance to goal (m) ≷ 3.0) | ρ: distance to goal (m) | ρ: |heading error| (rad) | ρ: speed (m/s) | η²: distance to goal (m) | η²: |heading error| (rad) | η²: speed (m/s) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | acceleration | 0.215 | 0.000 | 1.00 | nan | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | steering | 0.201 | 0.000 | 1.00 | nan | +nan | +nan | +nan | 0.00 | 0.00 | 0.00 | 0.00 |
- acceleration: median std_s(log σ) = 0.000; median σ far/near = nan; median executed-noise far/near = nan; median frac. of states with mean outside the action box = 0.91; within-seed (state) share of pooled Var[log σ] = 0.00
- steering: median std_s(log σ) = 0.000; median σ far/near = nan; median executed-noise far/near = nan; median frac. of states with mean outside the action box = 0.26; within-seed (state) share of pooled Var[log σ] = 0.00

**state-dependent σ(s)** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (distance to goal (m) ≷ 3.0) | ρ: distance to goal (m) | ρ: |heading error| (rad) | ρ: speed (m/s) | η²: distance to goal (m) | η²: |heading error| (rad) | η²: speed (m/s) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | acceleration | 0.102 | 0.266 | 2.50 | nan | +0.39 | +0.28 | -0.19 | 0.19 | 0.20 | 0.17 | 0.25 |
| 1 | steering | 0.128 | 0.292 | 2.57 | nan | +0.20 | +0.14 | -0.60 | 0.18 | 0.20 | 0.45 | 0.38 |
- acceleration: median std_s(log σ) = 0.266; median σ far/near = nan; median executed-noise far/near = nan; median frac. of states with mean outside the action box = 0.85; within-seed (state) share of pooled Var[log σ] = 1.00; ρ(σ, distance to goal (m)): median +0.39, 1/1 seeds with ρ > +0.1, 0/1 with ρ < −0.1; ρ(σ, |heading error| (rad)): median +0.28, 1/1 seeds with ρ > +0.1, 0/1 with ρ < −0.1; ρ(σ, speed (m/s)): median -0.19, 0/1 seeds with ρ > +0.1, 1/1 with ρ < −0.1
- steering: median std_s(log σ) = 0.292; median σ far/near = nan; median executed-noise far/near = nan; median frac. of states with mean outside the action box = 0.18; within-seed (state) share of pooled Var[log σ] = 1.00; ρ(σ, distance to goal (m)): median +0.20, 1/1 seeds with ρ > +0.1, 0/1 with ρ < −0.1; ρ(σ, |heading error| (rad)): median +0.14, 1/1 seeds with ρ > +0.1, 0/1 with ρ < −0.1; ρ(σ, speed (m/s)): median -0.60, 0/1 seeds with ρ > +0.1, 1/1 with ρ < −0.1

### simglucose-adolescent1-v0

**global σ** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (CGM (mg/dL) ≷ 100.0) | ρ: CGM (mg/dL) | ρ: CGM trend (mg/dL/min) | η²: CGM (mg/dL) | η²: CGM trend (mg/dL/min) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | basal_insulin | 0.125 | 0.000 | 1.00 | 1.00 | +nan | +nan | 0.00 | 0.00 | 0.00 |
- basal_insulin: median std_s(log σ) = 0.000; median σ far/near = 1.00; median executed-noise far/near = 1.06; median frac. of states with mean outside the action box = 0.07; within-seed (state) share of pooled Var[log σ] = 0.00

**state-dependent σ(s)** — per training seed, stoch eval episodes, last 1 evals

| seed | action | mean σ | std_s(log σ) | σ p95/p5 | σ far/near (CGM (mg/dL) ≷ 100.0) | ρ: CGM (mg/dL) | ρ: CGM trend (mg/dL/min) | η²: CGM (mg/dL) | η²: CGM trend (mg/dL/min) | R² (linear, all vars) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | basal_insulin | 0.964 | 2.041 | 1096.63 | 0.21 | -0.11 | -0.08 | 0.02 | 0.01 | 0.03 |
- basal_insulin: median std_s(log σ) = 2.041; median σ far/near = 0.21; median executed-noise far/near = 0.42; median frac. of states with mean outside the action box = 0.03; within-seed (state) share of pooled Var[log σ] = 1.00; ρ(σ, CGM (mg/dL)): median -0.11, 0/1 seeds with ρ > +0.1, 1/1 with ρ < −0.1; ρ(σ, CGM trend (mg/dL/min)): median -0.08, 0/1 seeds with ρ > +0.1, 0/1 with ρ < −0.1
