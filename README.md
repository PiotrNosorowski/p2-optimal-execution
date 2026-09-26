# N-Player Optimal Execution: MARL vs Analytical Nash Equilibrium

## Introduction

The aim of the project was to examine the convergence between Multi-Agent Reinforcement Learning approach and mathematical ground truth of position liquidation. The agents do not observe each others' actions or positions, yet got a feedback from the environment they influence by constant sell.

Following the framework established by Cartea et al. (2015), the primary objective of a trading agent is to maximize the expected execution performance criterion over a finite time horizon $[0, T]$:

$$H^{\nu}(t, x, S, q) = \mathbb{E}_{t,x,S,q} \left[ \underbrace{X_T^{\nu}}_{\text{Terminal Cash}} + \underbrace{Q_T^{\nu} \left(S_T^{\nu} - \alpha Q_T^{\nu}\right)}_{\text{Terminal Execution}} - \underbrace{\varphi \int_t^T (Q_u^{\nu})^2 \, du}_{\text{Inventory Penalty}} \right]$$

where:
* $X_T^{\nu}$ – cash balance at time $T$,
* $Q_T^{\nu}$ – remaining inventory at time $T$,
* $S_T^{\nu}$ – asset mid-price at time $T$,
* $\alpha$ – terminal liquidation penalty (`alpha`),
* $\varphi$ – running inventory risk penalty (`varphi`).

### Breaking Down Terminal Cash ($X_T^{\nu}$)

The accumulation of cash $X_T^{\nu}$ is modeled by integrating the instant revenue generated from selling inventory at speed $\nu_t$:

$$dX_t^{\nu} = \hat{S}_t^{\nu} \nu_t \, dt$$

where $\hat{S}_t^{\nu}$ represents the effective fill price. 

Accounting for a constant bid-ask spread $\Delta$ and a temporary price impact function $f(\nu_t)$, the general execution price for both acquisition (+) and liquidation (-) is formulated as:

$$\hat{S}_t^{\nu} = S_t^{\nu} \pm \left( \frac{1}{2}\Delta + f(\nu_t) \right)$$

Since this model focuses explicitly on a **liquidation** strategy (selling inventory at speed $\nu_t$), the execution price subtracts both spread and temporary impact penalties:

$$\hat{S}_t^{\nu} = S_t^{\nu} - \frac{1}{2}\Delta - f(\nu_t)$$

Substituting $\hat{S}_t^{\nu}$ yields the explicit differential cash dynamics:

$$dX_t^{\nu} = \left( S_t^{\nu} - \frac{1}{2}\Delta - f(\nu_t) \right) \nu_t \, dt$$

Assuming a linear temporary market impact $f(\nu_t) = \kappa \nu_t$ (represented as `kappa` in the codebase):

$$dX_t^{\nu} = S_t^{\nu} \nu_t \, dt - \frac{1}{2}\Delta \nu_t \, dt - \kappa \nu_t^2 \, dt$$

By the **Fundamental Theorem of Calculus**, integrating the differential $dX_t^{\nu}$ over the interval $[0, T]$ recovers the total change in cash ($\int_0^T dX_t^{\nu} = X_T^{\nu} - X_0^{\nu}$). Isolating the terminal state $X_T^{\nu}$ pulls the initial value $X_0^{\nu}$ outside the integral:

$$X_T^{\nu} = X_0^{\nu} + \int_0^T dX_t^{\nu} = X_0^{\nu} + \int_0^T \left( S_t^{\nu} \nu_t - \frac{1}{2}\Delta \nu_t - \kappa \nu_t^2 \right) dt$$

Assuming zero initial cash ($X_0^{\nu} = 0$) and setting aside fixed spread costs, substituting $dX_t^{\nu}$ directly yields:

> [!NOTE]
> ### Justification for Setting Aside the Spread Term ($\Delta$)
>
> The cumulative revenue loss due to the bid-ask spread over $[0, T]$ is given by:
>
> $$\int_0^T \frac{1}{2}\Delta \nu_t \, dt = \frac{1}{2}\Delta \int_0^T \nu_t \, dt$$
>
> Since $\nu_t = -\dot{Q}_t$, applying the Fundamental Theorem of Calculus yields:
>
> $$\int_0^T \frac{1}{2}\Delta \nu_t \, dt = \frac{1}{2}\Delta (Q_0 - Q_T)$$
>
> Under full liquidation ($Q_T = 0$), this term collapses to the deterministic constant $\frac{1}{2}\Delta Q_0$. Because $Q_0$ and $\Delta$ are fixed, exogenous parameters, this constant offset does not depend on the control trajectory $\nu_t$. Consequently, its derivative with respect to $\nu_t$ is zero, leaving the optimal execution policy $\nu_t^*$ unaffected. It can therefore be set aside during functional optimization without loss of generality.






## References

[1] Cartea, A., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
