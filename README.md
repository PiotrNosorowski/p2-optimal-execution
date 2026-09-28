# N-Player Optimal Execution: MARL vs Analytical Nash Equilibrium

## Table of Contents

1. [Introduction](#introduction)
2. [Breaking Down Terminal Liquidation](#breaking-down-terminal-liquidation)
3. [Breaking Down Running Inventory Penalty](#breaking-down-running-inventory-penalty)
4. [From Performance Criterion to the HJB Equation](#from-performance-criterion-to-the-hjb-equation)
5. [Deriving the Optimal Strategy and Ansatz](#deriving-the-optimal-strategy-and-ansatz)
6. [Analytical Nash Equilibrium](#analytical-nash-equilibrium)
7. [References](#references)



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

### Breaking Down Terminal Liquidation

$$\text{Terminal Value} = Q_T^{\nu} \left( S_T^{\nu} - \alpha Q_T^{\nu} \right)$$

At the end of the trading period ($t = T$), any remaining shares $Q_T^{\nu}$ must be sold immediately. The money received from this final sale is given by the formula above, where $\alpha$ (represented as `alpha` in the code) is the penalty parameter for selling leftover stock.

Expanding this formula splits the final value into two parts:

$$\text{Terminal Value} = Q_T^{\nu} S_T^{\nu} - \alpha (Q_T^{\nu})^2$$

* **Market Value ($Q_T^{\nu} S_T^{\nu}$):** The basic value of the remaining shares at the final market price $S_T^{\nu}$.
* **Terminal Penalty ($-\alpha (Q_T^{\nu})^2$):** A penalty for holding leftover stock at time $T$. As the remaining position $Q_T^{\nu}$ grows, the price discount per share increases linearly ($\alpha Q_T^{\nu}$), resulting in a quadratic penalty on the total value.

> [!NOTE]
> ### Role of $\alpha$
>
> The parameter $\alpha$ controls how strictly leftover inventory is penalized:
> 1. **Economic Meaning:** It represents the extra price drop caused by dumping all remaining shares at once at time $T$.
> 2. **Full Liquidation ($\alpha \to \infty$):** When $\alpha \to \infty$, the penalty for keeping any stock becomes infinite. This forces the trading strategy to sell all inventory during the period $[0, T]$, guaranteeing $Q_T^{\nu} = 0$.


### Breaking Down Running Inventory Penalty

While waiting to sell, holding unsold shares carries risk because stock prices can change. To account for this, the model adds an extra penalty cost over time:

$$\text{Running Inventory Penalty} = \phi \int_0^T (Q_t^{\nu})^2 \, dt$$

where $\phi$ (represented as `phi` in the code) sets how strongly the model penalizes holding stock.

Here is what each part means:

* **Shares Held ($Q_t^{\nu}$):** The number of unsold shares in your portfolio at time $t$.
* **Squared Penalty ($(Q_t^{\nu})^2$):** Makes holding large amounts of stock much more expensive. Holding 2,000 shares costs four times as much penalty as holding 1,000 shares.
* **Risk Parameter ($\phi$):** Controls how much the model fears holding stock. A higher $\phi$ forces the model to sell faster.
* **Total Over Time ($\int_0^T \dots dt$):** Adds up this penalty cost for every second from start ($t = 0$) to end ($t = T$).

> [!NOTE]
> ### Why Use a Squared Term ($\phi Q_t^2$)?
>
> Using $Q_t^2$ instead of just $Q_t$ does two main things:
> 1. **Prevents Big Losses:** Holding many shares carries higher price risk. Squaring the number of shares forces the model to treat big positions as very dangerous.
> 2. **Sells Fast First:** Because the penalty is highest when you hold many shares, the model sells quickly at the beginning and slows down as fewer shares remain.

### From Performance Criterion to the HJB Equation

The model begins with the expected total value formula $H^{\nu}$, which sums up all rewards and penalties from time $t$ to $T$:

$$H^{\nu}(t, x, S, q) = \mathbb{E}_{t,x,S,q} \left[ \underbrace{X_T^{\nu}}_{\text{Terminal Cash}} + \underbrace{Q_T^{\nu} \left(S_T^{\nu} - \alpha Q_T^{\nu}\right)}_{\text{Terminal Execution}} - \underbrace{\phi \int_t^T (Q_u^{\nu})^2 \, du}_{\text{Inventory Penalty}} \right]$$

To find the best outcome, we define the **Value Function** $V(t, x, S, q)$ as the maximum expected value achieved by choosing the optimal trading speed $\nu^*$:

$$V(t, x, S, q) = \sup_{\nu} H^{\nu}(t, x, S, q)$$

Applying Dynamic Programming yields the full Hamilton-Jacobi-Bellman (HJB) equation:

$$0 = \left( \partial_t + \frac{1}{2}\sigma^2 \partial_{SS} \right) V - \phi q^2 + \sup_{\nu} \left[ \left( \nu (S - f(\nu)) \partial_x - g(\nu) \partial_S - \nu \partial_q \right) V \right]$$

---

### Deriving the Optimal Strategy and Ansatz

#### 1. Isolating the Optimization Term to Find $\nu^*$
To maximize the HJB equation over $\nu$, we isolate the supremum term containing the trading speed:

$$\sup_{\nu} \left[ \left( \nu (S - f(\nu)) \partial_x - g(\nu) \partial_S - \nu \partial_q \right) V \right]$$

Plugging in linear temporary impact $f(\nu) = \kappa \nu$ and permanent impact $g(\nu) = b \nu$, the expression inside the bracket expands to:

$$\nu S \partial_x V - \kappa \nu^2 \partial_x V - b \nu \partial_S V - \nu \partial_q V$$

To find the optimal speed $\nu^*$, we set the derivative of this term with respect to $\nu$ equal to zero (First-Order Condition):

$$\frac{\partial}{\partial \nu} \left[ \nu S \partial_x V - \kappa \nu^2 \partial_x V - b \nu \partial_S V - \nu \partial_q V \right] = 0$$

$$S \partial_x V - 2\kappa \nu \partial_x V - b \partial_S V - \partial_q V = 0$$

Solving directly for $\nu$ yields the optimal trading speed formula:

$$\nu^* = \frac{(S \partial_x - b \partial_S - \partial_q) V}{2\kappa \partial_x V}$$

#### 2. Substituting $\nu^*$ Back into the Optimization Term
We take the original expression inside the supremum:

$$\left[ \left( \nu (S - f(\nu)) \partial_x - g(\nu) \partial_S - \nu \partial_q \right) V \right]$$

Plugging in linear temporary impact $f(\nu) = \kappa \nu$ and permanent impact $g(\nu) = b \nu$ expands this term to:

$$\nu (S \partial_x - b \partial_S - \partial_q) V - \kappa \nu^2 \partial_x V$$

Plugging our formula for $\nu^*$ in place of every $\nu$ gives:

$$\left( \frac{(S \partial_x - b \partial_S - \partial_q) V}{2\kappa \partial_x V} \right) (S \partial_x - b \partial_S - \partial_q) V - \kappa \left( \frac{(S \partial_x - b \partial_S - \partial_q) V}{2\kappa \partial_x V} \right)^2 \partial_x V$$

Simplifying both parts to a common denominator ($4\kappa \partial_x V$):

$$\frac{2 [(S \partial_x - b \partial_S - \partial_q) V]^2}{4\kappa \partial_x V} - \frac{[(S \partial_x - b \partial_S - \partial_q) V]^2}{4\kappa \partial_x V} = \frac{[(S \partial_x - b \partial_S - \partial_q) V]^2}{4\kappa \partial_x V}$$

Inserting this result back into the full HJB equation yields:

$$0 = \left( \partial_t + \frac{1}{2}\sigma^2 \partial_{SS} \right) V - \phi q^2 + \frac{[(S \partial_x - b \partial_S - \partial_q) V]^2}{4\kappa \partial_x V}$$

#### 3. Separating Cash from Risk (Ansatz)
At terminal time $T$, trading stops and the total portfolio value is known exactly from the terminal condition:

$$V(T, x, S, q) = x + qS - \alpha q^2$$

This reveals a natural structure: cash $x$, mark-to-market stock value $qS$, and a terminal inventory penalty $-\alpha q^2$. Extending this economic intuition backward to any time $t < T$ motivates our trial solution (Ansatz):

$$V(t, x, S, q) = x + qS + v(t, S, q)$$

where $v(T, S, q) = -\alpha q^2$. The terms represent:
* **$x + qS$:** Baseline portfolio value if all shares could be sold instantly at mid-price $S$ with zero market impact and zero risk.
* **$v(t, S, q)$:** The **inventory value function**. It extends the terminal penalty backward in time to track all future execution costs, price impact, and inventory risk from $t$ to $T$.

* **Why $v$ depends on $(t, S, q)$ and not $x$:**
  * **No $x$:** Cash carries zero execution risk or price impact ($\partial_x V = 1$), so its value is purely additive ($+x$) and drops out of the PDE completely.
  * **$(t, S, q)$:** These state variables carry all system dynamics—inventory risk ($\phi q^2$), remaining time ($T - t$), and market price ($S$).

#### 4. Substituting the Ansatz into the HJB Equation
We calculate the partial derivatives of $V(t, x, S, q) = x + qS + v(t, S, q)$:

* $\partial_x V = 1$
* $\partial_S V = q + \partial_S v$
* $\partial_q V = S + \partial_q v$
* $\partial_t V = \partial_t v$
* $\partial_{SS} V = \partial_{SS} v$

Plugging all partial derivatives directly into the complete optimization term $\frac{[(S \partial_x - b \partial_S - \partial_q) V]^2}{4\kappa \partial_x V}$:

$$\frac{[(S(1) - b(q + \partial_S v) - (S + \partial_q v))]^2}{4\kappa(1)}$$

Simplifying the numerator and squaring eliminates the negative sign:

$$\frac{\left( -\left[ b(q + \partial_S v) + \partial_q v \right] \right)^2}{4\kappa} = \frac{\left[ b(q + \partial_S v) + \partial_q v \right]^2}{4\kappa}$$

Substituting this expression back into the main HJB equation yields:

$$0 = \left( \partial_t + \frac{1}{2}\sigma^2 \partial_{SS} \right) v - \phi q^2 + \frac{\left[ b(q + \partial_S v) + \partial_q v \right]^2}{4\kappa}$$

#### 5. Reduction to $v(t, q)$
Since neither the PDE above nor the terminal condition $v(T, q) = -\alpha q^2$ explicitly depends on $S$, the function $v$ is independent of the share price ($v(t, S, q) = v(t, q)$). Thus, all price derivatives vanish:

$$\partial_S v = 0 \quad \text{and} \quad \partial_{SS} v = 0$$

Assuming zero permanent impact ($b = 0$), the term $bq$ vanishes as well, reducing the equation to its final form for $v(t, q)$:

$$0 = \partial_t v(t, q) - \phi q^2 + \frac{(\partial_q v(t, q))^2}{4\kappa}$$

#### 6. Expressing Optimal Speed in Terms of $v(t, q)$
We start from the general formula for optimal trading speed:

$$\nu^* = \frac{(S \partial_x - b \partial_S - \partial_q) V}{2\kappa \partial_x V}$$

Substituting our partial derivatives ($\partial_x V = 1$, $\partial_S V = q + \partial_S v$, $\partial_q V = S + \partial_q v$):

$$\nu^* = \frac{S(1) - b(q + \partial_S v) - (S + \partial_q v)}{2\kappa(1)}$$

Applying zero permanent impact ($b = 0$) and price independence ($\partial_S v = 0$) simplifies the expression directly to[cite: 1]:

$$\nu^*(t, q) = \frac{S(1) - 0 - (S + \partial_q v)}{2\kappa(1)} = -\frac{\partial_q v(t, q)}{2\kappa}$$

#### 7. Reducing the PDE to a Riccati ODE
Both the running inventory penalty ($\phi q^2$) and the terminal penalty ($-\alpha q^2$) are quadratic in $q$, so we propose a quadratic Ansatz for $v(t, q)$:

$$v(t, q) = -a(t) q^2$$

with terminal condition $a(T) = \alpha$, read off from $v(T, q) = -\alpha q^2$.

> [!NOTE]
> ### Why the minus sign?
> $v$ is a *value* (money gained), while holding inventory is a *cost*. Writing $v = -a q^2$ makes $a(t) > 0$ the cost per unit of squared inventory, which is exactly the variable `a` solved for in `ricatti.py`.

Its partial derivatives are:

* $\partial_t v = -\dot{a}(t) q^2$
* $\partial_q v = -2a(t) q$

Substituting these into the reduced PDE $0 = \partial_t v - \phi q^2 + \frac{(\partial_q v)^2}{4\kappa}$:

$$-\dot{a}(t) q^2 - \phi q^2 + \frac{(-2a(t) q)^2}{4\kappa} = 0$$

Squaring removes the minus sign, $(-2aq)^2 = 4a^2q^2$, so:

$$-\dot{a}(t) q^2 - \phi q^2 + \frac{a(t)^2}{\kappa} q^2 = 0$$

Factoring out $q^2$ (the equation must hold for every inventory level $q$) and solving for $\dot{a}$ gives a single Riccati Ordinary Differential Equation:

$$\dot{a}(t) = \frac{a(t)^2}{\kappa} - \phi, \qquad a(T) = \alpha$$

It is integrated backward in time from $T$ to $0$ and was implemented in Python (`riccati.py`):

```python
def da_dt(t, a):
    return a**2 / kappa - phi

ls = np.linspace(T, 0, 200)

solver = solve_ivp(da_dt, (T, 0), y0=[kappa], t_eval=ls,
                   rtol=1e-10, atol=1e-12)
```

`da_dt` is the right-hand side of the Riccati ODE, solved backward from the terminal condition $a(T) = \alpha$ (set to $\alpha = \kappa$ in the code) and evaluated at 200 points between $T$ and $0$.





## References

[1] Cartea, A., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
