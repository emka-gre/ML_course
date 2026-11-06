## 1. What the SVM objective says

$$
\min_{w}\; \frac{1}{N}\sum_{n=1}^N \ell(y_n x_n^\top w) + \frac{\lambda}{2}\|w\|^2, \qquad \ell(z) = \max\{0,\, 1-z\}
$$

The classifier predicts $\text{sign}(x_n^\top w)$. The quantity $z_n = y_n x_n^\top w$ is the **margin** of point $n$:

- $z_n > 0$: the point is classified correctly.
- $z_n \ge 1$: the point is correct with room to spare.

The **hinge loss** $\max\{0, 1-z\}$ turns this into a penalty:

- If $z_n \ge 1$, the loss is 0. The point is safely on the right side and is ignored.
- If $z_n < 1$, the loss grows linearly. The point is wrong, or correct but too close to the boundary.

So, unlike logistic regression, the SVM doesn't care about points that are already well classified. Only the "difficult" points matter. These become the support vectors.

The **regularizer** $\frac{\lambda}{2}\|w\|^2$ keeps $w$ small. Geometrically, the width of the margin band is $2/\|w\|$, so a small $\|w\|$ means a wide margin. The objective is a trade-off: get the points out of the margin, but keep the margin wide. $\lambda$ controls that trade-off.

## 2. Question 1: accuracy and primal objective

Both are direct translations of the formulas:

```python
def calculate_accuracy(y, X, w):
    predictions = np.where(X @ w >= 0, 1, -1)
    return np.mean(predictions == y)

def calculate_primal_objective(y, X, w, lambda_):
    hinge = np.maximum(0, 1 - y * (X @ w))
    return np.mean(hinge) + lambda_ / 2 * np.sum(w**2)
```

`y * (X @ w)` computes all margins $z_n$ at once.

## 3. Question 2: the stochastic (sub)gradient

**The problem.** The hinge loss has a kink at $z=1$, so it has no derivative there. We use a **subgradient** instead. For a convex function, this is any slope of a line that touches the function from below. Away from the kink it is just the normal derivative. At the kink you can pick any value in between, and in practice you pick one of the two sides.

**The derivative.** Use the chain rule with $z = y_n x_n^\top w$, so $\nabla_w z = y_n x_n$:

$$
\nabla_w\, \ell(y_n x_n^\top w) =
\begin{cases}
-\,y_n x_n & \text{if } y_n x_n^\top w < 1 \\
0 & \text{if } y_n x_n^\top w \ge 1
\end{cases}
$$

Add the regularizer's gradient $\lambda w$. The stochastic gradient for a randomly picked $n$ is

$$
\boxed{g_n(w) = \mathbb{1}[y_n x_n^\top w < 1]\,(-y_n x_n) + \lambda w}
$$

**Why it's valid.** $n$ is uniform over $\{1,\dots,N\}$, so $\mathbb{E}_n[g_n(w)]$ equals a subgradient of the full objective. The $\frac1N\sum$ becomes an expectation. On average, you step in the right direction.

```python
def calculate_stochastic_gradient(y, X, w, lambda_, n, num_examples):
    x_n, y_n = X[n], y[n]
    grad = lambda_ * w
    if y_n * (x_n @ w) < 1:
        grad = grad - y_n * x_n
    return grad
```

Your template's signature may differ slightly. Note that because the objective already has the $\frac{1}{N}$, no factor of $N$ is needed here.

## 4. The SGD update and its intuition

$$
w \leftarrow w - \eta_t\, g_n(w)
$$

Written out:

- **Point violates the margin:** $w \leftarrow (1-\eta_t\lambda)\,w + \eta_t\, y_n x_n$. Shrink $w$ slightly, and rotate it toward classifying $x_n$ correctly.
- **Point is safe:** $w \leftarrow (1-\eta_t\lambda)\,w$. Only shrink, which widens the margin.

The step size $\eta_t$ must decrease over time, for example $\eta_t \propto 1/t$. Without that, the random choice of $n$ keeps $w$ jittering around the optimum forever.

## 5. Connection to the coordinate ascent part

Compare this with the dual update from before, $w \leftarrow w + \frac{\Delta\alpha_n}{\lambda N} y_n x_n$. **Both methods move $w$ along $y_n x_n$ for one random point.** The differences:

| | SGD (primal) | Coordinate ascent (dual) |
|---|---|---|
| Step size | You choose $\eta_t$, must tune and decrease it | Exact closed-form optimum, no tuning |
| Progress | Noisy, objective can go up | Dual objective never decreases |
| Stopping criterion | None built in | Duality gap tells you how far you are |
| Convergence | Slow ($O(1/t)$ at best) | Typically much faster here |

That is why your coordinate ascent run reached a duality gap of nearly 0 at a primal value of about 0.25 in half a second. If you compare, SGD's final primal objective should be at or above that value, because 0.25 is the true optimum.