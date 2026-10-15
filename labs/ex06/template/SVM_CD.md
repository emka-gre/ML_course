## 1. The big picture: why a "dual" problem?

The primal SVM problem (equation (1) in your assignment) is

$$
\min_{w}\; P(w) = \frac{1}{N}\sum_{n=1}^N \max\big(0,\; 1 - y_n x_n^\top w\big) + \frac{\lambda}{2}\|w\|^2
$$

The hinge loss has a kink, so it isn't differentiable everywhere. SGD can handle this, but it needs a step size and progresses slowly.

The trick is to write the hinge loss as a maximization:

$$
\max(0, z) = \max_{\alpha \in [0,1]} \alpha z
$$

If $z>0$, the best $\alpha$ is 1. If $z<0$, the best $\alpha$ is 0. That gives one variable $\alpha_n \in [0,1]$ per data point. Swap the min and max, solve the inner minimization over $w$ (set the gradient to 0), and you get two results:

- the mapping $w(\alpha) = \frac{1}{\lambda N} X^\top Y \alpha$
- the dual problem (2)

The dual is a smooth concave quadratic with simple box constraints $0 \le \alpha_n \le 1$. That structure is what makes coordinate ascent easy.

A useful rewrite: $\alpha^\top YXX^\top Y\alpha = \|X^\top Y\alpha\|^2 = \lambda^2N^2\|w(\alpha)\|^2$. So

$$
f(\alpha) = \frac{1}{N}\sum_n \alpha_n - \frac{\lambda}{2}\|w(\alpha)\|^2
$$

**What $\alpha_n$ means.** Each $\alpha_n$ says how much point $n$ "pushes" on the classifier:
- $\alpha_n = 0$: the point is correctly classified with margin > 1. It is not a support vector.
- $0<\alpha_n<1$: the point sits exactly on the margin.
- $\alpha_n = 1$: the point violates the margin.

## 2. Question 1: the coordinate update

Change only coordinate $n$ by $\gamma$. Then $w$ changes by $\frac{\gamma}{\lambda N} y_n x_n$, and expanding gives

$$
f(\alpha+\gamma e_n) = \text{const} + \frac{\gamma}{N}\big(1 - y_n x_n^\top w\big) - \frac{\gamma^2\|x_n\|^2}{2\lambda N^2}
$$

This is a downward parabola in $\gamma$. Setting the derivative to 0 gives

$$
\gamma = \frac{\lambda N\,(1 - y_n x_n^\top w)}{\|x_n\|^2}
$$

Now the constraint. For a 1D concave parabola restricted to an interval, the best point is the unconstrained peak clipped into the interval. So

$$
\boxed{\alpha_n^{\text{new}} = \min\!\Big(1,\ \max\!\Big(0,\ \alpha_n + \frac{\lambda N\,(1 - y_n x_n^\top w)}{\|x_n\|^2}\Big)\Big)}
$$

All other coordinates stay the same.

**Intuition.** $y_n x_n^\top w$ is the margin of point $n$.
- If the margin is below 1, the point is violating, so $\alpha_n$ goes up and the point pushes harder.
- If the margin is above 1, the point is safe, so $\alpha_n$ goes down toward 0.

**Cost.** You need one dot product $x_n^\top w$ and one norm $\|x_n\|^2$. Both are $O(D)$.

## 3. Question 2: updating $w$ cheaply

Don't recompute $X^\top Y\alpha$, which costs $O(ND)$. Only one coordinate changed, so

$$
w^{\text{new}} = w + \frac{\alpha_n^{\text{new}} - \alpha_n}{\lambda N}\, y_n x_n
$$

This is also $O(D)$.

## 4. Your plot: duality and the gap

The key fact is **weak duality**. For any feasible $\alpha$,

$$
\underbrace{f(\alpha)}_{\text{dual}} \;\le\; P(w^*) \;\le\; \underbrace{P(w(\alpha))}_{\text{primal}}
$$

The dual is always a lower bound on the optimum, and the primal is always an upper bound. For SVMs, **strong duality** holds: at the optimum both curves meet at the same value. The difference between them, the **duality gap**, is a certificate of how far you are from optimal.

Here is how that explains each part of your plot:

- **The orange (dual) curve only goes up.** Each step exactly maximizes $f$ along one coordinate, so $f$ can never decrease. It starts near 0 because $\alpha = 0$ gives $f = 0$, which can't be shown on a log scale. After the first update it is tiny (~1e-4).
- **The blue (primal) curve starts at exactly 1.** With $\alpha=0$, you have $w=0$. Every point has margin 0, so each hinge loss is 1 and the regularizer is 0. That gives $P = 1 = 10^0$.
- **The blue curve is jumpy.** Coordinate ascent guarantees improvement of the dual, not the primal. Each update moves $w$ along one data vector $x_n$. That can flip the margins of many other points at once, so the hinge loss sometimes spikes.
- **The curves meet at about 0.25 near $10^5$ iterations.** The gap goes to 0. This confirms strong duality, and it means you have provably reached the optimum of the SVM objective. So the answer to "Is the gap going to 0?" is **yes**. The 91.56% training accuracy is the accuracy of that optimal $w$.

## 5. Comparison with SGD (Question 4)

Compare the final primal value $P(w)$ from your SGD run with the ~0.25 you got here.

Coordinate ascent typically wins for three reasons:
- It needs no step size, because each update is an exact closed-form optimum.
- The dual improves monotonically.
- The duality gap tells you when to stop.

SGD on the primal gives you no such certificate. With a decreasing step size it converges slowly, so it usually ends at a slightly higher primal value after the same time.

## 6. The warnings

The `predictions = np.where(X @ w >= 0, ...)` lines are runtime warnings from the matrix product; the actual message is cut off in your screenshot. On Macs with recent NumPy, `@` often emits spurious "divide by zero / overflow encountered in matmul" warnings even though the result is correct. Your accuracy looks sensible, so they're very likely harmless. If you want to be sure, check `np.isfinite(w).all()`.