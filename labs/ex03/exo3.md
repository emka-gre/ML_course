****Problem set 3****

1 Least Squares and Linear Basis Functions Models
Exerice 2:
    - You can see that RMSE decreases as we increase the degree of the polynomial. Does it mean that the fit gets better as we increase the degree? Which fit is the best in your view?
    : => Yes exactly, as RMSE decreases the error gets smaller so the function fits better the data. As we increase the degree in polynomial basis function we create model that is more complicated with more parameter and weights so it could learn better the data but then we need to be careful to not create the model a bit too comlicated for the amout of the date that we have because if not we could have overfiting on those data and that would mean that it would generalize poorly. So the best solution is for me the polynomial degree 7.

2 Evaluating Model Prediction Performance
    - Do you thinkthat the order of samples is important when doing the split?
    : I think that it is important bcs i want to have the diversity of data on both of my splits/sets so f ex in classification if we take 0.80 of the dataset as train set and the rest the as test set and in those 0.2 would be only one class... that would not be that good... so we prepfer random divivison...
    
    - Look at the training and test RMSE for degree 3. Does this make sense? Why?
    : What am i suppose to observe? the fact that the RMSE is still bigger in compairason to other degrees? If that.. then it makes sense because the model is pretty simple so he s not able to learn that efficiently as those for complex models so the error is still quite bigger.

    - Now look at RMSE for the other two degrees. Do these make sense? Why?
    : Yes, it makes sense because the error is smaller when the model is more complex and thats understendable because he manages to learn the data a lot better. But then in case of the model being a bit too complex we have no error or super small error and that could signifz overfitting.

    - I think based on those rmse test and train we could constat that proportion 0.5 is the best because the error is relatively ok and also the difference between train and test results is not that huge so we dont observe overfitting...

    - The test RMSE for degree 12 is ridiculously high for the split 10%-90%. Why do you think this is the case?
    : Because of underfitting and overfitting.

    -BONUS: Imagine you have 5000 samples instead of 50. Which split might be better in that situation?
    : Then the the most complex one could be the best because with more data iven complex models work nicely.


THEORY EXERCICES:
 - Cost functions:
 ### (c) Implementing the gradient with matrix operations

From (b), the gradient is

$$\nabla_{\boldsymbol{w}} \mathcal{L} = \frac{2}{N} \sum_{n=1}^N \frac{\boldsymbol{x}_n^\top \boldsymbol{w} - y_n}{y_n^2 + \epsilon}\, \boldsymbol{x}_n$$

Let $\mathbf{X}$ be the $N \times D$ data matrix (row $n$ is $\boldsymbol{x}_n^\top$) and $\boldsymbol{y}$ the $N \times 1$ target vector. You can compute this in three steps, using $\odot$ for element-wise multiplication and $\oslash$ for element-wise division.

1. Compute the error vector $\boldsymbol{e} = \mathbf{X}\boldsymbol{w} - \boldsymbol{y}$, which has size $N \times 1$.
2. Compute the weight vector $\boldsymbol{c} = 1 \oslash (\boldsymbol{y} \odot \boldsymbol{y} + \epsilon)$, also $N \times 1$.
3. Compute the gradient $\nabla \mathcal{L} = \frac{2}{N}\, \mathbf{X}^\top (\boldsymbol{e} \odot \boldsymbol{c})$, which has size $D \times 1$.

The last step works because $\mathbf{X}^\top \boldsymbol{v}$ is exactly $\sum_n v_n \boldsymbol{x}_n$, a weighted sum of the rows of $\mathbf{X}$.

There's an equivalent form that uses summation over rows instead. Multiply each row of $\mathbf{X}$ by the corresponding entry of $\boldsymbol{e} \odot \boldsymbol{c}$ (broadcasting), then sum over the rows and multiply by $\frac{2}{N}$. In NumPy, that's `2/N * (X * (e*c)[:, None]).sum(axis=0)`.

The weights $\boldsymbol{c}$ don't depend on $\boldsymbol{w}$, so you can compute them once and reuse them at every gradient step.

### (d) Sensitivity to outliers

Take $y_n = 1$ and $\epsilon = 1$, so the denominator is $1^2 + 1 = 2$.

**Relative-error cost $\mathcal{L}_n$:**

- With $f = 10$, the cost is $\frac{(10-1)^2}{2} = \frac{81}{2} = 40.5$.
- With $f = 100$, the cost is $\frac{(100-1)^2}{2} = \frac{9801}{2} = 4900.5$.

That is about 121 times larger. The cost still grows quadratically in the error, so it is very sensitive to outliers.

**Log cost $\mathcal{L}'_n$**, using the natural log:

- With $f = 10$, the cost is $(\log 11 - \log 2)^2 = (\log 5.5)^2 \approx 1.705^2 \approx 2.91$.
- With $f = 100$, the cost is $(\log 101 - \log 2)^2 = (\log 50.5)^2 \approx 3.922^2 \approx 15.38$.

That is only about 5.3 times larger. The log compresses big values, so the cost grows roughly like $(\log f)^2$ instead of $f^2$.

**Interpretation, following the note:** the model focuses on the samples with the largest cost. With $\mathcal{L}_n$, a single wild prediction or outlier produces a huge cost and dominates training, pulling $\boldsymbol{w}$ toward it. With $\mathcal{L}'_n$, that sample's cost stays moderate, so the rest of the data still matters, which makes it much more robust.

The gradients show the same thing. For the log cost,

$$\nabla_{\boldsymbol{w}} \mathcal{L}'_n = \frac{2\big(\log(f+1) - \log(y_n+1)\big)}{f+1}\, \boldsymbol{x}_n$$

Because of the $\frac{1}{f+1}$ factor, the gradient actually shrinks as the prediction gets very large. For $\mathcal{L}_n$, the gradient keeps growing linearly with the error.

**One caveat:** $\mathcal{L}'_n$ only makes sense when $f(\boldsymbol{x}_n, \boldsymbol{w}) > -1$ and $y_n > -1$. It's suited to non-negative targets such as prices or counts. Averaged over the dataset, this loss is the square of the RMSLE, the root mean squared log error.
