Problem Set 2, Sept 17, 2026

**Linear Regression and Gradient Descent**

Exercice 1:
- column of X represent a feature/one description of all datapoints.. with 1 as an offset...
- Each row of X represent a datapoint what data that is being desribed bz beatures from different columns
- Thats an offset that is there for w0... so its not 0..
- Size of y would be 2x1 and the size of X would be 3x2, and X32 would represent : ....



Exercice 2:
b) No, it doesnt look like a good estimate because in the first plot shose elipses are not that smooth and also in the second plot we could see that the red line is not fitted perfectly, but if we change from 10 grid spacing to 50 we are able to observe few changes like for example tgose elipses in the first plet seems to be a lat smoother and also the red line is more properby fitted on the points.

c) 
- with coearse grid w spacing 50 we have a fit that is not that super accurate but at least its a lot faster then a finer grid with a spacing 10, that is more accurate but also slower to compute
- With different values of grid spacing we could observe that the estimation of weights gets more accurate with finer/smaller spacing but it has also a computational effect because it also gets a lot slower. On the other hand with a bigger spacing / coarse grid search we could observe that the accurasy its not that accurate but on the other hand its pretty fast.


Exercice 3:
b) From the gradients values we are able to compute the weights and then we could compute the norm and when its smaller its more accurate etc and also we could try to find with the help of that for example the optimal weights etc which are the most precise for us etc.....
They are bigger if the weights are bigger etc...
c)
- yes the lost is being minimized with the change of the weights but i would say that those weights dont change that much... That at the end we are not that much able to observe the change in weights but a little bit in losses.
- yes the algorithm is converging into the smallest loss accessible in the number of iterations with those weights that are available... At the beginning the convergence speed is pretty hight but then it goes rather slowly closely to the end.


remark: max error in the grid search is when the point falls in the middle of the grid... thats why is more accurate when the grid is finer but also then its also computationally more difficult

d)
- while observing the different step sizes we could see that while using super small step size, it wasnt reallz able to converge in those 50 iterations so those steps were a bit tooo small. Then with those in the middle we were able to converge in those 50 iterations and also were pretty precise... On the other hand when we look at the step size 2.5 i would constat that that is too big... in this case we got a loss pretty low, lower then with others bit probbaly we were jut lucky... because with the step that is this big its easy to just miss it...

- while changing the different weight we are able to observe that eveything converged into kind of similar values... and those 0 values were a bit slower....

Exercice 4:
- The difference between GD and SGD is that GD computes the gradient on all N = 10000 points at every iteration, while SGD (batch_size = 1) computes it from one random point. So one SGD iteration is ~10000x cheaper, but the gradient is only a noisy estimate of the true one.
- In the plot, GD walks almost straight to the minimum (the star) and stops moving after ~15 iterations (with gamma = 0.7 the weights are frozen at w = (73.29, 13.48), loss = 30.77). SGD zig-zags: each step goes in the direction given by a single random point, so the trajectory is curved / noisy.
- The steps are NOT equally spaced: each step has size gamma * |error_n| * ||x_n||, and the error shrinks as we get closer to the optimum, so the steps get smaller (from ~13 at the start to ~0.5 at the end), just not smoothly like in GD.
- SGD never fully settles: near the minimum the gradient of a single point is not zero, so with a constant step size it keeps bouncing around in a neighbourhood of the minimum. After 50 iterations we end up around w = (71, 14.5) with a full-data loss of ~37 instead of the optimal 30.77. To get the same precision as GD we would need more iterations or a decreasing step size.
- Note: the loss has to be computed on the full dataset (compute_loss(y, tx, w)), not on the batch. The loss on one point jumps between ~0 and ~45 from one iteration to the next and says nothing about the quality of the model.


Exercice 5:


6 - still equally spaced steps but more nosier...