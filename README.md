# Off-Diagonal Ridge Regularisation
Standard ridge regularisation is mathematically equivalent to the following procedure:
1. Make many copies of the dataset
2. Add uncorrelated noise to each variable (equivalent to adding a multiple of the identity matrix to the covariance matrix of the variables)
3. Compute the OLS estimator on this perturbed dataset

In practice, we are sometimes faced with datasets where certain variables will clearly be correlated with nearby variables. It might therefore make sense to link the noise with a kernel function describing the correlation of the noise added to different variables.

In this repo, I present an example of this modified procedure. For a hand-written digit dataset (`sklearn.datasets.load_digits`), I regress a one-hot encoded class label against the pixels in the corresponding images.

I present average-case LOOCV MSE across 174 multiple non-overlapping samples, each of size 30 (containing 3 instances of each class), for the following estimators:
1. $\beta=(\frac{1}{n-1}X^TX+\lambda I)(\frac{1}{n-1}X^Ty)$
2. $\beta=(\frac{1}{n-1}X^TX+\lambda M)(\frac{1}{n-1}X^Ty)$

In these formulas:
- $X$ is the feature matrix
- $y$ is the target vector
- $n=29$ is the size of the training dataset
- $I$ is the $64 \times 64$ identity matrix
- $M$ is a $64\times 64$ square matrix (indexed by points in the image), with $M_{i,j} = \exp\left(-\left(\frac{d(i,j)}{\text{lengthscale}}\right)^2\right)$, where $d(i,j)$ is the distance between two points in the image

$X$ and $y$ are standardised so that within each chunk, every variable is mean-zero and has unit variance (or is constant zero). This eliminates the need to consider an intercept term, as well as any concerns about excessively penalising low-variance, high-signal predictors.

Results by digit are shown below. Digits for which the second estimator outperforms (in some cases) are:
- 1
- 3
- 4 (for very small regularisation penalty)
- 8
- 9

![Results](https://github.com/odenpetersen/offdiagonal-ridge/raw/main/output/mse_loocv.png)

A lengthscale approaching zero will have the same behaviour as ridge regularisation, but these results suggest that lengthscales slightly above zero may outperform ridge regularisation depending on the strength of the penalty.

This is what the empirical covariance matrix of the pixels looks like (in the particular one-dimensional order I've arranged them), vs. the penalty matrix added for regularisation.
![Empirical Covariance vs. Penalty](https://github.com/odenpetersen/offdiagonal-ridge/raw/main/output/matrices.png)
