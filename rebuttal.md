# rebuttal

## RE: Reviewer u8Kq








## RE: Reviewer u1HY
### 1: The proposed method uses Generalized Estimating Equations (GEE) to handle intra-cluster correlations. However, the problem setting involves two types of correlations: within-episode correlations and between-episode correlations within a cluster. The paper does not specify how the method addresses within-episode correlations.


### 2: Regarding Theorem 1, does the result achieve semiparametric efficiency, or is it limited to comparing parameter estimators within the set of estimating equations specified in Equation (5)? The operator in of equation (5) is non-smooth. Does it cause any challenge to the statistical inference?


### 3: Regarding Theorem 2, first it seems that the regret is not properly defined, but only appears in the proof of Theorem 2 in the appendix. The value functions have not been defined either. In addition, the type of regret being discussed is unclear. It seems the focus is on simple regret, not cumulative regret, but this should be explicitly stated. Whichever is being addressed, it is important to connect the theoretical results to existing literature. For example, if the variance of is plugged in, what is the regret? How does the regret scale with key factors such as the dimension of the state space, episode length, and number of episodes?




## RE: Reviewer MTVu

### Significance justification
* If the covariance structure is well-specified, the results in Figure 5 make sense, though I am curious why GFQI does not get asymptotically better as the number of clusters increases. Additionally, the performance of GFQI is not consistently better under some conditions, which leads me to question when GFQI is most appropriate to use. I would appreciate additional discussion of this.
    - [Jitao] increase the number of clusters to show the regret of GFQI can decrease to 0
    - We appreciate the reviewer’s insightful observation regarding the asymptotic behavior of GFQI as the number of clusters increases. Upon closer examination, we confirm that GFQI indeed exhibits improved performance with an increasing number of clusters. However, the results in Figure 5 might appear to suggest otherwise due to the limited range of clusters shown in the current plot. As the number of clusters continues to grow beyond the range displayed in Figure 5, the regret of GFQI continues to decrease, supporting its asymptotic improvement.

To address this concern more explicitly, we *include* an extended plot in the revised manuscript that visualizes the behavior of GFQI with a significantly larger number of clusters. This plot will demonstrate that GFQI consistently achieves better performance as the sample size increases, particularly when the covariance structure is well-specified.

Additionally, regarding the observation that GFQI’s performance is not consistently better under all conditions, we note that the advantage of GFQI depends heavily on the intra-cluster correlation (𝜓) and the degree of misspecification of the covariance structure. GFQI provides substantial benefits when the intra-cluster correlation is strong or when correctly specified correlation structures allow for more efficient policy learning. In cases of weaker correlation or greater misspecification, the relative advantage of GFQI diminishes, which aligns with our theoretical findings.


### Clarity
*
*
* You don’t define many terms including regret, MLE, and DGP. Please define these terms early in the paper for reader clarity.
  RE: done. add a notion section
