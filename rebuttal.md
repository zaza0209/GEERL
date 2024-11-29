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
    - To address this concern more explicitly, we *include* an extended plot in the revised manuscript that visualizes the behavior of GFQI with a significantly larger number of clusters. This plot will demonstrate that GFQI consistently achieves better performance as the sample size increases, particularly when the covariance structure is well-specified.
    - Additionally, regarding the observation that GFQI’s performance is not consistently better under all conditions, we note that the advantage of GFQI depends heavily on the intra-cluster correlation (𝜓) and the degree of misspecification of the covariance structure. GFQI provides substantial benefits when the intra-cluster correlation is strong or when correctly specified correlation structures allow for more efficient policy learning. In cases of weaker correlation or greater misspecification, the relative advantage of GFQI diminishes, which aligns with our theoretical findings.

### Novelty Justification:
* I believe that using GEE to improve the sample efficiency of FQI is a novel methodological approach. However, I am unclear of the performance of GFQI (the proposed approach), when the correlation structure is misspecified.
    - We appreciate the reviewer’s acknowledgment of the novelty of using GEE to improve the sample efficiency of FQI. Regarding the performance of GFQI under misspecified correlation structures, we acknowledge that this is a critical aspect to consider.

    - To address this concern, we have added a figure in the revised manuscript that explicitly examines the performance of GFQI when the independence correlation structure is used as a misspecified working correlation matrix. This scenario effectively assumes no intra-cluster correlation, providing a clear comparison against scenarios with correctly specified working correlation structures.

    - Our preliminary results show that GFQI retains its robustness under misspecification, aligning with the theoretical guarantee of asymptotic normality. However, as expected, its efficiency (e.g., variance reduction and regret minimization) is reduced compared to the case of a well-specified structure. We believe this additional analysis will provide greater clarity on the practical applicability and limitations of GFQI when the correlation structure is misspecified.

    - Thank you for highlighting this important consideration. We will ensure that the revised manuscript reflects this additional evaluation.

### Clarity
*
* In Figure 2, why does the delta between FQI/GFQI get smaller as the number of clusters increase? Is it because with a larger number of clusters, there is more shared structure between clusters that an algorithm can capitalize on? Is the distance between the clusters a factor here? Also is is the performance of GFQI similar to FQI if there is 1 cluster?
    - Thank you for the insightful questions regarding the trends observed in Figure 2.

    - The decreasing gap between FQI and GFQI as the number of clusters increases is primarily due to the properties of GEE. GFQI incorporates a working correlation matrix to account for intra-cluster correlations, providing efficiency gains over FQI (which corresponds to GEE with an independence working correlation matrix) when the intra-cluster correlation is relatively high compared to the sample size. However, as the number of clusters increases, the total sample size grows, and both FQI and GFQI converge to the same consistent estimates. This is because GEE’s consistency property ensures that, regardless of the working correlation matrix, the estimates converge to the true parameter values as the sample size increases. Hence, the performance gap diminishes as both methods converge.

    - This observation aligns with the theoretical results in Theorem 1, where the convergence rate for all GFQI estimators, regardless of the working correlation matrix, is of the order `O((MN)^{-1/2})`, where `N` is the number of state-action-reward-next-state tuples in the dataset, which depends on the number of clusters. As for the distance between clusters, GEE assumes that different clusters are independent, and thus it does not directly model the distance or relationship between clusters.

    - Regarding the case of a single cluster, GFQI and FQI are likely to perform similarly in practice. While it is theoretically possible to model intra-cluster correlation within a single cluster using a working correlation matrix, the efficiency gains depend on the size of the cluster and the strength of the correlation. If the intra-cluster correlation is strong and the cluster size is relatively small, the correct working correlation matrix could provide some efficiency gains by decorrelating the data. However, in most cases, a single cluster offers limited scope for leveraging such structures, making the performance of GFQI and FQI appear similar. Additionally, as the cluster size grows, both methods converge to the same consistent estimates, further reducing the potential difference.





   
* You don’t define many terms including regret, MLE, and DGP. Please define these terms early in the paper for reader clarity.
    - done. add a notion section
    - 
* What is the convergence criteria?
    - Thank you for pointing out the need for clarification on the convergence criteria.

    - The convergence criteria in our method are as follows: the algorithm is considered to have converged when the predicted responses from two consecutive fitted models have a relative difference smaller than `10^{-5}`, or when the maximum number of iterations (100) is reached. These criteria ensure both accuracy and computational feasibility during optimization.

    - We will include this detail in the revised manuscript to provide clarity on this aspect of the method.
      

### Additional Comments:
* How do Generalized Estimating Equations work when the relationship between X and Y is not linear? I imagine there are several examples in which a linear structure is too restrictive. The same is true for the assumption on the optimal Q-function.
    - We appreciate the reviewer’s insightful observation about the potential limitations of linear assumptions in both the Generalized Estimating Equations (GEE) framework and the optimal Q-function.

    - For the concern regarding the linear relationship in GEE, we agree that linear models can sometimes be restrictive. However, GEE can naturally accommodate non-linear relationships through the use of appropriate link functions (e.g., logit, probit) and advanced modeling techniques such as basis expansions, splines, or other non-linear transformations. In our paper, we leverage basis functions to approximate the Q-function, which effectively extends the framework beyond a strictly linear structure.

    - Regarding the assumption of linear combinations for the optimal Q-function, we would like to emphasize that our approach allows for significant flexibility. By approximating the Q-function as a linear combination of basis functions, the method can model complex relationships. Importantly, this approximation can capture any non-linear structure in the Q-function if the number of basis functions is allowed to grow sufficiently large. This ensures that our method remains highly flexible and theoretically robust even in scenarios with non-linear relationships.

    - We will revise the manuscript to clarify these points and highlight the flexibility of our approach in handling non-linear structures. Thank you for bringing up this important perspective, which allows us to strengthen the clarity and positioning of our work.
