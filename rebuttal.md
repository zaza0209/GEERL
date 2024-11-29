# rebuttal

## RE: Reviewer u8Kq

### Significance Justification:
* If I understand correctly, in the numerical study, a single agent learns a generalized policy for each baseline algorithm, which can be applied across different teams or clusters. While training one model is computationally more efficient than training separate models for each cluster, this approach might limit the agent's ability to optimize performance for specific clusters with unique characteristics, as it aims for a one-size-fits-all policy. This appears to be a significant advantage of GFQI compared to the baseline approaches. Training a separate agent for each cluster could allow the learning process to better adapt to the specific dynamics and characteristics of individual clusters, potentially improving performance. Could the authors please clarify or provide additional insights on this aspect?


Thank you for this thoughtful observation regarding the trade-off between training a generalized policy across clusters and training separate policies for each cluster.

You are correct that in the numerical study, a single agent learns a generalized policy for each baseline algorithm, which can then be applied across different teams or clusters. This approach is computationally efficient and aligns with the primary goal of our study: to design a method that accommodates intra-cluster correlations while maintaining scalability. GFQI’s strength lies in its ability to leverage information across clusters through the working correlation matrix, effectively addressing shared structures and dependencies, which are often present in real-world clustered data.

However, we agree that training separate models for each cluster could, in principle, allow for better adaptation to the unique dynamics and characteristics of individual clusters. While this might improve performance in cases where clusters exhibit highly distinct behaviors, it also comes with notable drawbacks:
1. **Increased Computational Cost:** Training separate agents for each cluster can significantly increase computational demands, especially when the number of clusters is large.
2. **Data Limitations:** Many real-world scenarios involve clusters with limited data. Training a separate model for each cluster could lead to overfitting or suboptimal performance due to insufficient data within individual clusters.
3. **Loss of Generalization:** A cluster-specific approach might not generalize well to new or unseen clusters, limiting its applicability in scenarios where policy transferability is important.

GFQI strikes a balance by learning a generalized policy that incorporates intra-cluster correlations, making it robust to shared structures across clusters while retaining computational feasibility. We recognize that for settings where clusters have drastically different characteristics and sufficient data is available, training cluster-specific models might offer complementary benefits. This could be an interesting direction for future work, where GFQI could be extended to hybrid approaches that combine cluster-specific and generalized policies.

We will revise the manuscript to clarify these points and include a discussion on the trade-offs between generalized and cluster-specific policies. Thank you for raising this important aspect, which helps us provide a more comprehensive perspective on the strengths and limitations of our approach.




### Non-conventional Contributions:
* The integration of generalized estimating equations (GEE), a classical statistical tool for longitudinal and clustered data, into the reinforcement learning (RL) framework is unconventional. This approach bridges the gap between statistical methods for correlated data and modern RL techniques, addressing a unique set of challenges rarely explored in RL literature.

* The paper tackles the underexplored problem of learning optimal policies in environments where data exhibits intra-cluster correlations. Such settings are common in real-world applications like healthcare and education, yet traditional RL methods typically assume independent and iid data, overlooking these dependencies.


    - Thank you for recognizing the non-conventional contributions of our work. We appreciate the acknowledgement of the novelty in integrating generalized estimating equations (GEE) into the reinforcement learning (RL) framework.

    - As highlighted, this integration represents a significant step forward in bridging classical statistical methods for correlated data with modern RL techniques. By leveraging the strengths of GEE, our approach directly addresses the challenges posed by intra-cluster correlations, which are prevalent in many real-world applications such as healthcare, education, and social sciences. Traditional RL algorithms often assume data independence, which limits their applicability in these settings. GFQI fills this gap by providing a robust and efficient way to learn optimal policies in environments with clustered data.

    - We also appreciate your recognition of the broader impact of tackling the problem of intra-cluster correlations in policy learning. By introducing a method that explicitly models these dependencies, our work opens new avenues for applying RL to practical problems where such correlations cannot be ignored. We will ensure these contributions are clearly emphasized in the revised manuscript to reflect their importance and relevance to both the statistical and RL communities.

    - Thank you for the thoughtful feedback and for highlighting the significance of these contributions.



### Clarity
* In my opinion, the paper can be improved in terms of clarity. The introduction section is not clear. After reading the intro I still don’t understand the problem. There is no clear problem statement. Only after reading section 3, that the problem become clear. The authors can improve the clarity of the paper by discussing that traditional RL methods typically assume iid data which is not the case here.



## RE: Reviewer u1HY
### Soundness justification
* The proposed method uses Generalized Estimating Equations (GEE) to handle intra-cluster correlations. However, the problem setting involves two types of correlations: within-episode correlations and between-episode correlations within a cluster. The paper does not specify how the method addresses within-episode correlations.
    - Thank you for highlighting this important distinction between within-episode and between-episode correlations within a cluster.

    - In our work, the proposed method focuses exclusively on addressing between-episode correlations within a cluster using Generalized Estimating Equations (GEE). These between-episode correlations arise naturally in clustered data due to shared characteristics or external factors that influence multiple episodes within the same cluster. By incorporating a working correlation matrix, our method models these dependencies, leading to improved efficiency in policy estimation.

    - However, we do not explicitly address within-episode correlations. The Markov Decision Process (MDP) assumption, which underpins our method, inherently assumes conditional independence between transitions within the same episode, given the current state-action pair. As a result, any potential within-episode correlation is not modeled in our framework.

    - We acknowledge this as a limitation of the current approach and an important direction for future work. Developing methods that can jointly model both within-episode and between-episode correlations would be an interesting extension to better capture complex dependency structures in reinforcement learning with clustered data.

    - We will clarify this point in the revised manuscript to ensure transparency about the scope and limitations of our method. Thank you for bringing this to our attention.


* Regarding Theorem 1, does the result achieve semiparametric efficiency, or is it limited to comparing parameter estimators within the set of estimating equations specified in Equation (5)? The operator in of equation (5) is non-smooth. Does it cause any challenge to the statistical inference?
    - The result is limited to comparing parameter estimators within the set of estimating equations specified in Equation (5).
    - Yes. We adopt the uniquess assumption (A3) and the marginal assumption to address this challenge.


* Regarding Theorem 2, first it seems that the regret is not properly defined, but only appears in the proof of Theorem 2 in the appendix. The value functions have not been defined either. In addition, the type of regret being discussed is unclear. It seems the focus is on simple regret, not cumulative regret, but this should be explicitly stated. Whichever is being addressed, it is important to connect the theoretical results to existing literature. For example, if the variance of $\beta$ is plugged in, what is the regret? How does the regret scale with key factors such as the dimension of the state space, episode length, and number of episodes?
      - If the covariance matrix of the temporal difference error is pluged in, the regret is still $-\frac{1}{2}\mathrm{tr}(\Var(\widehat{\bftheta})H)+O(N^{-3/2})$ as in Theorem 2 we do not assume the correct covariance matrix is used.
      - we can prove that the regret is
  $$
\begin{array}{r}
\phi_L^{\top}(a, s) \widehat{\beta}_{\left[T_1, T_2\right]}-Q^{\text {opt }}(a, s)=\frac{\phi_L^{\top}(a, s)}{N\left(T_2-T_1\right)} W_{\left[T_1, T_2\right]}^{-1} \sum_{i=1}^N \sum_{t=T_1}^{T_2-1} \phi_L\left(A_{i, t}, S_{i, t}\right) \delta_{i, t}^* \\
+O\left(\frac{L^{-p / d}}{(1-\gamma)^3}\right)+O\left(\frac{L \log (N T)}{(1-\gamma)^3 \in N T}\right)
\end{array}
$$
with probability at least $1-O\left(N^{-1} M^{-1}\right)$, and we recall that $\delta_{i, t}^*$ denotes the temporal difference $\operatorname{error} R_{i, t}+\gamma \max _a Q^{\text {opt }}\left(a, S_{i, t+1}\right)-Q^{\text {opt }}\left(A_{i, t}, S_{i, t}\right)$.



### Significance Justification:
* The empirical results are reasonable, showing that the proposed algorithm outperforms baselines as within-cluster correlation increases. However, a key concern is how the algorithm performs in real-world settings where the common effect within a cluster may not be as strong. For example, it would be helpful for the authors to report the actual value of the within-cluster correlation in their experiments to illustrate what constitutes a typical setting in real-world data.

### Novelty Justification:
* This work applies GEE to the estimation of Q-functions. This appears to be a relatively straightforward combination of the widely used methods GEE and FQI. While there may be additional technical challenges involved, these challenges are not clearly discussed in the paper.

Thank you for the observation regarding the combination of GEE and FQI in our work. While at first glance this may appear to be a straightforward integration, there are significant technical challenges involved in adapting GEE for Q-function estimation, particularly in the context of reinforcement learning with clustered data. We acknowledge that these challenges were not sufficiently highlighted in the original manuscript, and we appreciate the opportunity to elaborate further.

1. **Adapting GEE for Temporal Dependencies:**
   - Traditional GEE frameworks are designed for longitudinal or clustered data with independent observations across clusters but do not natively account for temporal dependencies within reinforcement learning trajectories. Adapting GEE to estimate Q-functions requires careful handling of these temporal structures, particularly in the computation of temporal difference (TD) errors and the construction of the working correlation matrix.

2. **Optimizing Basis Functions for Efficiency:**
   - One of the key technical challenges lies in identifying an optimal basis function within the GEE framework that minimizes the variance of the Q-function estimator. This involves iteratively solving the Bellman optimality equation while updating the working correlation matrix to model the intra-cluster correlations effectively. Balancing computational efficiency with statistical robustness in this iterative process is non-trivial.

3. **Robustness Under Misspecified Correlation Structures:**
   - Another challenge is ensuring the robustness of the GFQI estimator when the working correlation matrix is misspecified. While GEE provides theoretical guarantees of consistency, practical implementation requires careful modeling of the correlation structure to achieve efficiency gains, especially when intra-cluster correlations are complex or poorly understood.

4. **Extension to Reinforcement Learning:**
   - Unlike typical GEE applications in static settings, our approach extends GEE to a dynamic setting where policies interact with an evolving environment. This extension necessitates deriving new theoretical results to ensure that the proposed estimator achieves consistency and efficiency, even under the complexities of reinforcement learning tasks.

We will revise the manuscript to explicitly discuss these technical challenges and how they were addressed in our work. Highlighting these points will provide a clearer understanding of the contributions and the innovations involved in adapting GEE for Q-function estimation in reinforcement learning. Thank you for pointing out this opportunity to improve the clarity and impact of our work.

### Clarity
* The paper spends many pages explaining details of existing methods such as GEE, RL algorithms, and standard MDPs. Readers would benefit more from an explicit discussion of how the current work differs from previous work. This comparison is essential for understanding the novel aspects of the proposed method.

* In contrast, the proposed method is discussed in less detail. For example, key assumptions and theoretical results are omitted from the main paper, making it difficult for readers to grasp the foundation of the proposed method. Without details of Theorem 1, it is hard to understand why the optimal $\Phi$ has the given form on page 6.

* A description of the real data used would provide helpful context, including the number of users, the number of clusters, and the duration of the study.

* The motivation for the methodology would be stronger if the authors discussed the strength of within-cluster correlation in the real data and the generality of clustered data problems.

* Additionally, I have some questions about the notations:

    In model II on page 4, $m$ seems to represent the total number of subjects in a cluster, being an observed value of $M$ and taking values from 1 to infinity. However, in the next paragraph, $m$ is used as the index of each trajectory in a cluster, taking values from 1 to $M$. If this understanding is correct, using distinct notations for these cases would reduce ambiguity.
    In the simulation study, the number of decision times per week is unclear. It seems that there is an action for each day $t$, so there are 7 actions in a week. However, the horizon is defined as the number of weeks $T$.

## RE: Reviewer MTVu

### Soundness justification

* My biggest question focuses on the soundness of Theorem 1. In particular, one of the assumptions of Theorem 1 is that the environment is a Linear MDP. How does this accommodate the clustered structure in the data that is presented as a motivating example? It is not clear whether Theorem 1 implies the claimed benefits of GFQI. I would like additional clarification here.

* Additionally, for both Theorem 1 and Theorem 2, it would be nice to see a discussion of how a maximally misspecified correlation matrix affects the result.

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
* Figure 1 is not sufficiently motivating. What is the key takeaway here? There are no axes labels, and while there are some clustered color groups (e.g., more blue in the lower left of the grid), it is not clear that there is strong intra-cluster correlation. Please explain this more.

  
* In Figure 2, why does the delta between FQI/GFQI get smaller as the number of clusters increase? Is it because with a larger number of clusters, there is more shared structure between clusters that an algorithm can capitalize on? Is the distance between the clusters a factor here? Also is is the performance of GFQI similar to FQI if there is 1 cluster?
    - Thank you for the insightful questions regarding the trends observed in Figure 2.

    - The decreasing gap between FQI and GFQI as the number of clusters increases is primarily due to the properties of GEE. GFQI incorporates a working correlation matrix to account for intra-cluster correlations, providing efficiency gains over FQI (which corresponds to GEE with an independence working correlation matrix) when the intra-cluster correlation is relatively high compared to the sample size. However, as the number of clusters increases, the total sample size grows, and both FQI and GFQI converge to the same consistent estimates. This is because GEE’s consistency property ensures that, regardless of the working correlation matrix, the estimates converge to the true parameter values as the sample size increases. Hence, the performance gap diminishes as both methods converge.

    - This observation aligns with the theoretical results in Theorem 1, where the convergence rate for all GFQI estimators, regardless of the working correlation matrix, is of the order $O((MN)^{-1/2})$, where $N$ is the number of state-action-reward-next-state tuples in the dataset, which depends on the number of clusters. As for the distance between clusters, GEE assumes that different clusters are independent, and thus it does not directly model the distance or relationship between clusters.

    - Regarding the case of a single cluster, GFQI and FQI are likely to perform similarly in practice. While it is theoretically possible to model intra-cluster correlation within a single cluster using a working correlation matrix, the efficiency gains depend on the size of the cluster and the strength of the correlation. If the intra-cluster correlation is strong and the cluster size is relatively small, the correct working correlation matrix could provide some efficiency gains by decorrelating the data. However, in most cases, a single cluster offers limited scope for leveraging such structures, making the performance of GFQI and FQI appear similar. Additionally, as the cluster size grows, both methods converge to the same consistent estimates, further reducing the potential difference.


   
* You don’t define many terms including regret, MLE, and DGP. Please define these terms early in the paper for reader clarity.
    - done. add a notion section
    - 
* What is the convergence criteria?
    - Thank you for pointing out the need for clarification on the convergence criteria.

    - The convergence criteria in our method are as follows: the algorithm is considered to have converged when the predicted responses from two consecutive fitted models have a relative difference smaller than $10^{-5}$, or when the maximum number of iterations (100) is reached. These criteria ensure both accuracy and computational feasibility during optimization.

    - We will include this detail in the revised manuscript to provide clarity on this aspect of the method.
      

### Additional Comments:
* How do Generalized Estimating Equations work when the relationship between X and Y is not linear? I imagine there are several examples in which a linear structure is too restrictive. The same is true for the assumption on the optimal Q-function.
    - We appreciate the reviewer’s insightful observation about the potential limitations of linear assumptions in both the Generalized Estimating Equations (GEE) framework and the optimal Q-function.

    - For the concern regarding the linear relationship in GEE, we agree that linear models can sometimes be restrictive. However, GEE can naturally accommodate non-linear relationships through the use of appropriate link functions (e.g., logit, probit) and advanced modeling techniques such as basis expansions, splines, or other non-linear transformations. In our paper, we leverage basis functions to approximate the Q-function, which effectively extends the framework beyond a strictly linear structure.

    - Regarding the assumption of linear combinations for the optimal Q-function, we would like to emphasize that our approach allows for significant flexibility. By approximating the Q-function as a linear combination of basis functions, the method can model complex relationships. Importantly, this approximation can capture any non-linear structure in the Q-function if the number of basis functions is allowed to grow sufficiently large. This ensures that our method remains highly flexible and theoretically robust even in scenarios with non-linear relationships.

    - We will revise the manuscript to clarify these points and highlight the flexibility of our approach in handling non-linear structures. Thank you for bringing up this important perspective, which allows us to strengthen the clarity and positioning of our work.
