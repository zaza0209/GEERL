# rebuttal

## RE: Reviewer u8Kq

### Significance Justification:
* If I understand correctly, in the numerical study, a single agent learns a generalized policy for each baseline algorithm, which can be applied across different teams or clusters. While training one model is computationally more efficient than training separate models for each cluster, this approach might limit the agent's ability to optimize performance for specific clusters with unique characteristics, as it aims for a one-size-fits-all policy. This appears to be a significant advantage of GFQI compared to the baseline approaches. Training a separate agent for each cluster could allow the learning process to better adapt to the specific dynamics and characteristics of individual clusters, potentially improving performance. Could the authors please clarify or provide additional insights on this aspect?
 
This is an excellent comment. As you have discussed, the proposed GFQI focuses on learning a one-size-fits-all policy in our numerical study whereas the method you outlined learns cluster-specific policies. 

1. **Accommodating Heterogeneity:** To address your comment, we first argue that the proposed GFQI can be adapted to learn cluster-specific policies as well, in order to accommodate the heterogeneities across clusters. Specifically, we propose two approaches below: the first one is straightforward to implement, while the second requires further investigation, which we intend to pursue in future research.  

   - **Incorporating Cluster Characteristics in the State:**
     * The simplest approach is to augment the state with cluster-specific characteristics to learn policies that are tailored to individual clusters. In the motivating Intern Health Study, we these interns' universities or specialties as these characteristics.
     * When explicit cluster characteristics are unavailable, one-hot encoding can be utilized to denote each intern's cluster membership, enabling the application of the proposed GFQI. Notice that this approach differs from the cluster-specific policies mentioned earlier, as GFQI considers the correlation structure. Thus, GFQI is expected to learn more effective policies, as evidenced from our simulations and theories.
   - **Transfer Learning:** The second approach is to combine GFQI with existing transfer (reinforcement) learning approches to transfer knowledge from one cluster to another. This approach borrows information across clusters to improve learning while still allowing for adaptation to cluster-specific policies.

2. **Bias-Variance Trade-off:** Second, there is generally a bias variance trade-off between fitting a single common model and fitting separate models for each cluster. Learning a common policy reduces the variance of parameter estimates and is more sample efficient than learning separate policies for each cluster when all clusters are homogeneous. However, this approach can introduce potential biases in the presence of heterogeneous clusters.
 
We will revise the manuscript to clarify these points and include a discussion on the trade-offs between generalized and cluster-specific policies. Many thanks for raising this critical point.




### Non-conventional Contributions:
* The integration of generalized estimating equations (GEE), a classical statistical tool for longitudinal and clustered data, into the reinforcement learning (RL) framework is unconventional. This approach bridges the gap between statistical methods for correlated data and modern RL techniques, addressing a unique set of challenges rarely explored in RL literature.

* The paper tackles the underexplored problem of learning optimal policies in environments where data exhibits intra-cluster correlations. Such settings are common in real-world applications like healthcare and education, yet traditional RL methods typically assume independent and iid data, overlooking these dependencies.


    - Thank you for recognizing our contributions! We are largely encouraged by your assessment of our contributions! 



### Clarity
* In my opinion, the paper can be improved in terms of clarity. The introduction section is not clear. After reading the intro I still don’t understand the problem. There is no clear problem statement. Only after reading section 3, that the problem become clear. The authors can improve the clarity of the paper by discussing that traditional RL methods typically assume iid data which is not the case here.

    - Thank you for your feedback on the clarity of our paper. We appreciate your suggestions and will revise the introduction section to clearly state the problem and emphasize that traditional RL methods typically assume i.i.d. data, which is not the case in our study. This will help readers understand the problem more clearly from the outset.
      

## RE: Reviewer u1HY
### Soundness justification
* The proposed method uses Generalized Estimating Equations (GEE) to handle intra-cluster correlations. However, the problem setting involves two types of correlations: within-episode correlations and between-episode correlations within a cluster. The paper does not specify how the method addresses within-episode correlations.
    - This is an excellent point. First, we would like to clarify that by employing the MDP model for each episode, there are **no inherently within-episode correlations**. This is due to the Markov assumption under the MDP model. Should within-episode correlations exist, the rewards and next states are correlated and thus dependent upon past rewards and states, even when conditioned on the current state-action pair, leading to the violtion of the Markov assumption.

    - In our work, the proposed method focuses exclusively on addressing between-episode correlations using GEE, but still uses the MDP to model each individual trajectory. In that sense, **we did not handle within-episode correlations**. We acknowledge that this is a limitation of the current approach and an important direction for future work. We will clarify this point in the revised manuscript shall our paper be accepted. 

    - Finally, we outline two potential approaches to handle within-episode correlations: 
      * The first one is to use partially observed MDPs (POMDPs) to model the data, which is now non-Markov, due to the within-episode correlations. However, this is challenging as it remains unclear how to combine GEEs with traditional POMDP solutions.
      * The second approach is to assume the within-episode correlations exist only among rewards, but not states, as in https://academic.oup.com/jrsssb/article/86/3/623/7511800. In that case, the states continues to satisfy the Markov assumption, enabling MDP-type solutions to consistently learn the optimal policy. To extend GFQI to handle this situation, we need to combine all the estimation equations over time, and specify a large working matrix $V$ (analogous to Equation (6) in our paper). This matrix, now of dimension $(MT) \times (MT)$ rather than the original $M \times M$, must be block upper-diagonal to satisfy to the Bellman equation and cannot simply be a covariance matrix due to the aggregation of estimating equations over time. It remains unclear what is the optimal $V$ under such a structural constraint.
     
  However, these are beyond the scope of the current paper. We leave them for future research.


* Regarding Theorem 1, the operator in of equation (5) is non-smooth. Does this pose any challenges for statistical inference? Are there any assumptions needed to address the non-regularity [1, 2]?

  This is another excellent comment. In response:

  - First, it does pose challenges for statistical inference, particularly in nonregular cases where the optimal policy is not unique. This issue arises because the estimated optimal policy may not converge; instead, it might oscillate among multiple optimal policies. Such variability introduces additional uncertainties that are particularly challenging to quantify.

  - Here, similar to the work https://arxiv.org/pdf/1406.0764, we assume the optimal policy is unique to rule out non-regularity (see Assumption (A3) the supplementary material). 


* Regarding Theorem 2, first it seems that the regret is not properly defined, but only appears in the proof of Theorem 2 in the appendix. The value functions have not been defined either. In addition, the type of regret being discussed is unclear. It seems the focus is on simple regret, not cumulative regret, but this should be explicitly stated. Whichever is being addressed, it is important to connect the theoretical results to existing literature. For example, if the variance of $\beta$ is plugged in, what is the regret? How does the regret scale with key factors such as the dimension of the state space, episode length, and number of episodes?
      - If the covariance matrix of the temporal difference error is pluged in, the regret is still $-\frac{1}{2}\mathrm{tr}(\mathrm{Var}(\widehat{\theta})H)+O(N^{-3/2})$ as in Theorem 2 we do not assume the correct covariance matrix is used.
      - we can prove that the regret is
  $$
\sup_{\mathbf{A}, \mathbf{S}}|\mathbf{\Phi}_L^{\top}(\mathbf{A}, \mathbf{S}) \widehat{\beta}-Q^{*}(\mathbf{A}, \mathbf{S})|=
O\left(\frac{L^{-1/d}}{(1-\gamma)^2}\right)+O\left(\frac{L \sqrt{\log (MT)}}{(1-\gamma)^2 \sqrt{\epsilon MT} }\right)
$$
with probability at least $1-O\left(N^{-1}\right)$. Here $L$ is the number of basis functions and $d$ is the dimension of the state space.

We now begin the proof.
To upper bound $\sup _{\mathbf{A}, \mathbf{S}}\left|\mathbf{\Phi}_L^1(\mathbf{A}, \mathbf{S})\left(\beta^{(k) *}-\beta^{(k)}\right)\right|$, we define an intermediate quantity

$$
I_1 = \frac{\Sigma^{-1}}{MN} \sum_{i, t} \mathbf{\Phi}_{L} \left( A_{t}^{(i)}, S_{t}^{(i)} \right) \left\{ R_{t}^{(i)} + \gamma \max_{a^{\prime}} \mathbf{\Phi}_L^{\top} \left( a^{\prime}, S_{t+1}^{(i)} \right) \beta^{(k-1)} - \mathbf{\Phi}_L \left( A_{t}^{(i)}, S_{t}^{(i)} \right) \beta^{(k) *} \right\},
$$


It follows from (5) that

$$
\left\|\beta^{(k) *}-\beta^{(k)}-I_1\right\|_2=O\left(\frac{L(\epsilon N)^{-1} \log (N)}{(1-\gamma)^2}\right)+O\left(\frac{L^{1 / 2-p / d} \sqrt{(\epsilon N)^{-1} \log (N)}}{(1-\gamma)^2}\right),
$$

with probability at least $1-O\left(N^{-1} \right)$.
<!-- 
Meanwhile, using similar arguments to bounding the $\ell_2$ norm in the RHS of (8) in the supplementary, we obtain with probability at least $1-O\left(N^{-1} \right)$ that

$$
\left\|I_1-I_2\right\|_2=O\left(\frac{\sqrt{L(\epsilon N)^{-1} \log (N)}}{1-\gamma}\right)
$$
 -->

We obtain that

$$
\begin{aligned}
\sup _{\mathbf{A}, \mathbf{S}}\left|\mathbf{\Phi}_L^{\top}(\mathbf{A}, \mathbf{S})\left(\beta^{(k) *}-\beta^{(k)}\right)\right| & \leq \sup _{\mathbf{A}, \mathbf{S}}\left\|\mathbf{\Phi}_L(\mathbf{A}, \mathbf{S})\right\|_2\left\|\beta^{(k) *}-\beta^{(k)}-I_1\right\|_2 +\sup _{\mathbf{A}, \mathbf{S}}\left|\mathbf{\Phi}_L^{\top}(\mathbf{A}, \mathbf{S}) I_2\right|\\
&=O\left(\frac{L \sqrt{(\epsilon N)^{-1} \log (N)}}{1-\gamma}\right)+\sup _{\mathbf{A}, \mathbf{S}}\left|\mathbf{\Phi}_L^{\top}(\mathbf{A}, \mathbf{S}) I_2\right|,
\end{aligned}
$$

under the conditions on $L$ that $L$ is proportional to $N^{c_4}$ for some $0 < c_4 < 1/4$.

Finally, we can show that $\sup _{\mathbf{A}, \mathbf{S}}\left|\mathbf{\Phi}_L^{\top}(\mathbf{A}, \mathbf{S}) I_1\right|=O\left((1-\gamma)^{-1} L^{-1/ d}\right)$ by employing the bias control techniques developed by Huang (2003) (see Lemma 5.1 and Theorem A. 1 therein), based on which we can show that the basis function $\phi_L$ satisfies

$$
\left[\sup _{\mathbf{A}, \mathbf{S}}\left|h\left(A_t^{(i)}, S_t^{(i)}\right)\right|\right]^{-1} \sup _{\mathbf{A}, \mathbf{S}}\left|\mathbf{\Phi}_L^{\top}(\mathbf{A}, \mathbf{S}) \frac{\Sigma^{-1}}{N} \sum_{i,t} \mathbb{E}\mathbf{\Phi}_L\left(A_t^{(i)}, S_t^{(i)}\right) h\left(A_t^{(i)}, S_t^{(i)}\right)\right|=O(1),
$$

where the big- $O$ term on the RHS is uniform in any nonzero function $h$.

Consequently, we have

$$
\sup _{\mathbf{A}, \mathbf{S}}\left|\mathbf{\Phi}_L^{\top}(\mathbf{A}, \mathbf{S})\left(\beta^{(k) *}-\beta^{(k)}\right)\right|=O\left(\frac{L \sqrt{(\epsilon N )^{-1} \log (N)}}{1-\gamma}\right)+O\left(\frac{L^{-1 / d}}{1-\gamma}\right),
$$

and hence $\sup _{\mathbf{A}, \mathbf{S}}\left|Q^{(k) *}(\mathbf{A}, \mathbf{S})-Q^{(k)}(\mathbf{A}, \mathbf{S})\right|$ is of the same order of magnitude. It follows from the error analysis in the proof of Theorem 1 in the supplementary that

$$
\sup _{\mathbf{A}, \mathbf{S}}\left|Q^{*}(\mathbf{A}, \mathbf{S})-Q^{(k)}(\mathbf{A}, \mathbf{S})\right|=O\left(\frac{L \sqrt{(\epsilon N)^{-1} \log (N)}}{(1-\gamma)^2}\right)+O\left(\frac{L^{-1 / d}}{(1-\gamma)^2}\right)+O\left(\frac{\gamma^k}{1-\gamma}\right),
$$

with probability at least $1-O\left(N^{-1}\right)$. This establishes the rate of regret by noting that the number of FQI iterations much larger than $\log (N)$.


### Significance Justification:
* The empirical results are reasonable, showing that the proposed algorithm outperforms baselines as within-cluster correlation increases. However, a key concern is how the algorithm performs in real-world settings where the common effect within a cluster may not be as strong. For example, it would be helpful for the authors to report the actual value of the within-cluster correlation in their experiments to illustrate what constitutes a typical setting in real-world data.

### Novelty Justification:
* This work applies GEE to the estimation of Q-functions. This appears to be a relatively straightforward combination of the widely used methods GEE and FQI. While there may be additional technical challenges involved, these challenges are not clearly discussed in the paper.

Thank you for the observation regarding the combination of GEE and FQI in our work. We acknowledge that these challenges were not sufficiently highlighted in the original manuscript, and we appreciate the opportunity to elaborate further.

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


Thank you for your thoughtful feedback regarding the balance between background explanation and discussion of the novel aspects of our work.

We agree that the paper devotes significant space to explaining existing methods such as GEE, RL algorithms, and standard MDPs. While these details were included to ensure accessibility for a broad audience, we recognize the importance of focusing on the unique contributions of our work. In the revised manuscript, we will streamline the background sections by removing redundant material and instead allocate more space to explicitly discuss how the proposed Generalized Fitted Q-Iteration (GFQI) differs from Fitted Q-Iteration (FQI).

#### Key Differences Between FQI and GFQI:
1. **Handling of Intra-Cluster Correlations:**
   - FQI assumes that all data points are independent and identically distributed (i.i.d.), ignoring any dependencies between data points within the same cluster. This can lead to suboptimal policy estimates in settings where intra-cluster correlations are present.
   - GFQI, on the other hand, incorporates Generalized Estimating Equations (GEE) to explicitly model and account for intra-cluster correlations via a working correlation matrix. This adjustment improves sample efficiency and ensures more accurate Q-function estimation in clustered data scenarios.

2. **Working Correlation Matrix:**
   - FQI treats observations as independent, equivalent to using an independence working correlation matrix in GEE.
   - GFQI allows for flexible working correlation structures (e.g., exchangeable or other forms) to better capture the dependencies within clusters. This leads to more efficient parameter estimation when the correlation structure is appropriately specified.

3. **Theoretical Improvements:**
   - GFQI achieves minimal asymptotic variance in its Q-function estimation when the working correlation matrix is correctly specified, as shown in our theoretical results (Theorem 1). FQI does not provide this advantage, as it does not model intra-cluster correlations.

4. **Performance in Clustered Settings:**
   - Empirically, GFQI significantly outperforms FQI in settings with strong intra-cluster correlations, as demonstrated in our numerical studies. This highlights its practical advantage in scenarios where traditional FQI struggles due to its independence assumption.

We will incorporate these points into the revised manuscript to emphasize the novel contributions of GFQI compared to FQI. By focusing more on these differences, we aim to provide readers with a clearer understanding of how our method advances the state of the art in reinforcement learning for clustered data.

Thank you for this valuable feedback, which has helped us refine our presentation and better highlight the contributions of our work.


* In contrast, the proposed method is discussed in less detail. For example, key assumptions and theoretical results are omitted from the main paper, making it difficult for readers to grasp the foundation of the proposed method. Without details of Theorem 1, it is hard to understand why the optimal $\Phi$ has the given form on page 6.


Thank you for your helpful feedback regarding the clarity of the proposed method, and particularly for highlighting the need for more detail on the key assumptions and theoretical results.

We agree that a more thorough explanation of the assumptions and the formal theoretical results is necessary to fully convey the foundation of the proposed method. Due to the page limit, we have move those parts to the supplementary. In the revised manuscript, we will explicitly include these details to help readers better understand the theoretical framework behind our approach. Below is a summary of the key assumptions and the formal statements of Theorems 1 and 2:

### Assumptions:

1. **Realizability**  
   We assume the environment follows a linear Markov Decision Process (MDP) [Xie et al., 2023]. Both the reward function and the transition dynamics are linear in a known feature map $\phi(s,a)$, i.e.,
   \[
   \mathcal{T}(s' | a, s) = \phi(a, s)^\top \mu(s'),
   \]
   and
   \[
   \mathcal{R}(a, s) = \phi(a, s)^\top \omega.
   \]
   This assumption ensures that the Q-function and the environment dynamics can be represented by linear structures, which is central to deriving the properties of the GFQI estimator.

2. **Stability**  
    The matrix
    $$M^{-1}
\lambda_{\min}\mathbb{E} \left[\Phi^\top(\mathbf{A}, \mathbf{S})\phi(\mathbf{A}, \mathbf{S})-\gamma \Phi^\top(\mathbf{A}, \mathbf{S})\phi(\pi^{*}(\mathbf{S}^\prime),\mathbf{S}^\prime)\right]
    $$ is uniformly bounded away from zero.

3. **Uniqueness**
    $\pi^{*}$ is unique.

4. **FQI iterations**
    The maximum number of iterations $K$ in GFQI satisfies
$\log(N)\ll K = O(N^{c^\prime})$ for any $c^\prime > 0$. 

5. **Behavior policy**
     The data is generated by a Markov policy. 

6. **Value smoothness**
    Let $\pi(\beta)$ be the greedy policy derived by $\phi(a,s)^\top\beta$. 
Then the expected cumulative reward for $\pi(\beta)$ has third-order derivative w.r.t $\beta$. 


---

### Theorem 1: **Asymptotic Distribution of $\widehat{\beta}$**

**Formal Statement:**
Suppose Assumptions 1 and 2 are satisfied. The estimator $\widehat{\beta}$, computed by Algorithm 1 (Optimal FQI), has the following properties:

1. The asymptotic distribution of $\sqrt{MN}(\widehat{\beta} - \beta^{*})$ is normal:
   \[
   \mathcal{N}(\mathbf{0}, W^{-1} \Sigma W^{-1\top}),
   \]
   where:
   \[
   W(\mathbf{\Phi}) = \frac{1}{M} \mathbb{E}\left[ \mathbf{\Phi}(\mathbf{A}, \mathbf{S}) \left\{ \phi(\mathbf{A}, \mathbf{S}) - \gamma \phi(\pi^*(\mathbf{S}^\prime), \mathbf{S}^\prime) \right\} \right],
   \]
   and $\Sigma(\mathbf{\Phi}) = \frac{1}{M} \mathbb{E} \left( \mathbf{\Phi} \mathbf{V}^* \mathbf{\Phi}^\top \right)$.

2. When the correlation structure of the TD errors is correctly specified, and the estimator $\widehat{\Phi}^*(\mathbf{A}, \mathbf{S})$ converges to $\Phi^*(\mathbf{A}, \mathbf{S})$ with a rate at least $O(N^{-b} \log^{-1}(N))$ for some $b > 0$, $\widehat{\beta}$ achieves the minimal asymptotic variance:
   \[
   W(\Phi^*)^{-1}.
   \]

---

### Theorem 2: **Regret of the Estimated Optimal Policy**

**Formal Statement:**
Suppose Assumptions 1 and 3 are satisfied. The regret of the estimated optimal policy is given by:
\[
-\frac{1}{2} \mathrm{tr}(\mathrm{Var}(\widehat{\beta}) H) + O(N^{-3/2}),
\]
where $H = \left. \frac{\partial^2 \mathcal{V}(\pi(\beta))}{\partial \beta \partial \beta^\top} \right|_{\beta = \beta^*}$ and $\mathcal{V}(\pi(\beta)) = \sum_s V^{\pi(\beta)}(s) \rho(s)$, with $ \pi(\beta) $ derived by:
\[
\pi(\beta) = \arg \max_a \phi(a, s)^\top \beta.
\]

---

We will include these assumptions and the formal statements of Theorems 1 and 2 in the revised manuscript to provide readers with a clearer understanding of the theoretical foundations that underlie our approach. These results provide important insights into the asymptotic behavior and efficiency of the GFQI estimator, and we hope this additional detail will help readers better appreciate the rigor of our method.

Thank you again for your feedback, which has been invaluable in helping us improve the clarity and depth of our presentation.


* A description of the real data used would provide helpful context, including the number of users, the number of clusters, and the duration of the study.

* The motivation for the methodology would be stronger if the authors discussed the strength of within-cluster correlation in the real data and the generality of clustered data problems.

* Additionally, I have some questions about the notations:

    In model II on page 4, $m$ seems to represent the total number of subjects in a cluster, being an observed value of $M$ and taking values from 1 to infinity. However, in the next paragraph, $m$ is used as the index of each trajectory in a cluster, taking values from 1 to $M$. If this understanding is correct, using distinct notations for these cases would reduce ambiguity.
    In the simulation study, the number of decision times per week is unclear. It seems that there is an action for each day $t$, so there are 7 actions in a week. However, the horizon is defined as the number of weeks $T$.
    
    - we have corrected the notations.

## RE: Reviewer MTVu

### Soundness justification

* My biggest question focuses on the soundness of Theorem 1. In particular, one of the assumptions of Theorem 1 is that the environment is a Linear MDP. How does this accommodate the clustered structure in the data that is presented as a motivating example? It is not clear whether Theorem 1 implies the claimed benefits of GFQI. I would like additional clarification here.
Thank you for providing the details. Based on your points and the formal statement of Theorem 1, here is a draft of the rebuttal to the reviewer's feedback:

---


Thank you for your thoughtful comments and for raising the question regarding the soundness of Theorem 1 and its implications for the clustered structure in the data. We appreciate the opportunity to clarify these points.

**Regarding the Soundness of Theorem 1:**

You correctly point out that Theorem 1 assumes the environment is a Linear MDP. This assumption is crucial for establishing the asymptotic properties of the estimator $\widehat{\beta}$. However, it is important to note that the Linear MDP assumption does not preclude the presence of clustered data. In fact, the clustered structure is explicitly accounted for in the algorithm (GFQI) through the use of Generalized Estimating Equations (GEE), which allows us to model and handle the intra-cluster correlations.

If the linear MDP assumption does not hold, we can relax it to the realiziability assumption and the completeness assumption:

1. **Realizability**
$Q^*(a,s)$ and $Q^{(k),*}(a,s)$ (defined in \eqref{eq:k+1 iteration expected target Q} in Supplement) at all iteration $k$ is linearly realizable in a known feature map $\phi: \mathcal{A}\times\mathcal{S}\rightarrow \mathbb{R}^{d}$ if there exists a vector $\beta^*$ such that for all $(a,s)\in \mathcal{A}\times\mathcal{S}$, $Q^*(a,s)=\phi^\top(a,s)\beta^*$.

2. **Completeness**
 $\forall f\in \mathcal{F}_L$, $\mathcal{T}f\in \mathcal{F}_L$ where $\mathcal{F}_L$ is the space spanned by linear sieves $\Phi_L^\top(A,S)\beta$ for $\beta \in [-1,1]^{L}$ and $\mathcal{T}$ is the Bellman operator:
$$
(\mathcal{T}f)(A,S) =R(A,S) +\gamma \mathbb{E}_{S'\sim P(S'|S,A)}(\max_{a'}f(a',S')). 
$$
And in this case, the optimal Q function is not guranteed to have a linear form and some modeling error is needed to be included in analyzing the properties of the GFQI estimtor. 



**Clarification on the Benefits of GFQI:**

The efficiency claimed in Theorem 1 refers to the asymptotic variance of the estimator $\widehat{\beta}$. Specifically, when the correlation matrix is correctly specified, GFQI achieves the minimal asymptotic variance among the class of estimators computed by solving (5). This is a significant improvement over the ordinary FQI, which uses an independence correlation structure and thus does not account for the intra-cluster correlations.

To reiterate, Theorem 1 states that:

1. The asymptotic distribution of $\sqrt{MN}(\widehat{\beta} - \beta^*)$ is normal:
   \[
   \mathcal{N}(\mathbf{0}, W^{-1} \Sigma W^{-1\top}),
   \]
   where:
   \[
   W(\mathbf{\Phi}) = \frac{1}{M} \mathbb{E}\left[ \mathbf{\Phi}(\mathbf{A}, \mathbf{S}) \left\{ \phi(\mathbf{A}, \mathbf{S}) - \gamma \phi(\pi^*(\mathbf{S}^\prime), \mathbf{S}^\prime) \right\} \right],
   \]
   and $\Sigma(\mathbf{\Phi}) = \frac{1}{M} \mathbb{E} \left( \mathbf{\Phi} \mathbf{V}^* \mathbf{\Phi}^\top \right)$.

2. When the correlation structure of the TD errors is correctly specified, and the estimator $\widehat{\Phi}^*(\mathbf{A}, \mathbf{S})$ converges to $\Phi^*(\mathbf{A}, \mathbf{S})$ with a rate at least $O(N^{-b} \log^{-1}(N))$ for some $b > 0$, $\widehat{\beta}$ achieves the minimal asymptotic variance:
   \[
   W(\Phi^*)^{-1}.
   \]

**Regret Bound:**

Additionally, based on Theorem 1, we can derive the regret bound for the estimated optimal policy. The regret is given by:
\[
-\frac{1}{2} \mathrm{tr}(\mathrm{Var}(\widehat{\beta}) H) + O(N^{-3/2}),
\]
where $H = \left. \frac{\partial^2 \mathcal{V}(\pi(\beta))}{\partial \beta \partial \beta^\top} \right|_{\beta = \beta^*}$ and $\mathcal{V}(\pi(\beta)) = \sum_s V^{\pi(\beta)}(s) \rho(s)$, with $ \pi(\beta) $ derived by:
\[
\pi(\beta) = \arg \max_a \phi(a, s)^\top \beta.
\]

This regret bound further underscores the benefits of GFQI, as it shows that the estimation error of $\beta$ directly translates into the regret of the resulting policy, and GFQI's ability to achieve the minimal asymptotic variance leads to a more efficient policy learning process.





* Additionally, for both Theorem 1 and Theorem 2, it would be nice to see a discussion of how a maximally misspecified correlation matrix affects the result.


Thank you for your insightful comments and for suggesting a discussion on the impact of a maximally misspecified correlation matrix on the results of Theorem 1 and Theorem 2. We appreciate this opportunity to provide further clarification.

**Impact of Misspecified Correlation Matrix:**

To address your request, we provide the exact expressions of the asymptotic variance of the estimated parameters and the regret bound, highlighting their relationship with the working covariance matrix.

**Asymptotic Variance of the Estimated Parameters:**

The asymptotic variance of the estimated parameters $\widehat{\beta}$ is given by:
\[
W^{-1} \Sigma W^{-1\top},
\]
where:
\[
W(\mathbf{\Phi}) = \frac{1}{M} \mathbb{E}\left[ \mathbf{\Phi}(\mathbf{A}, \mathbf{S}) \left\{ \phi(\mathbf{A}, \mathbf{S}) - \gamma \phi(\pi^*(\mathbf{S}^\prime), \mathbf{S}^\prime) \right\} \right],
\]
and $\Sigma(\mathbf{\Phi}) = \frac{1}{M} \mathbb{E} \left( \mathbf{\Phi} \mathbf{V}^* \mathbf{\Phi}^\top \right)$.

Here, $\mathbf{\Phi}$ can be expressed as:
\[
\mathbf{\Phi}^*(\mathbf{S}, \mathbf{A}) = \Big[\phi^*(A^{(1)}, S^{(1)}), \cdots, \phi^*(A^{(M)}, S^{(M)})\Big] \mathbf{V}^{-1},
\]
where $\mathbf{V}$ is the working covariance matrix of the TD error. This matrix can be misspecified, meaning it may not accurately reflect the true underlying correlation structure.

**Regret Bound:**

The regret bound for the estimated optimal policy is given by:
\[
-\frac{1}{2} \mathrm{tr}(\mathrm{Var}(\widehat{\beta}) H) + O(N^{-3/2}),
\]
where $H = \left. \frac{\partial^2 \mathcal{V}(\pi(\beta))}{\partial \beta \partial \beta^\top} \right|_{\beta = \beta^*}$ and $\mathcal{V}(\pi(\beta)) = \sum_s V^{\pi(\beta)}(s) \rho(s)$, with $\pi(\beta)$ derived by:
\[
\pi(\beta) = \arg \max_a \phi(a, s)^\top \beta.
\]

The robustness properties of Theorem 1 and Theorem 2 are established for a general working covariance matrix that is not necessarily correct 


### Significance: somewhat significant (e.g., significant performance in some but not all experiments, missing a baseline)
- We appreciate your comment and would like to clarify that we have indeed included the ordinary Fitted Q Iteration (FQI) as a baseline in our experiments and compared its performance with our proposed algorithm, GFQI.
- In addition to FQI, we have also included other baseline algorithms such as Adapted GTD (AGTD), Conservative Q-Learning (CQL), Double Deep Q-Network (DDQN), and the behavior policy used in the Intern Health Study (IHS) to provide a comprehensive evaluation.


### Significance justification
* If the covariance structure is well-specified, the results in Figure 5 make sense, though I am curious why GFQI does not get asymptotically better as the number of clusters increases. Additionally, the performance of GFQI is not consistently better under some conditions, which leads me to question when GFQI is most appropriate to use. I would appreciate additional discussion of this.
    - We appreciate the reviewer’s insightful observation regarding the asymptotic behavior of GFQI as the number of clusters increases. Upon closer examination, we confirm that GFQI indeed exhibits improved performance with an increasing number of clusters. However, the results in Figure 5 might appear to suggest otherwise due to the limited range of clusters shown in the current plot. As the number of clusters continues to grow beyond the range displayed in Figure 5, the regret of GFQI continues to decrease, supporting its asymptotic improvement.
    - Additionally, regarding the observation that GFQI’s performance is not consistently better under all conditions, we note that the advantage of GFQI depends heavily on the intra-cluster correlation (𝜓) and the degree of misspecification of the covariance structure. GFQI provides substantial benefits when the intra-cluster correlation is strong or when correctly specified correlation structures allow for more efficient policy learning. In cases of weaker correlation or greater misspecification, the relative advantage of GFQI diminishes, which aligns with our theoretical findings.
    - In summary, the significance of the advantage of the GFQI verses other baselines depends on the sample size, the strongness of the intra-cluster correlation, and the correctly specified correlation structure.

### Novelty Justification:
* I believe that using GEE to improve the sample efficiency of FQI is a novel methodological approach. However, I am unclear of the performance of GFQI (the proposed approach), when the correlation structure is misspecified.
    - We appreciate the reviewer’s acknowledgment of the novelty of using GEE to improve the sample efficiency of FQI. Regarding the performance of GFQI under misspecified correlation structures, we acknowledge that this is a critical aspect to consider.

    - To address this concern, we have added a figure...

    - Our preliminary results show that GFQI retains its robustness under misspecification, aligning with the theoretical guarantee of asymptotic normality. However, as expected, its efficiency (e.g., variance reduction and regret minimization) is reduced compared to the case of a well-specified structure. We believe this additional analysis will provide greater clarity on the practical applicability and limitations of GFQI when the correlation structure is misspecified.

    - Thank you for highlighting this important consideration. We will ensure that the revised manuscript reflects this additional evaluation.

### Clarity
* Figure 1 is not sufficiently motivating. What is the key takeaway here? There are no axes labels, and while there are some clustered color groups (e.g., more blue in the lower left of the grid), it is not clear that there is strong intra-cluster correlation. Please explain this more.

  
* In Figure 2, why does the delta between FQI/GFQI get smaller as the number of clusters increase? Is it because with a larger number of clusters, there is more shared structure between clusters that an algorithm can capitalize on? Is the distance between the clusters a factor here? Also is is the performance of GFQI similar to FQI if there is 1 cluster?
    - Thank you for the insightful questions regarding the trends observed in Figure 2.

    - The decreasing gap between FQI and GFQI as the number of clusters increases is primarily due to the properties of GEE. GFQI incorporates a working correlation matrix to account for intra-cluster correlations, providing efficiency gains over FQI (which corresponds to GEE with an independence working correlation matrix) when the intra-cluster correlation is relatively high compared to the sample size. However, as the number of clusters increases, the total sample size grows, and both FQI and GFQI converge to the same consistent estimates. This is because GEE’s consistency property ensures that, regardless of the working correlation matrix, the estimates converge to the true parameter values as the sample size increases. Hence, the performance gap diminishes as both methods converge.

    - This observation aligns with the theoretical results in Theorem 1, where the convergence rate for all GFQI estimators, regardless of the working correlation matrix, is of the order $O((MN)^{-1/2})$, where $N$ is the number of state-action-reward-next-state tuples in the dataset, which depends on the number of clusters. As for the distance between clusters, GEE assumes that different clusters are independent, and thus it does not directly model the distance or relationship between clusters.

    - Regarding the case of a single cluster, GFQI can still gain better performance than FQI. The efficiency gains depend on the size of the cluster and the strength of the correlation. If the intra-cluster correlation is strong and the cluster size is relatively small, the correct working correlation matrix could provide some efficiency gains by decorrelating the data. As the cluster size grows, both methods converge to the same consistent estimates, further reducing the potential difference.
    
   


* It would be nice to clarify that Figures 3 and 4 define causal relationships (which are referred to as paths) between components of the MDP. It is still not clear to me whether a correlation (rather than a causation) also violates the independence assumption.

Thank you for your feedback and for pointing out the need for clarification regarding the causal relationships and correlations in Figures 3 and 4. We appreciate the opportunity to address this point.

**Clarification on Causal Relationships and Correlations**

In Figures 3 and 4, we indeed define causal relationships (referred to as paths) between components of the Markov Decision Process (MDP). These paths illustrate the dependencies and transitions within the MDP framework.

**Independence Assumption**

The independence assumption in the standard MDP requires that the state-action-reward triplets be independent across different trajectories. This means that there should be no directed paths connecting these triplets across different trajectories.

**Impact of Correlation**

Correlation can indeed violate the independence assumption. If there is a correlation between the state-action-reward triplets across different trajectories, it implies a dependency that is not accounted for in the standard MDP framework. This correlation can arise due to various factors, such as shared environmental conditions, social interactions, or other common influences within clusters.



* You don’t define many terms including regret, MLE, and DGP. Please define these terms early in the paper for reader clarity.
    - Thank you for your feedback and for pointing out the need to define key terms and we will add a notation section at the beginning of the paper to introduce all the important notations.
      
* What is the convergence criteria?
    - Thank you for pointing out the need for clarification on the convergence criteria.

    - The convergence criteria in our method are as follows: the algorithm is considered to have converged when the predicted responses from two consecutive fitted models have a relative difference smaller than $10^{-5}$, or when the maximum number of iterations (100) is reached. These criteria ensure both accuracy and computational feasibility during optimization.

    - We will include this detail in the revised manuscript to provide clarity on this aspect of the method.

* Why was a uniform behavior policy used? This seems like an easier setting and a little unrealistic. 
    - Thank you for your feedback and for questioning the choice of a uniform behavior policy.
    - A uniform behavior policy ensures that the data is generated from a balanced distribution of actions, which can lead to more accurate estimation of the Q-function and optimal policy. This is particularly important in offline RL settings where the behavior policy is fixed and cannot be altered.
      

### Additional Comments:
* How do Generalized Estimating Equations work when the relationship between X and Y is not linear? I imagine there are several examples in which a linear structure is too restrictive. The same is true for the assumption on the optimal Q-function.
    - We appreciate the reviewer’s insightful observation about the potential limitations of linear assumptions in both the Generalized Estimating Equations (GEE) framework and the optimal Q-function.

    - For the concern regarding the linear relationship in GEE, we agree that linear models can sometimes be restrictive. However, GEE can naturally accommodate non-linear relationships through the use of appropriate link functions (e.g., logit, probit) and advanced modeling techniques such as basis expansions, splines, or other non-linear transformations. In our paper, we leverage basis functions to approximate the Q-function, which effectively extends the framework beyond a strictly linear structure.

    - Regarding the assumption of linear combinations for the optimal Q-function, we would like to emphasize that our approach allows for significant flexibility. By approximating the Q-function as a linear combination of basis functions, the method can model complex relationships. Importantly, this approximation can capture any non-linear structure in the Q-function if the number of basis functions is allowed to grow sufficiently large. This ensures that our method remains highly flexible and theoretically robust even in scenarios with non-linear relationships. Besides, we can try other link functions than the linear function such as modeling the Q function as $f(s,a,\beta)$. The corresponding properties of the Q function can be similarly derived under GEE's framework.

    - We will revise the manuscript to clarify these points and highlight the flexibility of our approach in handling non-linear structures. Thank you for bringing up this important perspective, which allows us to strengthen the clarity and positioning of our work.
