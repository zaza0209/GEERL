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
* The integration of GEE, a classical statistical tool for longitudinal and clustered data, into the RL framework is unconventional. This approach bridges the gap between statistical methods for correlated data and modern RL techniques, addressing a unique set of challenges rarely explored in RL literature.

* The paper tackles the underexplored problem of learning optimal policies in environments where data exhibits intra-cluster correlations. Such settings are common in real-world applications like healthcare and education, yet traditional RL methods typically assume independent and iid data, overlooking these dependencies.


    - Thank you for recognizing our contributions! We are largely encouraged by your assessment of our contributions! 



### Clarity
* In my opinion, the paper can be improved in terms of clarity. The introduction section is not clear. After reading the intro I still don’t understand the problem. There is no clear problem statement. Only after reading section 3, that the problem become clear. The authors can improve the clarity of the paper by discussing that traditional RL methods typically assume iid data which is not the case here.

    - Thank you for your feedback on the clarity of our paper. 
    We mentioned that the existing RL algorithm relies on i.i.d. assumption in the first paragraph of the Challenge subsection of the introduction. We appreciate your suggestions and will revise the introduction section to clearly state the problem and emphasize that traditional RL methods typically assume i.i.d. data, which is not the case in our study. This will help readers understand the problem more clearly from the outset.
      

## RE: Reviewer u1HY
### Soundness justification
* The proposed method uses Generalized Estimating Equations (GEE) to handle intra-cluster correlations. However, the problem setting involves two types of correlations: within-episode correlations and between-episode correlations within a cluster. The paper does not specify how the method addresses within-episode correlations.
    - This is an excellent point. First, we would like to clarify that by employing the MDP model for each episode, there are **no inherently within-episode correlations**. This is due to the Markov assumption under the MDP model. Should within-episode correlations exist, the rewards and next states are correlated and thus dependent upon past rewards and states, even when conditioned on the current state-action pair, leading to the violtion of the Markov assumption.

    - In our work, the proposed method focuses exclusively on addressing between-episode correlations using GEE, but still uses the MDP to model each individual trajectory. In that sense, **we did not handle within-episode correlations**. We acknowledge that this is a limitation of the current approach and an important direction for future work. We will clarify this point in the revised manuscript shall our paper be accepted. 

    - Having said that, we **outline two potential approaches to handle within-episode correlations** and discuss their challenges: 
      * The first one is to use partially observed MDPs (POMDPs) to model the data, which is now non-Markov, due to the within-episode correlations. However, this is challenging as it remains unclear how to combine GEEs with traditional POMDP solutions.
      * The second approach is to assume the within-episode correlations exist only among rewards, but not states, as in [0]. In that case, the states continues to satisfy the Markov assumption, enabling MDP-type solutions to consistently learn the optimal policy. To extend GFQI to handle this situation, we need to combine all the estimation equations over time, and specify a large working matrix $V$ (analogous to Equation (6) in our paper). This matrix, now of dimension $(MT) \times (MT)$ rather than the original $M \times M$, must be block upper-diagonal to satisfy to the Bellman equation and cannot simply be a covariance matrix due to the aggregation of estimating equations over time. It remains unclear what is the optimal $V$ under such a structural constraint.
     
       These challenges are non-trivial, and addressing them is beyond the scope of this paper. Should this paper be accepted, we will include these related discussions. 


* Regarding Theorem 1, the operator in of equation (5) is non-smooth. Does this pose any challenges for statistical inference? Are there any assumptions needed to address the non-regularity [1, 2]?

  This is another excellent comment. In response:

  - First, it does pose challenges for statistical inference, particularly in nonregular cases where the optimal policy is not unique, also referred to as non-exceptional laws (see e.g., [1]). This issue arises because the estimated optimal policy may not converge; instead, it might oscillate among multiple optimal policies. Such variability introduces additional uncertainties that are particularly challenging to quantify.

  - In addition to posing challenges for statistical inference, this issue also complicates the asymptotic behavior of the estimator. Specifically, the estimator may not be asymptotically normal ([1], [2], [3]). 

  - In this paper, we did not consider statistical inference. So, the policy learning procedure works regardless of the presence of non-regularities. However, for theoretical purposes, we assume a regular setting to establish the asymptotic normality of the estimator. In particular, similar to the work [4], we assume the optimal policy is unique to rule out non-regularity (see Assumption (A3) in the supplementary material).

  - Alternative to assuming a regular setting, several approaches are available for statistical inference in non-regular settings, including the simple sample-splitting approach, the double-boostrap approach by [5], the penalization-based method developed by [6], the hard-thresholding approach in [1], the importance sampling method in [2], the one-step method developed in [1] and the subagging approach developed in [7]. We will discuss this in detail shall our paper be accepted. 

[0] Luo, S., Yang, Y., Shi, C., Yao, F., Ye, J., & Zhu, H. (2024). Policy evaluation for temporal and/or spatial dependent experiments. Journal of the Royal Statistical Society Series B: Statistical Methodology, qkad136. https://academic.oup.com/jrsssb/article-abstract/86/3/623/7511800

[1] Luedtke, Alexander R., and Mark J. Van Der Laan. "Statistical inference for the mean outcome under a possibly non-unique optimal treatment strategy." Annals of statistics 44.2 (2016): 713. https://arxiv.org/abs/1603.07573

[2] Zhu, Wensheng, Donglin Zeng, and Rui Song. "Proper inference for value function in high-dimensional Q-learning for dynamic treatment regimes." Journal of the American Statistical Association 114.527 (2019): 1404-1417. https://pmc.ncbi.nlm.nih.gov/articles/PMC6953729/pdf/nihms-987499.pdf

[3] Shi, Chengchun, et al. "Statistical inference of the value function for RL in infinite-horizon settings." Journal of the Royal Statistical Society Series B: Statistical Methodology 84.3 (2022): 765-793. https://arxiv.org/pdf/2001.04515

[4] Ertefaie, Ashkan. "Constructing dynamic treatment regimes in infinite-horizon settings." arXiv preprint arXiv:1406.0764 (2014). https://arxiv.org/pdf/1406.0764

[5] Chakraborty, Bibhas, Eric B. Laber, and Ying-Qi Zhao. "Inference about the expected performance of a data-driven dynamic treatment regime." Clinical Trials 11.4 (2014): 408-417. https://pmc.ncbi.nlm.nih.gov/articles/PMC4265005/

[6] Song, Rui, et al. "Penalized q-learning for dynamic treatment regimens." Statistica Sinica 25.3 (2015): 901.https://arxiv.org/pdf/1108.5338

[7] Shi, Chengchun, Wenbin Lu, and Rui Song. "Breaking the Curse of Nonregularity with Subagging---Inference of the Mean Outcome under Optimal Treatment Regimes." Journal of Machine Learning Research 21.176 (2020): 1-67. https://jmlr.org/papers/volume21/20-066/20-066.pdf


* Regarding Theorem 2, first it seems that the regret is not properly defined, but only appears in the proof of Theorem 2 in the appendix. The value functions have not been defined either. In addition, the type of regret being discussed is unclear. It seems the focus is on simple regret, not cumulative regret, but this should be explicitly stated. Whichever is being addressed, it is important to connect the theoretical results to existing literature. For example, if the variance of $\beta$ is plugged in, what is the regret? How does the regret scale with key factors such as the dimension of the state space, episode length, and number of episodes?



    - First, we would like to clarify that the regret discussed in our paper is not the cumulative regret commonly used in online settings. Instead, our focus is on the offline setting and the regret is defined as the sub-optimality gap, i.e., the expected $\gamma$-discounted cumulative reward under the learned policy compared to the optimal policy. This type of regret is commonly employed in offline RL literature. We initially omitted these definitions in the main text to make the paper less technical to improve the flow, but we are more than happy to use the extra page to formally define the regret & value function in the main text to avoid confusion，shall paper be accepted.
 
    
    - If the covariance matrix of the temporal difference error is pluged in, the regret is still $-\frac{1}{2}\mathrm{tr}(\mathrm{Var}(\widehat{\theta})H)+O(N^{-3/2})$ as in Theorem 2 we do not assume the correct covariance matrix is used. As $N$ is the total number of data tuples, the episode length and number of episodes affect the regret through $N$.


### Significance Justification:
* The empirical results are reasonable, showing that the proposed algorithm outperforms baselines as within-cluster correlation increases. However, a key concern is how the algorithm performs in real-world settings where the common effect within a cluster may not be as strong. For example, it would be helpful for the authors to report the actual value of the within-cluster correlation in their experiments to illustrate what constitutes a typical setting in real-world data.

Thank you for the suggestions. Our real-data analysis revealed modest within-cluster correlations: $0.05$ for step count, $0.02$ for sleep duration, and $0.05$ for mood score.

To clarify the relationship between our simulation and real-world settings: The baseline scenario where $\psi = 1$ corresponds to these observed real-world correlations. When we increase $\psi$, the correlation scales proportionally - for instance, $\psi = 3$ yields correlations around 0.15. From Figure 1, we can observe that GFQI performs better than FQI and other methods in almost all settings when correlation is around $0.15$, indicating the performance improvement of GFQI algorithm by taking account of the correlation structure. 
	
Moreover, we identified another interventional mobile health study that exhibits higher within-cluster correlations of approximately $0.3$ [1]. We will incorporate analysis of this additional dataset in the final manuscript to strengthen our empirical validation.

[1] Wang, Jitao, et al. "Effectiveness of gamified team competition as mHealth intervention for medical interns: a cluster micro-randomized trial." _NPJ Digital Medicine_ 6.1 (2023): 4. 

### Novelty Justification:
* This work applies GEE to the estimation of Q-functions. This appears to be a relatively straightforward combination of the widely used methods GEE and FQI. While there may be additional technical challenges involved, these challenges are not clearly discussed in the paper.
 

- First, as highlighted by Reviewer u8Kq, the proposed methodology is novel in the integration of GEE -- a classical statistical tool for longitudinal and clustered data -- to handle intra-cluster correlations in RL.

- Second, we would like to clarify the technical challenges. The major challenge lies in identifying an optimal basis function that minimizes the mean squared error or variance of the Q-function estimator. This task is particularly challenging because:

  * In RL, most works derive upper error bounds on the Q-function estimator, with less focus on its (asymptotic) variance -- not to mention how to identify the basis function that minimizes the variance. There a few expections that study the asymptotic variance, see e.g., the GTD [paper](https://arxiv.org/abs/2001.04515). However, they only considered policy evaluation where the Q-function is estimated from a single set of estimating equation. In contrast, we consider policy learning, which involves iteratively solving a sequence of GEEs. Hence, the error accumulated by the iterative process should be carefully taken into consideration.

  * Similarly, in the GEE literature, the problem is simpler as the GEE is typically solved once. It does not account for the error accumulation over multiple iterations.

  We are more than happy to include these discussions to the paper shall it be accepted. 


<!-- - Our work focuses on the under-researched issue of learning optimal policies in environments where data exhibits intra-cluster correlations. Such scenarios are prevalent in real-world applications such as healthcare and education, where traditional RL methods often assume independent and identically distributed (i.i.d.) data, thereby ignoring these dependencies. Our approach uniquely addresses this gap by employing GEE to model and manage these correlations.

- The integration of GEE into RL involves significant technical complexities, including adapting GEE to accommodate the sequential nature of RL data, accurately estimating Q-functions in the presence of correlated data, and optimizing policies under these conditions. These challenges necessitate novel adaptations and modifications to the standard GEE and FQI frameworks.-->

 

### Clarity
* The paper spends many pages explaining details of existing methods such as GEE, RL algorithms, and standard MDPs. Readers would benefit more from an explicit discussion of how the current work differs from previous work. This comparison is essential for understanding the novel aspects of the proposed method.


Thank you for pointing this out. We agree that our paper devotes a lot of space to explaining existing methods such as GEE, RL algorithms, and standard MDPs. While these details were included to ensure accessibility for a broad audience, we recognize the importance of focusing on the unique contributions of our work. Shall our paper be accepted, we plan to streamline the background sections by removing redundant materials and use the extra page to explicitly discuss how the proposed GFQI differs from FQI. We summarize the key differences below and will include the related discussions as mentioned. 


#### Key Differences Between FQI, AGTD and GFQI:
We will highlight the differences between GFQI and two closely related methods: FQI and GTD. Notice that these methods are detailed in Section 4 as well.

1. **Methodological differences:**

    - FQI assumes that all observations are i.i.d., which ignores intra-cluster correlations commonly present in clustered data. In contrast, GFQI incorporates GEE to explicitly model and account for these correlations via a working correlation matrix. This enables GFQI to achieve more efficient Q-function estimation and reduced regret in settings with clustered data.

    - GTD is designed for policy evaluation, aiming to estimate the value function for a fixed policy. The proposed GFQI focuses on the more complex problem of policy learning, where policy evaluation often serves as an intermediate step (Sutton and Barto, 2018). Even if GTD can be extended to policy learning (as discussed in Section 4, Algorithm 2), it still fundamentally differs from GFQI because it does not handle intra-cluster correlations.
 

3. **Theoretical differences:**
   
    GFQI achieves minimal asymptotic variance and regret in its Q-function estimation when the working correlation matrix is correctly specified, as shown in our theoretical results (Theorems 1 and 2). Consequently, GFQI achieves smaller asymptotic variance and regret than FQI and AGTD.

5. **Empirical differences:**
   
    Empirically, GFQI significantly outperforms FQI and AGTD in settings with strong intra-cluster correlations or small sample size, as demonstrated in our numerical studies. 


* In contrast, the proposed method is discussed in less detail. For example, key assumptions and theoretical results are omitted from the main paper, making it difficult for readers to grasp the foundation of the proposed method. Without details of Theorem 1, it is hard to understand why the optimal $\Phi$ has the given form on page 6.


Thank you for your helpful feedback regarding the clarity of the proposed method, and particularly for highlighting the need for more detail on the key assumptions and theoretical results.

Due to the page limit, we decided to move tho assumptions and formal theoretical statements to the Supplementary Materials. Shall our paper be accepted, we are more than happy to use the extra page to include these details in the main paper. Below is a summary of the key assumptions and the formal statements of Theorems 1 and 2 which we plan to include: 

### Assumptions:

1. **Realizability**  
   We assume the environment follows a linear Markov Decision Process (MDP) [Xie et al., 2023]. Both the reward function and the transition dynamics are linear in a known feature map $\phi(s,a)$, i.e.,
   
   $$
   \mathcal{T}(s' | a, s) = \phi(a, s)^\top \mu(s'),
   $$

   and
   
   $$
   \mathcal{R}(a, s) = \phi(a, s)^\top \omega.
   $$

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

1. The asymptotic distribution of $\sqrt{MN}(\widehat{\beta} - \beta^{\*})$ is normal:
   
   $$
   \mathcal{N}(\mathbf{0}, W^{-1} \Sigma W^{-1\top}),
   $$

   where:
   
     $$
   W(\mathbf{\Phi}) = \frac{1}{M} \mathbb{E}\left\[ \mathbf{\Phi}(\mathbf{A}, \mathbf{S}) \left\\{ \phi(\mathbf{A}, \mathbf{S}) - \gamma \phi(\pi^*(\mathbf{S}^\prime), \mathbf{S}^\prime) \right\\} \right\],
   $$

   and $\Sigma(\mathbf{\Phi}) = \frac{1}{M} \mathbb{E} \left( \mathbf{\Phi} \mathbf{V}^* \mathbf{\Phi}^\top \right)$.

2. When the correlation structure of the TD errors is correctly specified, and the estimator $\widehat{\Phi}^*(\mathbf{A}, \mathbf{S})$ converges to $\Phi^*(\mathbf{A}, \mathbf{S})$ with a rate at least $O(N^{-b} \log^{-1}(N))$ for some $b > 0$, $\widehat{\beta}$ achieves the minimal asymptotic variance:
   \[
   W(\Phi^*)^{-1}.
   \]

---

### Theorem 2: **Regret of the Estimated Optimal Policy**

**Formal Statement:**
Suppose Assumptions 1 and 3 are satisfied. The regret of the estimated optimal policy is given by:

$$
-\frac{1}{2} \mathrm{tr}(\mathrm{Var}(\widehat{\beta}) H) + O(N^{-3/2}),
$$

where $H = \left. \frac{\partial^2 \mathcal{V}(\pi(\beta))}{\partial \beta \partial \beta^\top} \right|_{\beta = \beta^*}$ and $\mathcal{V}(\pi(\beta)) = \sum_s V^{\pi(\beta)}(s) \rho(s)$, with $ \pi(\beta) $ derived by:

$$
\pi(\beta) = \arg \max_a \phi(a, s)^\top \beta.
$$

---

We hope these details can address your concerns. Thank you again for your feedback, which has been invaluable in helping us improve the clarity and depth of our presentation.


* A description of the real data used would provide helpful context, including the number of users, the number of clusters, and the duration of the study.
    - A total of $1565$ interns from $258$ institutions were included in the Intern Health Study. The study lasted for $26$ weeks. For each week, each intern had equal probability to be assigned to mood, activity, sleep or no notification week. Given a week when an intern was randomized to receiving notifications, every day they were further randomized (with $50\%$ probability) to receive a notification on that day.

* The motivation for the methodology would be stronger if the authors discussed the strength of within-cluster correlation in the real data and the generality of clustered data problems.
    - Thank you for the suggestions. Our real-data analysis revealed modest within-cluster correlations: $0.05$ for step count, $0.02$ for sleep duration, and $0.05$ for mood score. 
  
    - Clustered data structures are particularly prevalent in personalized healthcare settings where data collection naturally follows clustering structure. For instance, patient data is often collected across multiple hospitals, where individuals within the same hospital often share characteristics such as race, ethnicity, and socioeconomic status, leading to intra-hospital correlations. Our motivated real-world study of medical interns across various institutions further illustrates this phenomenon. Interns within the same institution typically demonstrate correlated outcomes due to shared experiences, including common curricula, training protocols, social interactions among peers, and geographic influences. The clustering issue is also common when collecting data from social network, where user behavior and interactions naturally form interconnected groups. Social media platforms, for example, exhibit clustering based on friend networks, shared interests, and geographic proximity.

* Additionally, I have some questions about the notations:

    In model II on page 4, $m$ seems to represent the total number of subjects in a cluster, being an observed value of $M$ and taking values from 1 to infinity. However, in the next paragraph, $m$ is used as the index of each trajectory in a cluster, taking values from 1 to $M$. If this understanding is correct, using distinct notations for these cases would reduce ambiguity.
    In the simulation study, the number of decision times per week is unclear. It seems that there is an action for each day $t$, so there are 7 actions in a week. However, the horizon is defined as the number of weeks $T$.

    - Thank you for pointing out the ambiguity regarding the notation in Model II on page 4 and the number of decision times in the simulation study. We have changed the unit of horizon to days in both figures and text.

    - You are correct in your understanding that in Model II, $ m $ is used in two different contexts: initially, it represents the total number of subjects in a cluster, which is an observed value of $M $, and then it is used as the index of each trajectory within a cluster, taking values from 1 to $M$. To avoid this confusion, we will revise the manuscript to use distinct notations for these two cases. Specifically, we will use $M$ to denote the total number of subjects in a cluster, and $m$ will continue to represent the index of each trajectory within the cluster.

    - Regarding the number of decision times per week, we acknowledge the confusion. You are correct that there is one action per day, so the number of actions in a week is 7. The horizon $T $ in the simulation is defined as the number of weeks, and we will clarify this point in the revised manuscript to ensure consistency and avoid any ambiguity.

    - Thank you again for your attention to these details. We will make the necessary revisions to improve the clarity of the manuscript. 

## RE: Reviewer MTVu

### Soundness justification

* My biggest question focuses on the soundness of Theorem 1. In particular, one of the assumptions of Theorem 1 is that the environment is a Linear MDP. How does this accommodate the clustered structure in the data that is presented as a motivating example? It is not clear whether Theorem 1 implies the claimed benefits of GFQI. I would like additional clarification here.

---


Thank you for your thoughtful comments. We appreciate the opportunity to clarify these points. In summary: 

- The linear MDP assumption is to simplify the procedure and the theoretical analysis. Meanwhile, the proposed GFQI can be easily extended to accommodate nonlinear environments. 
- The use of linear models results in interpretable Q-function estimators and decision rules, which are particularly valuable in biomedical studies (including our motivating example) where interpretability is important.
- Theorem 1 proves the efficiency of the estimated parameters in the Q-function estimator. In particular, it implies that the estimator computed by the proposed GFQI is more efficient compared to those by FQI or GTD-type policy learning algorithms. Such an efficiency in the parameter estimates further translates into a smaller regret, as demonstrated in Theorem 2.

We next elaborate the first and the last points in more detail below. 

**GFQI under nonlinear environments**: It is important to note that neither GEE nor FQI requires a linearity assumption. As such, linearity is not a key assumption to the proposed GFQI and it can be naturally extended to handle nonlinear environments. Specfically, consider a nonlinear function class $F(s,a;\beta)$ indexed by parameter $\beta$. We assume the following realizability and Bellman completeness assumptions: 

- **Realizability**: the optimal Q-function belongs to the nonlinear function class $F$.
- **Completeness**: For any $f\in F$, $\mathcal{T} f\in F$ as well where $\mathcal{T}$ denotes the Bellman operator.

Notice that the realizability and completeness assumptions are widely imposed in the RL literature. They can be further relaxed to allow approximation error to exist. Under the two assumptions, we use $F$ to parameterize the Q-function estimator at each FQI iteration. Equation (5) in our paper (with the linear function class replaced with $F$) remains valid to estimate the parameter $\beta^*$ involved in the optimal Q-function. Meanwhile, similar to Theorem 1, it can be shown that the optimal basis function $\mathbf{\Phi}^{\*}$ equals  

$$\mathbf{\Phi}^{\*}(\mathbf{A}, \mathbf{S}) = \Big[\phi^{\*}(A^{(1)}, S^{(1)}), \cdots, \phi^{\*}(A^{(M)}, S^{(M)})\Big]\mathbf{V}^{-1}$$,

where $\mathbf{V}$ is the covariance matrix of the cluster-wise TD error and $\phi^{\*}(a,s)=\frac{\partial f^{\*}(a,s;\beta)}{\partial \beta^{\*}}-\gamma \mathbb{E} \Big[\frac{\partial f^{\*}(\pi^{\*}(S'),S';\beta)}{\partial \beta^{\*}}|A=a,S=s\Big]$.  

**Clarification on the Benefits of GFQI:** The efficiency claimed in Theorem 1 refers to the asymptotic variance of the estimator $\widehat{\beta}$. Specifically, when the correlation matrix is correctly specified, GFQI achieves the minimal asymptotic variance among the class of estimators computed by solving (5). This is a significant improvement over the ordinary FQI, which uses an independence correlation structure and thus does not account for the intra-cluster correlations.

To reiterate, Theorem 1 states that:

1. The asymptotic distribution of $\sqrt{MN}(\widehat{\beta} - \beta^*)$ is normal:
   
   $$
   \mathcal{N}(\mathbf{0}, W^{-1} \Sigma W^{-1\top}),
   $$

   where:
   
   $$
   W(\mathbf{\Phi}) = \frac{1}{M} \mathbb{E}\left\[ \mathbf{\Phi}(\mathbf{A}, \mathbf{S}) \left\\{ \phi(\mathbf{A}, \mathbf{S}) - \gamma \phi(\pi^*(\mathbf{S}^\prime), \mathbf{S}^\prime) \right\\} \right\],
   $$

   and $\Sigma(\mathbf{\Phi}) = \frac{1}{M} \mathbb{E} \left( \mathbf{\Phi} \mathbf{V}^* \mathbf{\Phi}^\top \right)$.

2. When the correlation structure of the TD errors is correctly specified, and the estimator $\widehat{\Phi}^*(\mathbf{A}, \mathbf{S})$ converges to $\Phi^*(\mathbf{A}, \mathbf{S})$ with a rate at least $O(N^{-b} \log^{-1}(N))$ for some $b > 0$, $\widehat{\beta}$ achieves the minimal asymptotic variance:
   
   $$
   W(\Phi^*)^{-1}.
   $$

**Regret Bound:**

Additionally, based on Theorem 1, we can derive the regret bound for the estimated optimal policy. The regret is given by:

$$
-\frac{1}{2} \mathrm{tr}(\mathrm{Var}(\widehat{\beta}) H) + O(N^{-3/2}),
$$

where $H = \left. \frac{\partial^2 \mathcal{V}(\pi(\beta))}{\partial \beta \partial \beta^\top} \right|_{\beta = \beta^*}$ and $\mathcal{V}(\pi(\beta)) = \sum_s V^{\pi(\beta)}(s) \rho(s)$, with $ \pi(\beta) $ derived by:

$$
\pi(\beta) = \arg \max_a \phi(a, s)^\top \beta.
$$

This regret bound further underscores the benefits of GFQI, as it shows that the estimation error of $\beta$ directly translates into the regret of the resulting policy, and GFQI's ability to achieve the minimal asymptotic variance leads to a more efficient policy learning process.





* Additionally, for both Theorem 1 and Theorem 2, it would be nice to see a discussion of how a maximally misspecified correlation matrix affects the result.


    - Thank you for your insightful comments and for suggesting a discussion on the impact of a maximally misspecified correlation matrix on the results of Theorem 1 and Theorem 2. We appreciate this opportunity to provide further clarification.
    - For estimator with correct correlation matrix, the asymptotic varianceis $W^{\* -1}$ where $W^*=  \frac{1}{M} \mathbb{E}\left\[ \mathbf{\Phi}^{\*}(\mathbf{A}, \mathbf{S}) \left\\{ \phi(\mathbf{A}, \mathbf{S}) - \gamma \phi(\pi^*(\mathbf{S}^\prime), \mathbf{S}^\prime) \right\\} \right\]$. For any estimator obtained with misspecified correlation, the asymptotic variance is $W^{-1} \Sigma W^{-1\top}$ where $W$ depends on the $\mathbf{\Phi}$ which contains the misspecified correlation matrix. The difference between the two asymptotic variance would be $W^{\* -1} - W^{-1} \Sigma W^{-1\top}$. Similarly, the difference between the regret for correct and misspecified correlation structures would be proportional to $\mathrm{tr}(W^{\* -1} - W^{-1} \Sigma W^{-1\top})$ according to Theorem 2.

<!--
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
-->

### Significance: somewhat significant (e.g., significant performance in some but not all experiments, missing a baseline)
- We appreciate your comment and would like to clarify that we indeed included the following baselines in the experiments to provide a comprehensive evaluation:
    - the ordinary Fitted Q Iteration.
    - Adapted GTD (AGTD), 
    - Conservative Q-Learning (CQL), 
    - Double Deep Q-Network (DDQN), and 
    - the behavior policy used in the Intern Health Study (IHS)
    


### Significance justification
* If the covariance structure is well-specified, the results in Figure 5 make sense, though I am curious why GFQI does not get asymptotically better as the number of clusters increases. Additionally, the performance of GFQI is not consistently better under some conditions, which leads me to question when GFQI is most appropriate to use. I would appreciate additional discussion of this.


    Thank you for the detailed feedback. In response: 

    - First, we would like to clarify that GFQI **does** get asymptotically better in general as the number of clusters increases. Referring to the first row of Figure 5, the overall trend indicates that the regret decreases as the number of clusters increases. However, we acknowledge that this trend is not perfectly monotonic in some cases. For example, in the first panel, the regret slightly increases when the number of clusters changes from 20 to 25. This fluctuation may be due to randomness introduced by the random seed or an insufficient number of simulation repetitions. To address this, we conducted additional experiments with 30 clusters and observed that the regret decreases again, confirming GFQI's consistency. See Figure 1.

    - The performance of GFQI is dependent upon the strength of intra-cluster correlations, as shown in the panels with varying \(\psi\) in Figure 5 and detailed below. In most cases, GFQI demonstrates superior performance, and even in scenarios where it is not the best, its performance remains close to the optimal. 

        - **Low Intra-Cluster Correlation (\(\psi\) Small):**  
  In cases of weak intra-cluster correlation, as seen in the first panel of Figure 5, GFQI often achieves the lowest regret, though the improvement over other methods is less pronounced. Occasionally, GFQI may not perform the best, but the regret remains comparable to the best-performing method, suggesting robustness.

        - **High Intra-Cluster Correlation (\(\psi\) Large):**  
  When the intra-cluster correlation is strong, as seen in the third and fourth panels of Figure 5, GFQI consistently achieves the lowest regret across all cases, with significant improvements over other methods. This demonstrates that GFQI is particularly beneficial in scenarios with high intra-cluster correlations, where modeling these dependencies is critical for optimal policy learning.

    In summary, GFQI is most appropriate for settings with moderate to strong intra-cluster correlations. In weakly correlated settings, while its advantages may be less pronounced, GFQI remains robust and performs comparably to the best alternatives. We will revise the manuscript to include these discussions and provide further clarity on the scenarios where GFQI is most effective. Thank you for bringing this to our attention.


    <!--- We appreciate the reviewer’s insightful observation regarding the asymptotic behavior of GFQI as the number of clusters increases. Upon closer examination, we confirm that GFQI indeed exhibits improved performance with an increasing number of clusters. However, the results in Figure 5 might appear to suggest otherwise due to the limited range of clusters shown in the current plot. As the number of clusters continues to grow beyond the range displayed in Figure 5, the regret of GFQI continues to decrease, supporting its asymptotic improvement.
    - Additionally, regarding the observation that GFQI’s performance is not consistently better under all conditions, we note that the advantage of GFQI depends heavily on the intra-cluster correlation (𝜓). GFQI provides substantial benefits when the intra-cluster correlation is strong or when correctly specified correlation structures allow for more efficient policy learning. In cases of weaker correlation or greater misspecification, the relative advantage of GFQI diminishes, which aligns with our theoretical findings.
     In summary, the significance of the advantage of the GFQI verses other baselines depends on the sample size, the strongness of the intra-cluster correlation, and the correctly specified correlation structure.-->

### Novelty Justification:
* I believe that using GEE to improve the sample efficiency of FQI is a novel methodological approach. However, I am unclear of the performance of GFQI (the proposed approach), when the correlation structure is misspecified.
    - We appreciate the reviewer’s acknowledgment of the novelty of using GEE to improve the sample efficiency of FQI. Regarding the performance of GFQI under misspecified correlation structures, we acknowledge that this is a critical aspect to consider.

    - To address this concern, we have added a figure...
  
    - To address this concern, we perform an additional semi-synthetic analysis where the working correlation matrix is not correctly specified. Specifically, for each pair of individuals within the same cluster, we modified the global correlation coefficient $\rho$ by multiplying it by a random scale factor drawn from a uniform distribution over $[0.5, 2]$. Therefore, the correlations between individuals are not the same, leading to the correlation matrix not being exchangeable. When implementing GFQI, we use exchangeable as the working correlation matrix to show the robustness of our algorithm. The results are presented in Figure 2.

    - Our preliminary results show that GFQI retains its robustness under misspecification, aligning with the theoretical guarantee of asymptotic normality. However, as expected, its efficiency (e.g., variance reduction and regret minimization) is reduced compared to the case of a well-specified structure. We believe this additional analysis will provide greater clarity on the practical applicability and limitations of GFQI when the correlation structure is misspecified.

    - Thank you for highlighting this important consideration. We will ensure that the revised manuscript reflects this additional evaluation.

### Clarity
* Figure 1 is not sufficiently motivating. What is the key takeaway here? There are no axes labels, and while there are some clustered color groups (e.g., more blue in the lower left of the grid), it is not clear that there is strong intra-cluster correlation. Please explain this more.
    
    -  We use the following procedure to test the existence of intra-cluster correlation. We begin by conducting a linear regression of daily outcome (step count, sleep duration or mood score) against multiple predictors: the previous day's step count, sleep duration, mood score, whether receiving a message, and week index, using linear basis functions. Next, we extracted the residuals from this regression and analyzed them using a linear mixed effects model implemented using `lme4` package in R. This model incorporated both a fixed intercept and random intercepts for each institution. Finally, we use `ranova` function in `lmerTest` R package to test the significance of the random effect. The resulting p-value for all three variables (step count, sleep duration or mood score) are less than $2.2 \times 10^{-16}$, indicating the existence of intra-cluster correlation.
  
* In Figure 2, why does the delta between FQI/GFQI get smaller as the number of clusters increase? Is it because with a larger number of clusters, there is more shared structure between clusters that an algorithm can capitalize on? Is the distance between the clusters a factor here? Also is is the performance of GFQI similar to FQI if there is 1 cluster?

    - The decrease of delta with the number of clusters is expected. To elaborate, consider the case where the number of clusters increases to infinity. In that case, the total sample size grows to infinity, causing both FQI and GFQI to converge towards the same optimal policy. Consequently, this leads to the gap between them decaying to zero. 

    - This observation also aligns with the theoretical results in Theorem 1, where the convergence rate for all (G)FQI estimators, regardless of the working correlation matrix, is of the order $O((MN)^{-1/2})$, where $N$ is the number of state-action-reward-next-state tuples in the dataset, which depends on the number of clusters.
 
    - However, it is important to note that the constants proportional to these rates differ between the two methods. GFQI has a smaller constant. Therefore, although both methods converge to zero, GFQI does so at a faster rate. This makes our GFQI particularly valuable in finite-sample scenarios, which are common in biomedical studies.
  
    - As for the distance between clusters, GEE assumes that different clusters are independent, and thus it does not directly model the distance or relationship between clusters.

    - Regarding the case of a single cluster, GFQI can still gain better performance than FQI. The efficiency gains depend on the size of the cluster and the strength of the correlation. If the intra-cluster correlation is strong and the cluster size is relatively small, the correct working correlation matrix could provide some efficiency gains by decorrelating the data. 
    <!-- As the cluster size grows, both methods converge to the same consistent estimates, further reducing the potential difference. -->
    
   


* It would be nice to clarify that Figures 3 and 4 define causal relationships (which are referred to as paths) between components of the MDP. It is still not clear to me whether a correlation (rather than a causation) also violates the independence assumption.

Following your suggestion, we will mention that Figures 3 and 4 define causal relationships between components of the MDP in their titles. Independence is determined through associational relationships. In particular, if two random variables are independent, they must be uncorrelated. Therefore, if there is any correlation between them, regardless of the causal direction, the independence assumption is violated.


* You don’t define many terms including regret, MLE, and DGP. Please define these terms early in the paper for reader clarity.
    - Thank you for your feedback and for pointing out the need to define key terms and we will add a notation section at the beginning of the paper to introduce all the important notations.
      
* What is the convergence criteria?
    - The convergence criteria in our method are as follows: the algorithm is considered to have converged when the predicted responses from two consecutive fitted models have a relative difference smaller than $10^{-5}$, or when the maximum number of iterations (100) is reached. These criteria ensure both accuracy and computational feasibility during optimization.

    - We will include this detail in the revised manuscript to provide clarity on this aspect of the method.

* Why was a uniform behavior policy used? This seems like an easier setting and a little unrealistic. 
    - We use the uniform behavior policy because it is indeed used in the IHS. A uniform behavior policy ensures that the data is generated from a balanced distribution of actions, which can lead to more accurate estimation of the Q-function and optimal policy. This is particularly important in offline RL settings where the behavior policy is fixed and cannot be altered.
      

### Additional Comments:
* How do Generalized Estimating Equations work when the relationship between X and Y is not linear? I imagine there are several examples in which a linear structure is too restrictive. The same is true for the assumption on the optimal Q-function.

We appreciate this insightful comment. In response: 

- We agree that linear models can sometimes be restrictive. However, neither GEE nor FQI requires a linearity assumption. As such, linearity is not a key assumption to the proposed GFQI and it can be naturally extended to handle nonlinear environments. Please refer to our response to your first comment for details about these extensions. 

- Meanwhile, we would like to emphasize that our approach approximates the Q-function as a linear combination of basis functions rather than using a simple linear model. Such an approximation can capture non-linear structures —- for example, when the Q-function is smooth —- as the number of basis functions grows to infinity. This ensures that our method remains flexible and theoretically robust even in scenarios with non-linear environments. 

We will revise the manuscript to clarify these points and highlight the flexibility of our approach in handling non-linear structures. Thank you for bringing up this important perspective, which allows us to strengthen the clarity and positioning of our work.
