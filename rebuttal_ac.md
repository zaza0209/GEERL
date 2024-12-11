### Summary of Rebuttal to All Reviewers:

We would like to summarize our detailed responses to the feedback from all three reviewers, highlighting the key points addressed and the actions we plan to take to improve the manuscript.

---

#### **Reviewer u8Kq:**
1. **Significance Justification:**
   - The reviewer raised concerns about the trade-offs between training a single generalized policy versus separate models for each cluster. We provided strategies to address heterogeneity, such as incorporating cluster characteristics into the state representation and using transfer reinforcement learning techniques. We will revise the manuscript to clarify these points and discuss the trade-offs in more detail.

2. **Non-conventional Contributions:**
   - The reviewer acknowledged the novelty of integrating generalized estimating equations (GEE) into the RL framework, which bridges the gap between statistical methods for correlated data and modern RL techniques. We will emphasize this contribution in the revised manuscript to highlight its significance.

3. **Clarity:**
   - The reviewer suggested improving the clarity of the introduction by explicitly stating the problem and emphasizing that traditional RL methods assume i.i.d. data. We will revise the introduction accordingly to ensure clarity from the outset.

---

#### **Reviewer u1HY:**
1. **Soundness Justification:**
   - The reviewer questioned how the method addresses within-episode correlations and the non-smoothness of the operator in Theorem 1. We clarified that the method focuses on between-episode correlations and provided additional details on the assumptions and theoretical results.

2. **Regret Analysis:**
   - The reviewer raised concerns about the definition of regret and its scaling with key factors. We provided a detailed explanation of the regret bound and its dependence on factors such as the dimension of the state space, episode length, and number of episodes.

3. **Significance Justification:**
   - The reviewer requested more information about the real-world data and the strength of intra-cluster correlations. We will include detailed descriptions of the real-world data and discuss the generality of clustered data problems in the revised manuscript.

4. **Novelty Justification:**
   - The reviewer questioned the technical challenges of combining GEE and FQI. We elaborated on the challenges, such as adapting GEE for temporal dependencies, optimizing basis functions, and ensuring robustness under misspecified correlation structures. We will revise the manuscript to explicitly discuss these challenges.

5. **Clarity:**
   - The reviewer suggested streamlining the background explanation and focusing on the novel aspects of the proposed method. We will revise the manuscript to emphasize the differences between GFQI and FQI and provide a clearer discussion of the key assumptions and theoretical results.

---

#### **Reviewer MTVu:**
1. **Soundness Justification:**
   - The reviewer questioned the soundness of Theorem 1 and its implications for clustered data. We clarified that the Linear MDP assumption does not preclude the presence of clustered data and that the clustered structure is handled through GEE. We also discussed the robustness of GFQI under misspecified correlation structures.

2. **Significance Justification:**
   - The reviewer noted that GFQI does not always outperform baselines and questioned when GFQI is most appropriate. We explained that GFQI's advantage depends on the strength of intra-cluster correlations and the degree of misspecification of the correlation structure. We will include additional discussion on this in the revised manuscript.

3. **Novelty Justification:**
   - The reviewer acknowledged the novelty of using GEE to improve the sample efficiency of FQI but questioned its performance under misspecified correlation structures. We provided preliminary results showing that GFQI retains robustness under misspecification and will include this analysis in the revised manuscript.

4. **Clarity:**
   - The reviewer requested clarification on causal relationships and correlations in Figures 3 and 4. We explained that correlations can violate the independence assumption and will revise the manuscript to clarify this point.

---

### Attention to Reviewer u1HY:

We would like to bring your attention to Reviewer u1HY, who initially assigned a score of 1 and later increased it to 2. The reviewer acknowledged the helpful clarifications provided in our rebuttal but still raised concerns about writing clarity and presentation, which are not critical issues affecting the core contributions of our work. We have already developed a comprehensive plan to address these concerns during the revision process, as outlined above.

The reviewer also recognized the novelty of integrating GEE into the RL framework and acknowledged the significance of tackling the underexplored problem of intra-cluster correlations. We are committed to improving the clarity and presentation of the paper while preserving its scientific rigor and contributions.

Thank you for your attention and support during this rebuttal period.

Best regards,  
Authors
