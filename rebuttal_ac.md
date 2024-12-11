Dear AC,

Many thanks for handling our paper. To assist you in decision making, we provide a concise summary of our detailed responses prepared during the rebuttal to the feedback from all three reviewers:

---

#### **1. Theory:**
- Refined the regret bound in Theorem 2, including its scaling with key factors such as the dimension of the state space, episode length, and number of episodes.  
- Clarified the assumptions and technical details of our theories, including the uniqueness and marginal assumptions to ensure regularity.  
- Discussed the theoretical performance of GFQI under misspecified correlation structures, aligning with the theoretical guarantees of GEE.

---

#### **2. Empirical Studies:**
- Conducted additional numerical studies to investigate the performance of GFQI under misspecified correlation structures, showing its robustness to model misspecification.
- Provided detailed descriptions of the real-world datasets, including the number of users, clusters, and study duration, to improve transparency and understanding.  
- Discussed the strength of intra-cluster correlations in the real data and the generality of clustered data problems.  
- Explained why the gap between GFQI and FQI decreases as the number of clusters increases, highlighting the convergence properties of GEE.  

---

#### **3. Methodology:**
- Elaborated on the technical challenges of combining GEE and FQI, such as adapting GEE for temporal dependencies, optimizing basis functions, and ensuring robustness under misspecified correlation structures.
- Emphasized that the method focuses on between-episode correlations and does not explicitly address within-episode correlations, which is a limitation for future work.  
- Streamlined the background explanation and emphasized the differences between GFQI and FQI.  
- Discussed the appropriateness of GFQI in different scenarios, noting that its advantage depends on the strength of intra-cluster correlations and the degree of misspecification of the correlation structure.  
- Addressed the limitations of linear assumptions in GEE and RL by highlighting the flexibility of basis functions and link functions.

---


We also would like to bring your attention to Reviewer u1HY, who initially assigned a score of 1 and later increased it to 2. The reviewer acknowledged that all their concerns regarding contributions, theoretical soundness, and empirical relevance were addressed in our rebuttal. However, they still raised concerns about writing clarity and presentation, which we believe are not critical issues affecting the core contributions of our work. We have already developed a comprehensive plan to address these concerns during the revision process; refer to our response to Reviewer u1HY. 

Best regards,  
Authors
