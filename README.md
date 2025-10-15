# GRPO Core Algorithm Research 2025 Cheatsheet

These research focuses on exploring modifications to the core GRPO algorithm (reward, advantage, accumulation function, clipping, etc).


## DAPO \- Decoupled Clip and Dynamic Sampling Policy Optimization

> [https://arxiv.org/pdf/2503.14476v1](https://arxiv.org/pdf/2503.14476v1)   
> ByteDance  
> 2025-03-18

Asymmetric clipping, filters out uniform groups, token-shared gradient normalization, length penalties

* **Clip Higher:**  
  * clip\_high \> clip\_low  
* **Dynamic Sampling:**  
  * Ignore group if all outputs are correct or incorrect  
* **Token-Level Policy Gradient Loss:**  
  * Move the 1/|o\_{g, t}| (\# of token per generation) outside the summation to 1/|o\_{g}| (\# of token across all generation, shared)  
* **Overlong Reward Shaping:**  
  * Overlong filtering: mask entire sequence if it is too long (ignore gradients)  
  * Soft overlong punishment: add a R\_length term to reward:  
    * 0 if |s| \< L\_max \- L\_cache  
    * \-1 if |s| \> L\_max  
    * ((L\_max \- L\_cache) \- |s|) / L\_cache otherwise

## GPG \- Group Policy Gradient

> [https://arxiv.org/pdf/2504.02546v2](https://arxiv.org/pdf/2504.02546v2)  
> Alibaba, Inc.  
> 2025-04-17

Removes importance sampling ratio, uses direct policy gradients only

* **New objective:**  
  * sum\_group(sum\_tokens(-log pi(o) \* A))  
    * Note: no importance sampling ratio. pi(o) is the logprobs of the tokens  
  * Removes the need for reference model compared to GRPO  
* **Note:** theoretically interesting, but no follow up, third party benchmarks, or users (via VERL). Perhaps it doesn’t work well in practice?

## MT-GRPO \- Turn-Level Credit Assignment

> [https://arxiv.org/pdf/2505.11821](https://arxiv.org/pdf/2505.11821)  
> University of Minnesota, Minnesota, USA; Prime Intellect; Morgan Stanley  
> 2025-05-17

Separate turn-level advantages computed within each turn's responses only

* Early experimental stage, assumes the following 2-turn response structure:  
  * {query} \-\> turn1: {reason, search tool call} \-\> {search result} \-\> turn2: {reason, answer}  
* Turn1 rewards includes if the search result was used in turn2’s answer.  
* A\_{turn1} and A\_{turn2} are computed among the group’s turn1 and turn2 responses only  
* A\_final \= A\_{turn1} \+ lambda \* A\_{turn2}

## CPPO: Accelerating the Training Time of GRPO

> [https://arxiv.org/pdf/2503.22342](https://arxiv.org/pdf/2503.22342)  
> Xiamen University, Fujian, China; Shanghai Innovation Institute, Shanghai, China; Skywork AI, Singapore, Singapore; East China Normal University, Shanghai, China  
> 2025-03-28

Filters out responses with low absolute advantage before gradient computation

* The paper reveals that responses with low absolute value advantage score do not contribute much to the learning, so prunes them out early to save generation budget for other responses  
* **CPPO:**  
  * Variant 1: throw away responses where |A| \< threshold  
  * Variant 2: retain only the top P% responses sorted by |A|  
* Result: same accuracy even at P=87.5%, but reduces training time by 80\*  
  * **NOTE: the claim of training time reduction by 80% doesn’t make sense if this refers to end-to-end wall-clock training time. GRPO training time should be dominated (70-90%) by generation (autoregressive decoding), but this method only saves on prompt processing computation and does not reduce generation, which is a much smaller fraction (\~10%). Perhaps they mean “ignoring generation time, this method reduces the rest of the training by 80%”?** 

## ProRL \- Prolonged Reinforcement Learning

> [https://arxiv.org/pdf/2505.24864](https://arxiv.org/pdf/2505.24864)  
> Nvidia  
> 2025-05-30

Periodic reference policy reset with optimizer state reinitialization

* Builds on top of DAPO  
* **New KL Regularization**:  
  * L\_prorl \= L\_dapo \- beta \* D\_{kl}(policy\_new || policy\_ref)  
* **Reference Policy Reset:**  
  * Periodically update policy\_ref \= policy\_new and reinitialize optimizer state

## GSPO \- Group Sequence Policy Optimization

> [https://arxiv.org/pdf/2507.18071](https://arxiv.org/pdf/2507.18071)  
> Qwen Team, Alibaba Inc.  
> 2025-07-28

Sequence-normalized importance ratio. Only one stable with MoEs.

* **Group-Based Advantage Estimation:** A\_i \= (r(x, y\_i) \- mean(r(x, y\_i) for all i)) / std(r(x, y\_i) for all i)  
  * Same as TIC-GRPO  
* **Sequence-Level Importance Ratio:** s(y\_i) \= prod(r(y\_{i,t})/r\_old(y\_{i,t}))^(1/|y\_i|) for all t  
  * Same as TIC-GRPO but with the x^(1/|y\_i|) sequence-length normalizing term  
* **GSPO-Token:** s(y\_{i,t}) \= sg\[s(y\_i)\] / sg\[r(y\_{i,t})\] \* r(y\_{i,t})   
  * sg\[\] is the stop-gradient operation  
  * Intuition: this has equivalent value if s(y\_i) but the gradient of the original r(y\_{i,t})  
* **NOTE: only viable one for MoE models. Others need a complicated “replay” system and it’s still less stable than this one.**

## Edge-GRPO \- Entropy-Driven Advantage and Guided Error Correction

> [https://arxiv.org/pdf/2507.21848v1](https://arxiv.org/pdf/2507.21848v1)  
> Beihang University, Beijing, China  
> 2025-07-29

Adds correct solutions to prompts to guide learning, entropy-normalized advantages

* **Guided Error Correction:**  
  * For incorrect samples in a group, artificially add to prompt:  
    * 50% probability: reflection  
    * 25% probability: reflection \+ correct answer to solution  
    * 25% probability: replace entirely with reference solution  
* **Entropy-Driven Advantage:**  
  * Standardized\_entropy \= entropy / mean(entropy) of group  
  * A \= A / standardized\_entropy

TIC-GRPO \- A Trajectory-Corrected Approach with Fast Convergence  
[https://arxiv.org/pdf/2508.02833v2](https://arxiv.org/pdf/2508.02833v2)  
Peking University, Beijing, China; Ohio State University, Ohio, USA  
2025-08-07

Sequence-level importance ratio. Same as GSPO but not normalized by sequence length.

* **Group-Based Advantage Estimation:** A\_i \= (r(x, y\_i) \- mean(r(x, y\_i) for all i)) / std(r(x, y\_i) for all i)  
  * Same as GSPO  
* **Sequence-Level Importance Ratio:** s(y\_i) \= prod(r(y\_{i,t})/r\_old(y\_{i,t})) for all t  
  * Same as GSPO without the x^(1/|y\_i|) sequence-length normalizing term  
* **Modified Asymmetric Clipping:** instead of clip(pi\*A), uses min(pi\*A, clip(pi\*a))

## DCPO \- Dynamic Clipping Policy Optimization

> [https://arxiv.org/pdf/2509.02333v1](https://arxiv.org/pdf/2509.02333v1)  
> Bauchuan Inc  
> 2025-09-02

Probability-dependent dynamic clipping bounds, smooth advantage blending with history

* **Dynamic Adaptive Clipping Bounds**  
  * Old way: minimize variance of |r-1| \<= epsilon  
    * Result: clip r between 1-epsilon and 1+epsilon  
  * New way: minimize variance of |(r-1)p| \<= epsilon  
    * Result: clip r between 0.5 \+ 0.5\*sqrt(max(1-(4\*epsilon/q), 0)) and 0.5 \+ 0.5\*sqrt(1+(4\*epsilon/q))  
    * q \= old probability (the denominator of the importance ratio sample)  
* **Smooth Advantage Standardization**  
  * Instead of just A \= A\_new, also have A\_total \= (r \- mean\_batch)/(var\_batch)  
  * SA\_new \= ((i-1)/i)\*A\_new \+ (1/i)\*A\_total  
    * Puts more emphasis on A\_new  
  * SA\_toal \= (1/i)\*A\_new \+ ((i-1)/i)\*A\_total  
    * Puts more emphasis on A\_total  
  * A \= min(SA\_new, SA\_total)  
* **Token Level Mean Loss**  
  * Drop the batch size averaging term 1/|G|  
  * The A term is already standardized across the batch

## Group-in-Group Policy Optimization for LLM Agent Training

> [https://arxiv.org/pdf/2505.10978](https://arxiv.org/pdf/2505.10978)  
> Nanyang Technological University, Singapore, Singapore; Skywork AI, Singapore, Singapore  
> 2025-09-03

Episode-level and step-level advantages

* For LLM agents in limited environment states (i.e. environments with a small number of memory states, e.g. shopping online), one can group sequences of tokens into episodes, and within each episode, group steps by the environmental state (which should have duplicates in limited environments).  
* The Advantage term A\_episode is now standardized across episode, and A\_step is now standardized across steps.  
* Then A\_new \= A\_episode \+ omega \* A\_step

## GRPO-LEAD \- Length Dependent Rewards, Explicit Penalties, and Advantage Reweighting for Difficulty

> [https://arxiv.org/pdf/2504.09696](https://arxiv.org/pdf/2504.09696)   
> Johns Hopkins University, Maryland, USA   
> 2025-09-19

Length-dependent rewards, weight advantage by question difficulty

* **Length Dependent Rewards and Explicitly Penalties:** addition reward term  
  * R\_accuracy \= exp(-alpha z) if o is correct, \-1 if o is incorrect  
  * z is normalized length of sequence among the group, z \= (|o|-mu)/(sigma \+ epsilon)  
* **Advantage Reweighting for Difficulty:**  
  * rho\_q \= (\# correct) / (\# total) for a question q  
  * w(rho\_q) \= logistic(rho\_q; A, B, k, rho\_0) \= A \+ (B-A)/(1 \+ exp(k\*(rho\_q-rho\_0)))  
  * A^new\_i \=  
    * A^old\_i \* w(rho\_q) if A^old\_i \> 0  
    * A^old\_i \* (1-w(rho\_q)) if A^old\_i \<= 0  
    * This ensures for difficult problems (low rho\_q), correct responses get larger updates, and for easy problems (high rho\_q), incorrect responses get larger updates

## λ-GRPO \- GRPO is a process reward model

> [https://arxiv.org/pdf/2509.21154](https://arxiv.org/pdf/2509.21154)  
> Saarland University, Saarland, Germany  
> 2025-09-25

Transform token-level A to process-level A, then scale by 1/|λ|

* Main argument:  
  * GRPO is a non-trivial PRM when each group has overlapping prefixes, which common in practice  
* New formulation:  
  * Change GRPO’s inner term from token level to process level:  
    * Token level: r\_{i,t} \* A\_{i} \- D\_{i,t}, where D is the KL divergence regularization term  
    * Process level equivalent: |λ| \* (r\_{t}(λ) \* A(λ) \- D\_{t}(λ)) where λ is the current process  
  * Scale the inner term by 1/|λ| so that it just becomes: (r\_{t}(λ) \* A(λ) \- D\_{t}(λ))

## PSPO \- Probability Smoothing Policy Optimization

> [https://arxiv.org/pdf/2509.21282](https://arxiv.org/pdf/2509.21282)  
> University of Southampton, UK; The Alan Turing Institute, UK  
> 2025-09-25

Smooth important ratio with 1

* **Smoothed importance sampling ratio:** r\_smoothed \= (1-alpha)\*r\_old \+ alpha  
* **NOTE:** theoretically interesting, but performs the same as GRPO (num\_iter\_per\_sample=1) with the same fixed generation budget.

## DIVER \- Diversity-Incentivized Exploration for Versatile Reasoning

> [https://www.arxiv.org/pdf/2509.26209](https://www.arxiv.org/pdf/2509.26209)  
> Nanjing University, Jiangsu, China; Shanghai AI Laboratory, Shanghai, China; The Chinese University of Hong Kong, Hong Kong, China; Westlake University, Zhejiang, China  
> 2025-09-30

Add intrinsic diversity rewards (e.g. BLEU score) to rewards

* **Intrinsic Reward**  
  * R \= R\_original \+ lambda \* R\_intrinsic. R\_intrinsic can be:  
    * **Textual Diversity:** TD(o\_i) \= 1/(G-1)\\sum\_{j}(1-BLEU(o\_i, o\_j))  
    * **Equational Diversity:** ED(o\_i) \= |F(o\_i) \\ F\_{-i}|/|F(o\_i)| where F(o\_i) \= {formulas in o\_i}, F\_{-i} \= F(o\_j) for all j \!= i  
    * R\_intrinsic is clipped between \[0, sigma\]

## GRPO-λ \- Credit Assignment with GRRPO

> [https://arxiv.org/pdf/2510.00194v1](https://arxiv.org/pdf/2510.00194v1)  
> Huawei Technologies; Chandar Research Lab; Mila Quebec AI Institute, Quebec, Canada; University of Montreal, Quebec, Canada  
> 2025-09-30

Discounted accumulated importance ratios, clamped negative advantages to \-0.1

* Change importance sampling ratio to include all previous ones, discounted  
  * r\_new(o\_t) \= exp(sum\_{i=0, t}(gamma\*lambda)^i \* r(o\_{t-i}) \- sum\_{i=0, t}(gamma\*lambda)^i \* r\_old(o\_{t-i}))  
  * A variant on discount for long sequences: discount \= max((gamma\*lambda)^i, (gamma\*lambda)^{t-i}). This will put more weight on both the beginning and the end.  
* Clamps negative A to \-0.1 rather than using them directly.   
  * Negative A with accumulated log-probabilities create destructive gradients that destabilize training  
* Uses a supervised fine-tuning step before RL training to prevent reward hacking

## It Takes Two: Your GRPO is Secretly DPO

> [https://arxiv.org/pdf/2510.00977](https://arxiv.org/pdf/2510.00977)  
> University of Montreal, Quebec, Canada; McGill University, Quebec, Canada; MILA \- Quebec AI Institute, Quebec, Canada; University of Manitoba, Manitoba, Canada; The Chinese University of Hong Kong, Hong Kong, China; Zhejiang University, Zhejiang, China; Huawei Noah’s Ark Lab  
> 2025-10-01

Contrastive learning with group size 2, rejects uniform outcomes

* The author reframes GRPO as contrastive learning, similar to DPO  
* As a result, GRPO only needs group size of k=2 (one correct, one incorrect) to learn effectively  
* Reject prompts where both responses are both correct or both incorrect  
* Achieves competitive scores with k=16, but 70% less training time

## Dr. GRPO \- GRPO Done Right

> [https://arxiv.org/pdf/2503.20783](https://arxiv.org/pdf/2503.20783)  
> Sea AI Lab, Singapore, Singapore; National University of Singapore, Singapore, Singapore  
> 2025-10-06

Removes length denominator and std normalization from advantage function

* **Remove response length bias:**  
  * Remove the 1/|o\_t| denominator. If A \> 0 (correct response), it favors brevity, but if A \< 0 (incorrect response), it favors verbosity since it is penalized less  
* **Remove question-level difficulty bias:**  
  * In A, remove the denominator std(R), which is small for very easy or very hard problems where most outcomes are either correct or incorrect, which dominates the gradient updates  
* Mathematically equivalent to RLOO: A^{RLOO} \= k/(k-1) \* A^{DR.GRPO}
