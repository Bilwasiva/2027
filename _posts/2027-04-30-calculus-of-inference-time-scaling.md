---
layout: distill
title: "The Calculus of Inference-Time Scaling: Unifying Process Verification, Group Relative Policy Optimization (GRPO), and Search-Based Reasoning"
description: "An investigation of training signals, verification reliability, and budgeted inference in reasoning systems. Interactive exhibits examine group-relative gradients, process aggregation, synthetic search, and verifier selection risk."
date: 2027-04-30
future: true
htmlwidgets: true
hidden: true
authors:
  - name: Anonymous
bibliography: 2027-04-30-calculus-of-inference-time-scaling.bib
toc:
  - name: "From pretraining to inference-time compute"
  - name: "PPO, DPO and group-relative policy optimization"
  - name: "Group advantages and finite-sample bias"
  - name: "Process and outcome verification"
  - name: "Search under a generation budget"
  - name: "Verifier tail risk and over-optimization"
  - name: "Reference implementation and reproducibility"
  - name: "Limitations and open questions"
  - name: "Rollout dependence and length normalization"
  - name: "Verifier calibration and distribution shift"
  - name: "Adaptive stopping and shared verifier failures"
  - name: "Global inference budget allocation"
  - name: "Connections to recent literature and research agenda"
---

<!-- AUTHOR-REVIEW SCAFFOLD ONLY. Expanded AI-assisted technical editing and research-note draft on separate branch; NOT a submission-ready human-written manuscript. The supplied draft contains unsupported claims and illustrative curves; it is preserved separately. Write the main scientific narrative yourself, review the AI-assisted research exhibits and disclose assistance accurately before any formal submission. Do not submit this scaffold as a finished article. -->

**Author-review edition:** The following expanded technical notes, source-based edits, and exhibit interpretations require substantive personal revision. They are not a certification of human authorship or publication readiness.

## From pretraining to inference-time compute

### The problem is not just how long a model thinks

The original draft asks three linked questions: why group-relative optimization can remove a learned critic, when outcome or process feedback is useful, and how a fixed inference budget should be divided between sampling and revision. A concrete running example is a mathematical problem with multiple plausible approaches. A policy can produce a correct derivation, a derivation with a subtle arithmetic mistake, or an internally inconsistent argument that happens to reach the right answer. Training determines the distribution of these possibilities; verification determines how they are evaluated; search determines which possibilities are explored and returned.

A useful distinction is between three probabilities: the probability that one completion is correct, the probability that the explored set contains a correct completion, and the probability that the selected completion is correct. The first measures proposal quality; the second measures coverage; the third measures the end-to-end system. They are not interchangeable. A better proposal distribution can improve all three, but stronger search can improve coverage without improving selection if the verifier favors misleading candidates.

The training-compute budget pays for data, forward passes, backward passes, optimizer updates, and—in online RL—rollout generation and reward evaluation. The inference budget pays for prompt processing, generated tokens, verifier calls and possibly repeated search expansions. Equal output-token counts need not imply equal FLOPs, latency, memory use or dollar cost. Longer sequential traces have different critical-path latency from independent candidates that can run in parallel. The architecture exhibit below should be read as a conceptual decomposition, not as a claim about the private implementations of any named commercial reasoning model.

**Scope of the evidence.** The analytical examples use explicitly stated probability models. The executed studies use synthetic tasks and synthetic verifier scores; they are not empirical comparisons of pretrained LLMs. The cited papers provide separate published evidence. The purpose of the synthetic studies is to isolate mechanisms and make counterexamples reproducible, not to establish the best production reasoning architecture.


<!-- Author narrative: use a concrete example of a reasoning problem, distinguish learning a proposal distribution from selecting a candidate at inference, and explain why generated tokens, FLOPs, latency and money are different budgets. Anchor the perspective in the papers rather than claiming private implementation details of o1/o3. -->
<d-cite key="snell2025,deepseek2025,hariri2026"></d-cite>

<div class="l-page">
  <iframe title="Reasoning system architecture and process aggregation" src="{{ 'assets/html/2027-04-30-calculus-of-inference-time-scaling/process-supervision.html' | relative_url }}" frameborder="0" height="720px" width="100%" loading="lazy"></iframe>
</div>

## PPO, DPO and group-relative policy optimization

### Which parts of a reasoning system are actually being optimized?

PPO, DPO and GRPO occupy different positions in the post-training pipeline. PPO can use a learned critic to estimate token-level advantages and a clipped objective to limit some incentives for large policy changes. Generalized advantage estimation trades variance against bias through bootstrapping. With sparse terminal feedback, a prefix value function may be difficult to learn, especially while the policy distribution changes. This is a motivation for alternative estimators, not evidence that a critic is inherently inferior.

DPO instead operates on preference pairs. In the standard formulation, the model is encouraged to increase the relative log likelihood of the preferred completion compared with the rejected completion, relative to a fixed reference model. Its usual offline training objective does not require fresh online rollouts at every update. It still depends on the quality, coverage and provenance of the preference data.

The original outcome-supervised GRPO formulation groups completions for the same prompt and uses normalized rewards in place of the learned value estimator. A reward can come from an exact mathematical checker, a program's tests, a learned outcome model, or another specified evaluator. RL with verifiable rewards describes the source of feedback; GRPO describes the policy optimization algorithm. Neither requires a particular inference-time search algorithm.

| Method | Training signal | Learned critic | Important trade-off |
|---|---|---|---|
| Supervised fine-tuning | Demonstrated token sequences | No | Quality and coverage of demonstrations |
| DPO | Preferred/rejected pairs | No in standard DPO | Preference distribution and reference policy |
| PPO | Rewards and estimated advantages | Often yes | Value estimation, rollout and optimizer costs |
| GRPO | Relative rewards within a prompt group | No separate learned critic in original GRPO | More rollouts, dependence and normalization effects |

A separately trained critic adds weights, gradients, optimizer state and activations, but the percentage overhead depends on parameter sharing, precision, sharding and scheduling. Removing it does not remove the costs of the reference policy, rollout engine, reward model or policy optimizer.

### Interpreting clipping and KL

For a positive advantage, the clipped surrogate stops rewarding further increases in the importance ratio once it exceeds the upper threshold. For a negative advantage, it stops rewarding further decreases below the lower threshold. Clipping is not a hard constraint on the entire policy distribution and does not guarantee a fixed KL distance after an optimizer update. The KL expression below is pointwise nonnegative because exp(x) is at least 1+x. Under suitable support and current-policy sampling, its expectation equals forward KL; with reused old-policy samples the same expectation identity is not automatic. The estimator's nonnegativity does not imply convexity with respect to all neural network parameters.


<!-- Author narrative: contrast PPO's policy and learned value losses, DPO's paired-preference objective, and GRPO's within-prompt reward baseline. No universal critic memory tax. Distinguish verifier type from the optimizer. -->
<d-cite key="schulman2017,rafailov2023,shao2024"></d-cite>

$$
\rho_{it}=\exp(\log\pi_\theta(o_{it}\mid h_{it})-\log\pi_{\rm old}(o_{it}\mid h_{it})).
$$

$$
A_i=\frac{r_i-\bar r}{\sigma_r+\epsilon_a},\qquad
\ell_{it}=-\min(\rho_{it}A_i,\operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)A_i)+\beta k_3.
$$

$$
k_3=\exp(x)-x-1,\qquad x=\log\pi_{\rm ref}(o_{it}\mid h_{it})-\log\pi_\theta(o_{it}\mid h_{it}).
$$

<!-- Explain that k3 is nonnegative and has forward-KL expectation under current-policy sampling with suitable support. Off-policy rollout reuse does not automatically preserve that expectation. -->

## Group advantages and finite-sample bias

### A group mean is not an independent baseline

For binary rewards, a group containing k successful completions out of G has mean m=k/G and population variance m(1-m). When 0<k<G, its normalized advantages are proportional to the square roots of the odds: passing completions receive sqrt((1-m)/m), and failing completions receive -sqrt(m/(1-m)), ignoring the numerical stabilizer. For k=2 and G=8, these are approximately +1.732 and -0.577. The same successful reward receives different weight depending on the other samples in its group.

There is a subtle distinction between an unbiased estimate of the mean reward and an unbiased policy-gradient estimator. Let g_i denote a completion score gradient. Expanding the same-group baseline gives one diagonal term and G-1 cross terms. Independence and the zero-mean score identity remove the cross terms in expectation, but the diagonal contribution of each sample to its own baseline remains. This produces the factor (G-1)/G shown below. A leave-one-out baseline, which excludes the completion being updated, removes this particular finite-sample factor. Standardization, token weighting, clipping and off-policy reuse introduce other effects; the simple identity does not justify calling the full GRPO update unbiased.

Uniform reward groups have no centered reward signal. They can arise by chance even when a task is neither impossible nor solved. At p=0.1 and G=8, the probability of mixed outcomes is approximately 56.95%; at G=32 it is approximately 96.57%. These are conditional IID calculations, not measurements of an actual training run. Larger groups can improve the chance of reward diversity, but may consume rollouts that could instead be allocated to additional prompts. KL regularization can still have a gradient when the centered reward contribution vanishes.

**Reading the explorer:** vary group size and success probability, then compare the raw score-function estimator, the centered estimator, leave-one-out centering and the z-score estimator. Their exact means and standard deviations refer to the simplified Bernoulli-logit model. A variance comparison only makes sense alongside the target quantity and bias of each estimator.


<d-cite key="shao2024,liu2025,yu2025"></d-cite>

$$
P(\text{mixed rewards}\mid p,G)=1-p^G-(1-p)^G.
$$

$$
\mathbb E\!\left[\frac1G\sum_i (r_i-\bar r)\nabla\log\pi(o_i)\right]
=\frac{G-1}{G}\nabla\mathbb E[r]
$$

<!-- Assumptions: IID on-policy rollouts, score-function gradient, no standardization, clipping or length normalization. A leave-one-out baseline removes this factor. The result is not a general proof of unbiased GRPO or Dr. GRPO. Explain how group normalization, length normalization and dynamic sampling differ. -->

<div class="l-page">
  <iframe title="Exact group gradient and clipping explorer" src="{{ 'assets/html/2027-04-30-calculus-of-inference-time-scaling/grpo-explorer.html' | relative_url }}" frameborder="0" height="640px" width="100%" loading="lazy"></iframe>
</div>

## Process and outcome verification

### Correctness, progress and reward are different quantities

An outcome checker evaluates the final answer against a specified criterion. For mathematics this could be symbolic equivalence; for code it could be hidden tests; for theorem proving it could be a proof checker. Learned outcome reward models estimate a score and can make false-positive or false-negative errors. Even deterministic tests may fail to represent the intended specification.

A process reward model scores a step or prefix. Three possible targets should be separated: whether the current step is locally valid, whether the complete prefix is valid, and whether the prefix increases the probability that a subsequent solver reaches a correct answer. The third is related to progress or value-to-go and depends on the continuation policy. A locally valid step may lead nowhere useful, and a promising plan may include a repairable mistake. A process verifier is therefore not automatically a perfect search value function.

For scores of 0.95 repeated five times, the product is about 0.774; for twenty such scores it is about 0.358. A product is a valid joint probability when each factor has the appropriate conditional interpretation, not merely because every factor is between zero and one. A minimum score highlights the weakest step but ignores the accumulation of moderately risky steps. A last-step score can miss earlier faults; a geometric mean normalizes the length dependence but changes the target. All four should be compared using independent correctness labels and a consistent step segmentation procedure.

A proof-checking analogy is useful: checking every line may expose an early invalid inference; checking only the final result may miss a faulty proof with a coincidentally correct answer. Conversely, a strict step-level check can reject a legitimate alternative derivation if the verifier does not understand it. Process-based pruning can save generation but can also eliminate the only viable branch. The architecture and aggregation exhibit allows these scoring choices to be explored without assuming one universally best method.


<d-cite key="lightman2024"></d-cite>

<!-- Author narrative: distinguish exact outcome checkers, learned outcome reward models, step-level correctness and value-to-go. Compare minimum, last, product and geometric mean. Product is a chain-rule probability only with appropriate conditional probabilities. Explain label noise, segmentation, calibration, and backtracking. Return to the interactive process aggregation exhibit above. -->

## Search under a generation budget

### Coverage, selection and repair

Best-of-N sampling generates independent candidates conditional on a prompt and uses a scoring function to select one. Oracle coverage is the probability that at least one candidate is correct. With a learned verifier, selected-answer accuracy cannot exceed coverage on the same candidate set. The difference between them is a measurable selection gap. Self-consistency is not oracle selection: it aggregates answers and can favor a common incorrect answer if the candidate distribution is systematically wrong.

For IID samples with p=0.05, the oracle coverage after 32 samples is about 80.63%, while 64 samples yield about 96.25%. This improvement is conditional on a fixed positive p. If p=0, independent sampling never discovers a correct answer. If the model's candidates are correlated or if prompts have heterogeneous success probabilities, substituting a global average into the IID formula is generally incorrect.

Sequential revision uses earlier candidates and critiques as context. If an incorrect answer is repaired with probability a and a correct answer becomes incorrect with probability b, a simple two-state model yields p_(t+1)=a+(1-a-b)p_t. When a+b>0, its stationary success probability is a/(a+b). Revision can improve or degrade quality depending on the initial state distribution and the transition probabilities. This is an illustrative Markov model, not a guarantee about language-model self-correction.

Beam search retains a limited number of promising prefixes at each expansion depth. It can benefit from meaningful intermediate scores but risks premature pruning. UCT selects among visited actions using empirical terminal reward and an exploration bonus; its performance depends on the quality of rollout estimates, branching, depth and score noise. The two algorithms are different, and the synthetic experiments explicitly distinguish them.

### How to read the executed search experiments

The first study contains 36 conditions with 1,200 trials each. It varies the probability of a valid reasoning step, the separation between valid and invalid verifier scores, and the maximum generated-step budget. Its revision transition probabilities are assumptions. A high measured revision score is evidence about this particular model, not evidence that revision dominates real LLM search.

The second study uses fixed depth-seven binary trees shared across algorithms. It reports 3,200 distinct task trees, 16,000 task-condition evaluations and 48,000 algorithm evaluations. Each traversal is charged as a generated edge even if an implementation might cache it. Beam search uses intermediate scores; UCT backs up terminal proxy scores. The methods therefore share a maximum generation budget but not necessarily equal verification compute.

For the step-validity setting 0.65, stronger score separation and a maximum budget of 224 edges, the measured selected accuracy was 56.625% for sampling, 73.125% for beam and 61.500% for UCT. Under weaker score separation, sampling reached 33.25%, beam 57.625% and UCT 37.25%. These numerical differences illustrate the importance of verifier information and task structure. They do not establish universal rankings. A task with no valid terminal leaf cannot be solved by any of these methods under the fixed-tree model.

For a real system, record total generated tokens, prefilling, verifier invocations, prefix reuse, latency percentiles, throughput, GPU-hours and selected correctness separately. A comparison at equal generated tokens is not necessarily a comparison at equal total compute.


<d-cite key="snell2025,huang2024,hariri2026"></d-cite>

$$
P(\text{at least one correct candidate})=1-(1-p)^N.
$$

<!-- Author narrative: compare independent sampling with majority voting, reranking, beam search, MCTS and sequential repair. Distinguish oracle coverage from selection accuracy, and explain why revision and search require additional assumptions. Report generated steps separately from verifier compute, memory and latency. -->

<div class="l-page">
  <iframe title="Executed synthetic search study" src="{{ 'assets/html/2027-04-30-calculus-of-inference-time-scaling/budgeted-search.html' | relative_url }}" frameborder="0" height="800px" width="100%" loading="lazy"></iframe>
</div>

<d-cite key="kocsis2006,wang2023"></d-cite>

$$
a^*=\arg\max_a\left[\widehat Q(s,a)+c\sqrt{\frac{\log N(s)}{N(s,a)}}\right].
$$

<div class="l-page">
  <iframe title="Executed finite-tree UCT, beam and sampling comparison" src="{{ 'assets/html/2027-04-30-calculus-of-inference-time-scaling/uct-search.html' | relative_url }}" frameborder="0" height="800px" width="100%" loading="lazy"></iframe>
</div>

<!-- Author: contrast the fixed-tree study with the previous synthetic revision model. Analyze coverage versus selected accuracy, solvability ceilings and verifier-cost asymmetry. The UCT method backs up proxy terminal scores, not true labels. Do not generalize rankings beyond the assumed environments. -->


## Verifier tail risk and over-optimization

### Why selecting the largest score changes the error distribution

Selecting candidates by maximum proxy reward is different from evaluating random candidates. If a small subset of incorrect answers receives exceptionally high scores, increasing N increases the chance that this subset is encountered. A proxy can therefore look accurate on random samples while becoming unreliable on selected samples. This is a selection-distribution shift, closely related to reward-model overoptimization.

In the rare-tail model used here, each independent candidate is correct with probability p. Conditional on being incorrect, it receives a catastrophic false-positive score with probability h. Every catastrophic score exceeds every correct score. Let a=1-(1-p)h be the probability of avoiding a catastrophic error on one draw, and b=(1-p)(1-h) the probability of an ordinary incorrect draw. The probability of no catastrophic error and at least one correct candidate is a^N-b^N. This equals selection accuracy only when every ordinary correct candidate outranks every ordinary incorrect candidate. Otherwise it is an upper bound under the specified assumptions.

The simulated score distributions allow ordinary overlap and therefore do not assume perfect ordinary ranking. With base correctness 0.25 and a 2% rare-tail rate conditional on incorrectness, selected correctness was 75.50% at N=16, 36.27% at N=64, and 1.70% at N=256. Meanwhile the mean selected proxy score rose from about 1.103 to 1.545. The experiment comprises 9,000 independent candidate streams across three tail settings and 2,304,000 candidate draws. Budgets within a stream reuse nested candidates and must not be counted as independent replications.

A second verifier with a different error mechanism improved selected accuracy in this example, but it did not remove the failure mode. A formal verifier may offer stronger guarantees for the property it checks; incomplete unit tests or an incorrect formal specification can still accept behavior contrary to the intended goal. Stronger assurance comes from independently checking the target property and understanding the limitations of the verifier—not merely from maximizing its score.


<d-cite key="gao2023"></d-cite>

$$
P(\text{no catastrophic proxy error and at least one correct})
=(1-(1-p)h)^N-[(1-p)(1-h)]^N.
$$

<!-- Author narrative: derive this upper bound for the specified mixture model and compare it with measured synthetic accuracy. Discuss the limitations of two-score averaging, shared errors, formal verifier specification gaps and adaptive stopping. Do not present the synthetic curve as a replication of the Gao et al. fitted law. -->

<div class="l-page">
  <iframe title="Executed verifier rare-tail ablation" src="{{ 'assets/html/2027-04-30-calculus-of-inference-time-scaling/verifier-tail-risk.html' | relative_url }}" frameborder="0" height="800px" width="100%" loading="lazy"></iframe>
</div>

## Reference implementation and reproducibility

### Reproduction checklist and scope of the code

The displayed PyTorch implementation isolates the policy loss; it does not implement full rollout collection, distributed optimizer state, a reward service, reference-model synchronization or checkpointing. Its inputs represent token log probabilities for each prompt and each completion. Completion masks exclude padding and prompt positions. Reward and baseline computations are detached from the policy update. Population standard deviation is used explicitly to avoid ambiguity across implementations.

Three reductions are shown: the mean of per-completion means, a global mean over valid tokens, and a fixed denominator with centered rewards. The alternatives illustrate different normalization choices; they should not be treated as complete reproductions of particular RL training frameworks. The example uses a numerical floor for log products of step probabilities. In extreme ratio regimes a production implementation should also monitor overflow, finite gradients, KL, policy entropy, clip fraction, valid completion length and rollout staleness.

The five stochastic exhibit generators are embedded in their corresponding HTML pages. Copy a generator into a JavaScript file and run it with Node.js; no pretrained-model checkpoint is needed. Fixed pseudorandom seeds make the examples reproducible. Deterministic execution does not demonstrate generalization to unseen real-world data.

A recommended real-model follow-up would freeze the generator and verifier checkpoints, define independent calibration and test prompts, pre-specify the compute budgets and stopping criteria, and report prompt-level paired uncertainty. Keep exact-checker labels separate from verifier scores; record all failed and truncated attempts. When several budgets and methods are compared, treat unadjusted intervals as descriptive or use a pre-specified multiple-comparison procedure. Release code, split definitions, configurations and seeds together.


<!-- The three stochastic figures contain the deterministic Node.js generators, parameters, and output tables. Add your own explanation of the tested PyTorch loss, masks, population-standard-deviation convention, fixed denominator and off-policy caveats. The reference implementation is supplied separately for author review. -->

```python
import torch

def grpo_loss(new, old, ref, rewards, mask, eps=.2, beta=.04,
              mode='grpo', max_completion_length=None):
    # [B,G,T] log-probabilities, [B,G] rewards.
    if new.ndim != 3 or new.shape != old.shape or new.shape != ref.shape or new.shape != mask.shape or rewards.shape != new.shape[:2]:
        raise ValueError('Incompatible shapes')
    if mode not in ('grpo','fixed','token') or new.shape[1]<2 or not 0<=eps<1 or beta<0:
        raise ValueError('Invalid configuration')
    mask=mask.bool(); lengths=mask.sum(-1)
    if (lengths==0).any(): raise ValueError('Each completion needs a valid token')
    r=rewards.detach().float(); adv=r-r.mean(-1,keepdim=True)
    if mode!='fixed': adv=adv/(r.std(-1,unbiased=False,keepdim=True)+1e-8)
    n=torch.where(mask,new.float(),0.)
    o=torch.where(mask,old.detach().float(),0.)
    f=torch.where(mask,ref.detach().float(),0.)
    ratio=(n-o).exp(); a=adv.unsqueeze(-1)
    surrogate=torch.minimum(ratio*a,ratio.clamp(1-eps,1+eps)*a)
    x=f-n
    kl=torch.expm1(x)-x
    loss=torch.where(mask,-surrogate+beta*kl,0.)
    if mode=='grpo': return (loss.sum(-1)/lengths).mean()
    if mode=='token': return loss.sum()/lengths.sum()
    if max_completion_length is None or max_completion_length < lengths.max().item():
        raise ValueError('Specify a fixed maximum completion length')
    return loss.sum()/(new.shape[0]*new.shape[1]*max_completion_length)

def process_scores(p, mask):
    # Candidate-level aggregation; scores need not be calibrated.
    mask=mask.bool()
    if p.shape!=mask.shape or p.ndim!=2 or p.shape[1]==0 or (~mask.any(-1)).any():
        raise ValueError('Invalid step mask')
    if ((p[mask]<0)|(p[mask]>1)|~torch.isfinite(p[mask])).any():
        raise ValueError('Invalid step score')
    logp=torch.where(mask,p.clamp_min(1e-30).log(),0.)
    last=(torch.arange(p.shape[1],device=p.device).expand_as(mask).masked_fill(~mask,-1)).max(-1).values
    return {'log_product':logp.sum(-1),
            'minimum':p.masked_fill(~mask,float('inf')).min(-1).values,
            'last':p.gather(-1,last[:,None]).squeeze(-1),
            'geometric_mean':(logp.sum(-1)/mask.sum(-1)).exp()}
```

## Limitations and open questions

### What the results establish—and what they do not

The analytical examples establish consequences of their explicit assumptions. The simulations demonstrate that the corresponding failure modes can occur in controlled environments. They do not estimate the real prevalence of catastrophic verifier errors, actual gradient variance of a pretrained reasoning model, the speed of a GPU training stack, or the best search algorithm on an established benchmark. No pretrained model has been trained or evaluated here.

Several important variables have been simplified: task difficulty is represented by a probability or fixed tree; generation cost is counted as steps, edges or candidate units; scores are generated from artificial distributions; revision transitions are fixed; verifier costs are illustrative; token length, cache locality and deployment concurrency are not modeled jointly. The calibration shift is deliberately constructed. Some synthetic strategies know condition parameters that would need to be estimated in deployment.

The open question is whether these mechanisms predict failure on held-out problems when the generator, verifier and search procedure are real learned systems. A stronger empirical test would use separate open-weight models, independently verifiable problems, distribution-shifted tasks, a fixed generation budget and measured verifier cost. It would compare random-candidate calibration with calibration on selected candidates, and measure the sensitivity of decisions to adversarial or correlated verifier errors.


<!-- Author narrative required: no pretrained LLM or real hardware benchmark, artificial step-success and revision model, simplified proxy scores, cost omits verifier calls, no claim about actual proprietary models. Propose a real model experiment with fixed prompt split, shared sampling budget, calibrated verifier, blinded gold labels, length-stratified metrics, and uncertainty across problems. Verify conflicts of interest and AI-use disclosure personally. -->


## Rollout dependence and length normalization

### Dependence can cap the value of a larger group

The dependence exhibit uses a mixture in which a shared Bernoulli outcome is drawn with probability c; otherwise the outcomes are independently drawn. Pairwise correlation is c in this particular model. With probability c, all outcomes are identical, so only the independent component contributes to mixed rewards. The sample-mean variance follows from the sum of G marginal variances and G(G-1) pairwise covariances.

At G=32 and c=0.5, the variance-equivalent effective sample size is 32/16.5, or approximately 1.94 independent observations. This does not imply that replacing G by 1.94 in the oracle-coverage equation yields the correct probability. Effective sample size describes variance; tail events depend on the full joint distribution. Across different prompts, even conditionally independent samples can look correlated because prompt difficulty varies. Separate these two sources of dependence.

The length example compares an 800-token passing completion with a 200-token failing completion. Per-completion mean reduction gives each completion equal total scalar weight before multiplication by score gradients, while the global token mean assigns more total weight to the longer completion. Fixed-denominator centering changes both length and reward scaling. The direction and size of the actual parameter gradient depend on the score gradients, not just the scalar coefficients. This motivates logging the joint distribution of reward, token length, entropy and group composition during RL.


<d-cite key="shao2024,liu2025,yu2025,chen2021"></d-cite>

$$
P(\text{mixed})=(1-c)[1-p^G-(1-p)^G],\qquad
G_{\rm eff}=\frac{G}{1+(G-1)c}.
$$

<!-- Author: derive the mixture model; separate within-prompt correlation from heterogeneous prompt difficulty. Explain the independence assumptions and why an ESS for the sample mean cannot be substituted into pass@N. Interpret the scalar gradient coefficients, not full gradient vectors. -->

<div class="l-page">
  <iframe title="Exact correlated rollout model and length-weighting comparison" src="{{ 'assets/html/2027-04-30-calculus-of-inference-time-scaling/correlated-rollouts.html' | relative_url }}" frameborder="0" height="800px" width="100%" loading="lazy"></iframe>
</div>

<div class="caption">Supplementary AI-assisted research exhibit: Exact correlated rollout model and length-weighting comparison. The assumptions and reproduction instructions are included in the figure.</div>


## Verifier calibration and distribution shift

### A calibrated confidence is not necessarily a better ranker

Calibration asks whether examples assigned a predicted probability near x are actually correct about x of the time. Discrimination asks whether correct examples receive higher scores than incorrect examples. AUROC measures a ranking property, whereas Brier score and negative log likelihood measure probabilistic predictions. Expected calibration error (ECE) depends on the binning procedure and sample size.

Temperature scaling divides logits by a positive scalar before the sigmoid or softmax. For binary sigmoid scores it preserves their ordering. Consequently it cannot, by itself, change which candidate wins an argmax over those scores. It can change confidence thresholds, uncertainty decisions and cost-aware stopping if those decisions use the transformed probabilities.

The synthetic study selects a temperature of 2.60 on 16,000 validation examples and freezes it for separate IID and shifted test sets of 16,000 examples each. IID ECE declines from 0.1187 to 0.0180, and Brier score from 0.1869 to 0.1700, while AUROC remains about 0.8269. Under the constructed label shift, calibrated ECE is still about 0.1963. The experiment illustrates how calibration can improve in-distribution but fail under shift, even when ranking is similar.

In practical evaluation, fit calibration parameters on a disjoint split; assess reliability at the actual selection budget; stratify by difficulty, length and task family; and report uncertainty as well as aggregate ECE. A probability estimated on random model outputs is not automatically valid for the maximum-scoring output.


<d-cite key="guo2017,zhou2026,li2026"></d-cite>

$$
\mathrm{Brier}=\frac1n\sum_i(\widehat p_i-y_i)^2,\qquad
\widehat p_i(T)=\sigma(z_i/T).
$$

<!-- Author: distinguish ranking, calibration, thresholding and selected-answer accuracy. Interpret unchanged AUROC, reduced IID ECE and persistent shifted ECE. The temperature is selected on an independent validation split; monotonic calibration cannot improve argmax ranking by itself. -->

<div class="l-page">
  <iframe title="Held-out reliability diagrams and calibration metrics" src="{{ 'assets/html/2027-04-30-calculus-of-inference-time-scaling/calibration-lab.html' | relative_url }}" frameborder="0" height="800px" width="100%" loading="lazy"></iframe>
</div>

<div class="caption">Supplementary AI-assisted research exhibit: Held-out reliability diagrams and calibration metrics. The assumptions and reproduction instructions are included in the figure.</div>


## Adaptive stopping and shared verifier failures

### Ensembles are only as diverse as their error mechanisms

The shared-tail model has two proxy verifiers, each with marginal false-positive tail probability h conditional on an incorrect candidate. With probability ch they share the same tail event; otherwise the residual events are independent. The resulting joint tail probability below equals h squared when c=0 and equals h when c=1. Thus even a small shared component can substantially reduce the nominal advantage of combining verifiers.

The executed study uses nine conditions, each with 2,000 validation streams and 2,000 independent test streams of 64 candidates. Thresholds are chosen on validation using accuracy minus 0.012 times an illustrative cost. Each candidate costs one generation unit plus 0.25 per verifier evaluation. These are unitless modeling choices, not measured FLOPs. The minimum of two proxy scores is a conservative selection heuristic; it is not a calibrated probability.

At base correctness p=0.25 and h=0.02 with independent tail errors, fixed-64 single-verifier selection achieved 37.60% accuracy, compared with 98.15% for the minimum of two. When the shared component was 0.75, the corresponding two-verifier result dropped to 47.30%. The single adaptive rule achieved 92.65% in the shared-tail setting, with mean illustrative cost about 6.53 rather than 80 for fixed-64. All these figures are synthetic test-set observations. Because the adaptive thresholds were tuned separately for each known condition, they do not establish the performance of a deployable router that must infer the condition.

In production, independently measure the errors of each verifier and their joint false-positive tail; compare ensembles at matched total verification cost; monitor changes after generator updates; and provide a path for abstention or escalation when confidence is insufficient. A collection of identical models or models trained on similar data may provide less independent evidence than its member count suggests.


<d-cite key="gao2023,snell2025,li2026,zhou2026"></d-cite>

$$
h_{\rm joint}=ch+(1-ch)\left(\frac{(1-c)h}{1-ch}\right)^2.
$$

<!-- Author: explain independent and shared tail events. Compare the fixed-64 and adaptive policies, uncertainty, and cost. The synthetic utility price and condition-specific validation thresholds are assumptions, not measured hardware costs or a deployable difficulty oracle. Discuss safety of abstention and the limits of correlated verifier ensembles. -->

<div class="l-page">
  <iframe title="Held-out compute-aware selection and correlated error ablation" src="{{ 'assets/html/2027-04-30-calculus-of-inference-time-scaling/adaptive-budget.html' | relative_url }}" frameborder="0" height="800px" width="100%" loading="lazy"></iframe>
</div>

<div class="caption">Supplementary AI-assisted research exhibit: Held-out compute-aware selection and correlated error ablation. The assumptions and reproduction instructions are included in the figure.</div>


## Global inference budget allocation

### A budget is an upper bound, not a target

Under IID oracle coverage, the expected improvement from adding one sample to an existing allocation of n is p(1-p)^n. These marginal gains decrease with n, so allocating the next unit to the largest remaining gain is optimal for the separable discrete oracle problem with a fixed unit cost and a minimum allocation. Imperfect verification can change the shape of the benefit curve: the marginal gain can become negative.

The three-prompt allocation exhibit makes this effect concrete. With success probabilities 0.05, 0.35 and 0.70, a shared catastrophic error parameter of 0.02 and at most 48 candidate units, equal allocation gives 16 candidates to each prompt. Exhaustive enumeration of feasible integer allocations selects 25, 8 and 4 candidates respectively. Expected correct answers increase from approximately 2.136 to 2.294 out of three, with 11 candidate units unused. These are exact calculations for the stipulated ideal ranking model; they are not measured improvements in a serving system.

A practical scheduler must estimate success and verifier-risk curves for unseen requests, account for differing sequence lengths and batching, and satisfy tail-latency constraints. If a harder prompt has almost zero probability of success, allocating all remaining resources to it may not be efficient. If the estimated risk of false acceptance is high, escalation to a more reliable checker may be preferable to generating more candidates. A useful production report would include a Pareto comparison of accuracy, risk, coverage, latency and cost rather than collapsing all choices into one number.


<d-cite key="snell2025,setlur2026"></d-cite>

$$
\max_{\substack{n_i\geq1\\\sum_i n_i\leq B}}
\sum_i\left[(1-(1-p_i)h_i)^{n_i}-((1-p_i)(1-h_i))^{n_i}\right].
$$

<!-- Author: derive the exact objective only under the ideal ranking and rare-tail assumptions in the exhibit. Discuss marginal returns, unused compute, the distinction between a known-success oracle and learned routing, latency/SLO constraints and verifier cost. -->

<div class="l-page">
  <iframe title="Exact heterogeneous-workload compute allocation explorer" src="{{ 'assets/html/2027-04-30-calculus-of-inference-time-scaling/compute-allocation.html' | relative_url }}" frameborder="0" height="800px" width="100%" loading="lazy"></iframe>
</div>

<div class="caption">Supplementary AI-assisted research exhibit: Exact heterogeneous-workload compute allocation explorer. The assumptions and reproduction instructions are included in the figure.</div>


## Connections to recent literature and research agenda

### Reading map and application checklist

The foundational policy-optimization papers provide the training-side framework; outcome and process supervision papers explain how feedback can be assigned; the inference-scaling studies examine how generated candidates are explored and selected; reward-overoptimization and calibration work explain why optimizing a proxy can be risky. Recent research on process progress, verification dynamics and learned exploration expands this picture. The article's controlled examples are complementary to these studies, not replications of their benchmark results.

| Application | Gold correctness criterion | Possible verifier failure | Metric or experiment |
|---|---|---|---|
| Mathematical reasoning | Independent symbolic or human-reviewed solution | Incorrect but plausible derivation | Coverage, selection gap, proof validity |
| Code generation | Hidden tests and a validated specification | Passing incomplete tests | Functional correctness, test coverage, runtime |
| Theorem proving | Valid proof of the intended theorem | Misstated theorem or invalid informal translation | Checked proof rate, proof cost |
| Retrieval-augmented QA | Claim-level evidence and independent answer labels | Shared citation or evidence errors | Grounded precision, selective risk |
| Tool-using agents | Verified environment state and constraints | Optimizing an incomplete success criterion | Task success, violations, tool cost |

The recurring lesson is to distinguish a useful mechanism from a universal guarantee. More rollouts may yield redundant evidence; a stronger-looking proxy may select worse outputs; more expensive search may not be more effective; and a lower calibration error does not imply better ranking. The proper comparison measures proposal quality, coverage, selection correctness, and resource use together.

### Author review questions

What is the specific scientific contribution beyond the existing literature? Which assumptions are necessary for each derivation? Which numbers are newly computed synthetic observations and which are cited results? Are there counterexamples to each proposed recommendation? Can a reader reproduce the experiment without a hidden dependency? Would a different score distribution, task family or cost function reverse the conclusion? These questions should guide the final personally written synthesis.


<d-cite key="uesato2022,wang2024shepherd,setlur2025,zhou2026,li2026,setlur2026,hariri2026"></d-cite>

<!-- Author literature synthesis: Uesato et al. and Math-Shepherd compare outcome/process supervision; Setlur et al. (2025) distinguish process progress under a prover policy from absolute step correctness; Zhou et al. (2026) study generator/verifier capability and task difficulty; Li et al. (2026) diagnose reward-model degradation during search; Setlur et al. (2026) study compute extrapolation through learned exploration. Discuss how each relates to the corresponding exhibits and where their real-model evidence differs from our synthetic evidence. Do not imply these studies were reproduced. Potential applications: code generation with hidden tests, formal theorem proving, tool-using agents, grounded QA, and scientific hypothesis generation. For each, identify an independent gold label, a verifier failure mode, a cost constraint, and a measurable outcome. -->
