# QMIND — Independent HumanEval+ evaluation (2026-11-20)

**Status: completed and verified.** Independent Docker-isolated EvalPlus 0.3.1 grading was successful on the 12 matched saved baseline/steered HumanEval+ generations from the team's `demo/11-20` branch (source commit `8fe09ca8353c132fa3e2cb902eb0540d7a2d45f3`).

- [Successful GitHub Actions run](https://github.com/jacobpowerr/qmind-coding-eval/actions/runs/44862912776)
- [Full evaluation artifacts: results, exact code, raw responses, manifests, and SHA-256 hashes](https://github.com/jacobpowerr/qmind-coding-eval/actions/runs/44862912776/artifacts/14689999384)
- [Team's experimental methods and original control results](https://github.com/aiejvn/test-time-reasoning/blob/demo/11-20/linear_probe/REPORT.md)

## Question and setup

Does additive steering of DeepSeek-R1-Distill-Llama-8B's layer-29 residual stream along a GSM8K-trained `prompt_last` correctness-probe direction improve coding-task accuracy? This is also a test of transfer from math to code. The intervention is `h <- h + alpha * u` at every token, with `alpha=3.8645427703857425` (0.1 typical residual norm). Both conditions used greedy decoding, 4,096 max new tokens, and batch size 8.

## Results

| Metric | No intervention | Probe steering |
|---|---:|---:|
| HumanEval base correct | 10/12 (83.3%) | 10/12 (83.3%) |
| HumanEval+ correct | 10/12 (83.3%) | 10/12 (83.3%) |
| Truncated responses | 2/12 | 2/12 |
| Mean response tokens | 2353.6 | 2301.1 |

**Net HumanEval+ accuracy difference: 0 percentage points.** One task was fixed (HumanEval/143); one was broken (HumanEval/154); ten did not change correctness. Exact paired McNemar p-value: **1.0**. All 24 independent verdicts agreed with the original team subprocess grader.

| Task | Baseline | Steering | Note |
|---|---|---|---|
| HumanEval/143 | fail | pass | Baseline truncated; steered solution completed |
| HumanEval/126 | fail | fail | Both truncated after reasoning about duplicate conditions |
| HumanEval/154 | pass | fail | Steered response entered a repetitive loop and truncated |

**Observed pattern:** every incorrect output in both groups hit the 4,096-token cap, and every completed output passed. This is descriptive, not proof that the steering direction acts on reasoning completion or that all failures are solely caused by truncation.

## Interpretation and limitations

- The result provides **no evidence of a net coding-accuracy gain** on this small held-out subset; it cannot establish equivalence or absence of any population effect.
- This was a **12-task subset of the team's own independent test partition**, not the original standalone evaluator's 20-development/40-held-out split.
- The original experiment did not record immutable model/tokenizer revisions, a generation RNG seed, or top-p. These are preserved as **unknown** rather than invented.
- The existing host validator requires those missing fields; the audit reused its original Docker worker and result-normalization routines directly. **The full strict-provenance CLI was not used unchanged.**
- The team's report contains random-direction and ablation controls, but **this independent rerun only graded baseline and positive probe steering**; do not present the controls as independently reproduced.
- Predictive probe performance does not itself establish a causal reasoning mechanism. Human-readable traces are useful exploratory evidence, not a substitute for mechanistic causal tests.

## Next experiment

Pre-register the next intervention using development tasks; assess it on a *fresh* test set rather than optimizing on these 12 held-out examples. Include multiple magnitude-matched random directions, consistent batch composition, model/tokenizer commit IDs, response truncation/termination metrics, and correctness. Compare paired changes rather than only aggregate pass rate.

> **Short conclusion:** Independent official EvalPlus grading confirmed no net HumanEval+ improvement from layer-29 probe steering: both conditions solved 10/12 problems, with one fix and one regression. Both changes coincided with whether the model completed its output within the 4,096-token budget.
