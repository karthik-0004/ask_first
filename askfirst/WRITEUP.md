# One-Page Writeup: AskFirst Clary Pattern Detector

## 1. Approach to the Reasoning Problem

The core challenge in this problem is that user health history is not a clean table of facts. It is messy, conversational, and spread across timestamps where the user may never explicitly connect related events. A naive keyword approach fails quickly because isolated mentions like "hair fall" in a late session are often meaningless unless linked to earlier behavior changes, such as a restrictive diet introduced weeks prior.

The system addresses this by assembling each user history into a single structured prompt with profile context plus chronologically ordered sessions. Temporal ordering is enforced so direction of change is preserved: what happened first, what changed later, and what symptom emerged afterward. A compact timeline index is placed before the full context to give the model an anchor for sequence-aware reasoning before it reads the detailed conversation blocks.

Prompt design is intentionally strict. The model is told to report explicit cause-effect links, include session IDs and timestamps as evidence, and describe the temporal gap between events. It is also constrained to only report patterns supported by at least two sessions, which discourages one-off speculation and pushes the output toward traceable evidence chains instead of surface-level associations.

Confidence scoring is a practical safeguard for downstream use. A pattern observed across behavior reversals and repeated outcomes should rank higher than a weak two-point narrative fit. By requiring confidence justification tied to session-level evidence, the system forces the model to externalize why it trusts a pattern rather than only labeling it high or low confidence.

## 2. Where the System Fails or Hallucinates Confidently

The biggest failure mode is coherent-sounding narrative construction. GPT-4o can take two unrelated events separated by time and still generate a plausible biological story that reads convincingly. Even with strict prompts and confidence fields, this can still produce false positives when temporal proximity is mistaken for causality.

There is also a hard context tradeoff. If a user has 50 or more sessions, truncating to the most recent 10 can drop the true upstream trigger that explains a current symptom. The timeline summary helps with orientation, but it does not replace selective retrieval over deeper history.

The system has no independent medical truth-checking. It depends on model priors and prompt constraints, not verified clinical validation. That means edge cases, uncommon interactions, or outdated assumptions may still pass through as well-structured but incorrect insights.

Finally, the current architecture is single-call. A better approach would decompose reasoning into stages: identify symptom clusters, retrieve relevant historical evidence per cluster, test causal hypotheses, and then score confidence. Narrower scoped reasoning steps would reduce hallucination pressure compared with one large end-to-end generation.

## 3. What Would Be Built Differently With More Time

- Retrieval-augmented context: embed each session, then for each detected symptom retrieve the 5 most semantically relevant past sessions rather than sending everything at once.
- Multi-step reasoning pipeline: step 1 extract symptom timeline, step 2 hypothesize causes, step 3 verify hypothesis against session evidence, step 4 score, each as a separate LLM call with structured output.
- Human-in-the-loop verification: show the pattern to a health professional for sign-off before displaying to user.
- Longitudinal database: store patterns across sessions persistently so new sessions can confirm or deny existing hypotheses automatically.

The system finds the 8 planted patterns reliably on the synthetic dataset because the dataset was designed to have clear signals. On real-world messy data with noise, ambiguity, and missing context, the false positive rate would be significantly higher and the temporal reasoning would require validation.
