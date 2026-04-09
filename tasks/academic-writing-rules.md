# Academic Writing Rules

These rules apply to ALL paper drafts, manuscripts, and LaTeX documents across all projects.

## Style: Analytic and Declarative Only

- Use analytic or declarative style exclusively. Never use narrative-illustrative or narraustrative vague writing.
- State what happened, what was measured, and what it demonstrates. Use declarative sentences with concrete subjects and verbs.
- Violations include:
  - "one primitive became three, then six" (narrative storytelling)
  - "This is where the agent's limitations became clear" (scene-setting)
  - "The strongest evidence came in the final session" (narrative framing)
  - "Early in this project, the agent produced 'cool demos'" (anecdotal)
  - "The threshold we crossed" (narrative milestone language)

## Structure

- Each section follows: statement, supporting evidence, conclusion.
- Transitions between sections are logical, not clever.
- Section titles are descriptive and plain. No wordplay, no parenthetical asides (e.g., not "Worked, Then Didn't, Then Did").

## Low-Information Descriptors

- Never use "(failed)" or "(succeeded)" as standalone descriptors. Always explain the specific outcome, then the causal mechanism. E.g., not "embedded LLM (failed)" but "embedded LLM achieved 0% success on multi-step tasks because the model could not resolve contextual constraints from tool schemas."

## No AI-Slop Patterns

- No "What we learned:", "The key insight:", "Why this matters:" as standalone headings or italic asides. Conclusions are stated as objective scientific observations integrated into the paragraph.
- No informal language in captions or titles.

## Formatting

- Never use emdashes (---). Use commas, semicolons, colons, or restructure the sentence.
- Never use sentence fragment annotations like "_Correction:_ the user did X." Explain resolutions in full prose: how the correction was delivered (descriptive prompt, few-shot examples, corrected design file) and what was retained.
- Figure captions with subfigures must reference the labels: "(A) description... (B) description..."
- Subfigure labels use uppercase: (A), (B), (C). Match the case used in the actual figure images.

## Colon Usage

- Do not overuse colons. Excessive colons read as AI-generated. Use periods, conjunctions, or restructure the sentence instead. Only use colons where absolutely necessary (e.g., introducing a formal list title, or where no other punctuation works). When in doubt, use a period and start a new sentence.

## Figure Citations

- Every figure and subfigure must be explicitly cited in the body text. No figure should appear without a corresponding textual reference. This is standard academic paper writing practice.

## Voice

- Direct, objective, scientific.
- First person plural ("we") is acceptable for describing our own work.
- No hedging, no filler, no rhetorical flourish.

## Entity Chronology

- No concept should be referenced before it is introduced. Trace all entities through the paper and verify that first mention occurs at or after the point where the concept is defined or described.

## Citations

- Cite where observations agree with existing literature.
- Do not overclaim. State what was observed and what it demonstrates; do not extrapolate beyond the evidence.
