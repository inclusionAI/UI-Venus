# Judge protocols and language

## PerceptionBench

The recorded evaluation uses **self-hosted Deepseekv4-flash-0731**. Repeated LLM
judgments may vary; retain judge configuration and failure counts when comparing
runs. The runtime templates are adapted from
[MoonshotAI/PerceptionBench](https://github.com/MoonshotAI/PerceptionBench), including
its [teacher-grading prompt](https://github.com/MoonshotAI/PerceptionBench/blob/master/eval/judge_prompt.txt).
The zoom-in variant grades the extracted final answer while ignoring crop metadata
and intermediate reasoning. Attribution is recorded separately in [NOTICE](../NOTICE).

## HR-Bench

Report **strict option matching**, averaged over four option rotations, separately
for 4K and 8K: `splits.HRBench4K.strict_accuracy` and
`splits.HRBench8K.strict_accuracy` in the combined report. Optional LLM judgments
and primary-metric fields remain available in full reports but are not the
publication headline.

## Training CoT reward: Chinese runtime prompt, English explanation

The optional CoT reward uses the Chinese prompt in
[zoomin_grounding_plugin.py](../agentic_gui_grounding_rl/plugins/grounding_zoom_in/zoomin_grounding_plugin.py).
The following explanation is documentation only and is **not sent to the judge**.

The judge sees task text, reasoning, actions, and crop feedback, without images.
It assesses reasoning quality rather than solving the task: focus on the target,
useful observation choices, use of feedback, thought/action consistency, and
convergence without redundant actions. Oracle mode additionally checks alignment
with the reference target or infeasibility label.

Scoring anchors are 0.9–1.0 for focused, coherent, feedback-driven reasoning;
0.6–0.8 for sound reasoning with minor issues; 0.3–0.5 for ignored feedback,
inconsistent actions, repetition, or loss of focus; and 0.0–0.2 for little useful
reasoning. The prompt requests a Chinese assessment of at most 80 characters,
followed by `Score: 0.xx`. Translating the runtime prompt changes the reward input.
