"""Zoom-In textual crop-image protocol for ms-swift GRPO: five-component reward.

Multi-turn crop-policy GRPO plugin: the model zooms into UI screenshots with
textual ``crop_image`` actions and answers with a normalized coordinate.
This variant registers a five-component reward:

* WRONG final answer  -> the gated process reward is exactly 0 (no
  "crop well, answer wrong" credit, no credit for lucky wide-area search);
* RIGHT final answer  -> a process score (best crop IoU, repeat / no-improve
  penalties) minus ``ZOOMIN_CALL_COST`` (default 0.05) per successful
  crop_image call, so every extra turn must be paid for with IoU improvement;
  one well-placed crop outscores a longer search chain, and a direct correct
  answer (no crop) keeps its full accuracy credit without any process
  component;
* ``zoomin_zoom_bonus``: correct answers earned through effective zooming on
  hard samples still pay an extra bonus (outcome-shaped);
* ``zoomin_cot_rm`` (AsyncORM) flattens each assistant turn's ``Think``
  reasoning + tool-call decision + crop feedback into a text-only transcript
  and asks an external OpenAI-compatible judge endpoint for a 0~1
  reasoning-quality score, gated by accuracy so verbose-but-wrong reasoning
  cannot be farmed. The judge is called through the standard OpenAI Python
  SDK (``pip install openai``): point ``ZOOMIN_COT_RM_API_BASE`` at any
  OpenAI-compatible endpoint (e.g. ``https://api.deepseek.com/v1`` with
  ``ZOOMIN_COT_RM_MODEL=deepseek-chat``, or a local vLLM server).

The shaping components are complementary: the gated process reward removes
credit from process gaming and prices tool calls; the zoom bonus keeps paying
for the *outcome* of good zooming on hard samples; the CoT judge pays for the
*quality of the reasoning and tool-call decisions* themselves; PRM / entropy
turn filtering still masks bad decision turns on the loss side.

Registered names:

* scheduler: ``zoomin_grounding_scheduler``
* rewards: ``zoomin_accuracy``, ``zoomin_crop_process_gated``,
  ``zoomin_format``, ``zoomin_zoom_bonus``, ``zoomin_cot_rm``.

Process-reward knobs (``ZOOMIN_REPEAT_*`` / ``ZOOMIN_NO_IMPROVE_PENALTY`` /
PRM/entropy filters / ``ZOOMIN_ZOOM_BONUS*``) and the judge's
``ZOOMIN_COT_RM_*`` configuration are all documented in the plugin README.
"""

from __future__ import annotations

import asyncio
import math
import numbers
import os
import secrets
import statistics
import textwrap
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from swift.rewards.orm import AsyncORM, ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns

from zoomin_grounding_protocol import (
    DEFAULT_CALL_COST,
    DEFAULT_COT_MAX_ACTION_CHARS,
    DEFAULT_COT_MAX_SEGMENT_CHARS,
    DEFAULT_COT_MAX_TOTAL_CHARS,
    DEFAULT_CROP_MAX_PIXELS,
    DEFAULT_CROP_UPSCALE_FACTOR,
    DEFAULT_MAX_CROP_ASPECT_RATIO,
    DEFAULT_MAX_REPEAT_PENALTY,
    DEFAULT_NO_IMPROVE_PENALTY,
    DEFAULT_PRM_IOU_WEIGHT,
    DEFAULT_REPEAT_IOU_THRESH,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_ZOOM_BONUS,
    DEFAULT_ZOOM_BONUS_HARD_AREA_RATIO,
    DEFAULT_ZOOM_BONUS_MIN_QUALITY,
    INVALID_ACTION_CORRECTION,
    MAX_CROP_IMAGE_CALLS,
    accuracy_reward,
    assistant_visible_content,
    build_cot_transcript,
    build_system_prompt,
    crop_results_user_text,
    execute_crop,
    extract_cot_score,
    extract_instruction,
    extract_task_from_system_prompt,
    format_reward,
    gated_crop_process_reward,
    has_think_blocks,
    json_safe_value,
    message_content_text,
    parse_text_action,
    point_in_bbox_accuracy,
    registered_path,
    safe_task_component,
    turn_prm_scores,
    zoom_bonus_reward,
)


def _has_non_finite(value: Any) -> bool:
    """Detect NaN/Inf floats nested in JSON-like structures.

    ``numbers.Real`` also covers ``numpy.float64``/``numpy.float32``: the
    latter is not a ``float`` subclass but is equally rejected by JSON
    serialization. ``int`` subclasses (including ``bool``) are always finite.
    """
    if isinstance(value, numbers.Real):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_non_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_non_finite(item) for item in value)
    return False


def _mask_non_finite(value: Any) -> Any:
    """Replace NaN/Inf floats with None so ``json.dumps(allow_nan=False)`` succeeds."""
    if isinstance(value, numbers.Real):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _mask_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mask_non_finite(item) for item in value]
    return value


def _safe_json_value(value: Any) -> Any:
    """``json_safe_value`` that never raises: mask non-finite floats instead.

    ``json_safe_value`` raises ``ValueError`` on NaN/Inf. Raising inside
    ``step()`` would abort the whole rollout trajectory (the base scheduler
    does not catch step exceptions), so degrade to null placeholders and let
    the failed crop flow return the correction prompt instead.
    """
    try:
        return json_safe_value(value)
    except ValueError:
        return _mask_non_finite(value)


class ZoomInAccuracy(ORM):
    """Pure point-in-bbox accuracy, identical to the VERL reward."""

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        del kwargs
        return [accuracy_reward(completion, ground_truth)
                for completion, ground_truth in zip(completions, solution)]


orms['zoomin_accuracy'] = ZoomInAccuracy


class ZoomInCropProcessGated(ORM):
    """Gated process reward: outcome-gated crop scoring + per-call tool cost.

    Two changes over the ungated process reward:

    1. **outcome gate** — trajectories whose final answer is NOT inside the GT
       bbox (same criterion as ``zoomin_accuracy``) receive exactly 0,
       regardless of how well their crops overlapped the GT. Process credit no
       longer decouples from the result.
    2. **call cost** — for correct trajectories, each successful crop_image
       call charges ``ZOOMIN_CALL_COST`` (default 0.05). An extra crop is
       only worth it if it raises the best-IoU base score, which removes the
       incentive to pad turns with exploratory crops on easy samples.

    Correct no-crop trajectories and refusal samples score 0 here (neutral);
    their credit comes from the accuracy component alone (plus the zoom
    bonus when a correct zooming trajectory qualifies).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeat_iou_thresh = float(
            os.environ.get('ZOOMIN_REPEAT_IOU_THRESH', str(DEFAULT_REPEAT_IOU_THRESH)))
        self.repeat_penalty = float(os.environ.get('ZOOMIN_REPEAT_PENALTY', str(DEFAULT_REPEAT_PENALTY)))
        self.no_improve_penalty = float(
            os.environ.get('ZOOMIN_NO_IMPROVE_PENALTY', str(DEFAULT_NO_IMPROVE_PENALTY)))
        self.max_repeat_penalty = float(
            os.environ.get('ZOOMIN_MAX_REPEAT_PENALTY', str(DEFAULT_MAX_REPEAT_PENALTY)))
        self.call_cost = float(os.environ.get('ZOOMIN_CALL_COST', str(DEFAULT_CALL_COST)))

    def __call__(self, completions, solution, rollout_infos=None, **kwargs) -> List[float]:
        del kwargs
        if rollout_infos is None:
            rollout_infos = [{} for _ in solution]
        rewards = []
        for completion, ground_truth, info in zip(completions, solution, rollout_infos):
            # outcome gate: identical criterion to zoomin_accuracy, so the
            # process component activates exactly when accuracy scores 1.0
            if accuracy_reward(completion, ground_truth) < 1.0:
                rewards.append(0.0)
                continue
            tool_events = info.get('tool_events') if isinstance(info, dict) else None
            rewards.append(
                gated_crop_process_reward(
                    tool_events,
                    ground_truth,
                    repeat_iou_thresh=self.repeat_iou_thresh,
                    repeat_penalty=self.repeat_penalty,
                    no_improve_penalty=self.no_improve_penalty,
                    max_repeat_penalty=self.max_repeat_penalty,
                    call_cost=self.call_cost,
                ))
        return rewards


orms['zoomin_crop_process_gated'] = ZoomInCropProcessGated


class ZoomInFormat(ORM):
    """Format reward: canonical protocol compliance of the last assistant turn.

    1.0 for the strict parse path (coordinate after ``</think>`` or a clean
    ``<action>`` block), 0.5 for fallback-only parses, 0.0 for unparseable output.
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        del kwargs
        return [format_reward(completion) for completion in completions]


orms['zoomin_format'] = ZoomInFormat


class ZoomInZoomBonus(ORM):
    """Zoom-in bonus: correct answers earned THROUGH effective zooming are
    worth more than answers produced directly.

    Pays a fixed bonus only when the trajectory (1) ends with a correct final
    point, (2) belongs to a hard sample (small GT bbox, unless disabled), and
    (3) contains at least one effective crop whose heuristic PRM score clears
    ``min_quality`` — so cropping an irrelevant region cannot farm the bonus.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bonus = float(os.environ.get('ZOOMIN_ZOOM_BONUS', str(DEFAULT_ZOOM_BONUS)))
        self.min_quality = float(
            os.environ.get('ZOOMIN_ZOOM_BONUS_MIN_QUALITY', str(DEFAULT_ZOOM_BONUS_MIN_QUALITY)))
        self.hard_area_ratio = float(
            os.environ.get('ZOOMIN_ZOOM_BONUS_HARD_AREA_RATIO', str(DEFAULT_ZOOM_BONUS_HARD_AREA_RATIO)))
        self.hard_only = os.environ.get('ZOOMIN_ZOOM_BONUS_HARD_ONLY', '1') not in ('0', 'false', 'False')
        self.iou_weight = float(os.environ.get('ZOOMIN_PRM_IOU_WEIGHT', str(DEFAULT_PRM_IOU_WEIGHT)))
        if self.bonus < 0:
            raise ValueError('ZOOMIN_ZOOM_BONUS must be nonnegative.')
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError('ZOOMIN_ZOOM_BONUS_MIN_QUALITY must be in [0, 1].')
        if not 0.0 < self.hard_area_ratio <= 1.0:
            raise ValueError('ZOOMIN_ZOOM_BONUS_HARD_AREA_RATIO must be in (0, 1].')

    def __call__(self, completions, solution, rollout_infos=None, **kwargs) -> List[float]:
        del kwargs
        if rollout_infos is None:
            rollout_infos = [{} for _ in solution]
        rewards = []
        for completion, ground_truth, info in zip(completions, solution, rollout_infos):
            tool_events = info.get('tool_events') if isinstance(info, dict) else None
            final_point = info.get('prediction_relative_0_1000') if isinstance(info, dict) else None
            rewards.append(
                zoom_bonus_reward(
                    completion,
                    tool_events,
                    ground_truth,
                    bonus=self.bonus,
                    min_quality=self.min_quality,
                    hard_area_ratio=self.hard_area_ratio,
                    hard_only=self.hard_only,
                    iou_weight=self.iou_weight,
                    final_point=final_point,
                ))
        return rewards


orms['zoomin_zoom_bonus'] = ZoomInZoomBonus


JUDGE_SYSTEM_PROMPT = textwrap.dedent("""
    你是一个 GUI grounding agent 的推理过程质量评审员。该 agent 的任务是在一张 UI 截图
    中定位任务指令对应目标的中心点；它可以先多次裁剪图片放大观察（zoom-in），最后输出
    归一化到 [0,1000] 的坐标，任务不可行时输出 [-1.00,-1.00]。

    你只能看到文本：任务指令、agent 每一轮的思考（think）、动作决策（action）以及
    裁剪动作的执行结果（feedback）。你无法看到图片，也不需要自己解决任务——只评估
    思考过程的质量。

    评估维度：
    {criteria}

    打分锚点：
    - 0.9~1.0：思考始终围绕定位目标展开，充分利用每轮裁剪反馈，思考与动作参数一致，
      逐步收敛，简洁无冗余；
    - 0.6~0.8：思考总体合理，存在不影响主线的小瑕疵（如轻微的坐标表述不精确）；
    - 0.3~0.5：存在明显问题——忽略工具反馈、思考与动作不一致、重复同一无效操作、
      推理与任务目标脱节之一；
    - 0.0~0.2：几乎无有效推理、内容大量复读、或与任务完全无关。

    先给出不超过 80 字的简短评价，然后另起一行单独输出分数，格式必须为：
    Score: 0.xx
    （0.0~1.0 的浮点数，保留两位小数）
""").strip()

ORACLE_CRITERION = (
    '6. 标注对齐：对照参考标注（GT 区域或不可行标记），裁剪区域与坐标方向是否在逐步'
    '逼近正确位置；refusal 样本则评估"任务不可行"的判断依据是否充分。'
)

BLIND_CRITERIA = textwrap.dedent("""
    1. 目标导向：每一轮思考是否围绕"如何定位目标任务目标"展开；
    2. 反馈利用：后续思考是否利用了之前的裁剪执行结果（成功区域范围 / 失败原因）；
    3. 言行一致：思考中描述的区域、方向与 action 中的坐标参数是否一致；
    4. 增量推进：每一轮是否有新信息或新决策，没有重复相同的裁剪或推理；
    5. 简洁性：没有无意义的填充与复读。
""").strip()

ORACLE_CRITERIA = f'{BLIND_CRITERIA}\n{ORACLE_CRITERION}'


class ZoomInCotRM(AsyncORM):
    """CoT process reward: LLM-as-judge over the trajectory's thinking chain.

    One judge call per trajectory. The transcript keeps only assistant turns'
    ``<think>`` text + structured action summary + text-only crop feedback
    (images/base64 dropped), so a text-only judge endpoint is sufficient.

    Anti-hacking: the judge score is multipled by the binary accuracy reward
    (``ZOOMIN_COT_RM_GATE_ACCURACY=1`` by default) — wrong answers earn no
    CoT credit, so verbose-but-wrong reasoning cannot be farmed. Trajectories
    without any ``<think>`` block skip the judge call and score 0.

    Endpoint configuration (env vars, read at trainer init on all ranks). The
    judge talks to ANY OpenAI-compatible chat-completions endpoint through the
    official OpenAI Python SDK (``pip install openai``):

    * ``ZOOMIN_COT_RM_API_BASE``   OpenAI-compatible base URL, required to
      enable the judge (e.g. ``https://api.deepseek.com/v1`` with
      ``ZOOMIN_COT_RM_MODEL=deepseek-chat``, or ``http://host:8000/v1`` for a
      local vLLM server). Empty by default: the judge stays disabled and every
      trajectory receives the neutral score.
    * ``ZOOMIN_COT_RM_API_KEY``    API key / bearer token (leave empty for
      unauthenticated endpoints; a non-empty placeholder is sent instead)
    * ``ZOOMIN_COT_RM_API_BASE_2`` / ``ZOOMIN_COT_RM_API_KEY_2``
      optional second endpoint; samples are randomly split 50/50 across
      endpoints, with automatic failover to the other endpoint on failure
    * ``ZOOMIN_COT_RM_MODEL``      judge model name
    * ``ZOOMIN_COT_RM_CONCURRENCY`` max parallel judge calls per rank
    * ``ZOOMIN_COT_RM_TIMEOUT``    per-request timeout (seconds)
    * ``ZOOMIN_COT_RM_RETRIES``    retries per sample after failure
    * ``ZOOMIN_COT_RM_MAX_TOKENS`` judge decoding budget
    * ``ZOOMIN_COT_RM_TEMPERATURE`` judge sampling temperature (0 = deterministic)
    * ``ZOOMIN_COT_RM_NEUTRAL``    score returned on judge failure
    * ``ZOOMIN_COT_RM_GATE_ACCURACY`` 1 = multiply by correctness (default)
    * ``ZOOMIN_COT_RM_ORACLE``     1 = also give the judge the GT annotation
    * ``ZOOMIN_COT_RM_MAX_SEGMENT_CHARS`` / ``ZOOMIN_COT_RM_MAX_ACTION_CHARS`` /
      ``ZOOMIN_COT_RM_MAX_TOTAL_CHARS`` transcript truncation budgets
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        # --- endpoint pool: primary + optional secondary ---
        # Any OpenAI-compatible endpoint works; nothing internal is baked in.
        # An empty base disables the judge (all scores become the neutral
        # fallback) so training is never blocked on an external service.
        self._endpoints = []  # list of (api_base, api_key)
        primary_base = os.environ.get('ZOOMIN_COT_RM_API_BASE', '').rstrip('/')
        primary_key = os.environ.get('ZOOMIN_COT_RM_API_KEY', '')
        if primary_base:
            self._endpoints.append((primary_base, primary_key))
        secondary_base = os.environ.get('ZOOMIN_COT_RM_API_BASE_2', '').rstrip('/')
        secondary_key = os.environ.get('ZOOMIN_COT_RM_API_KEY_2', '')
        if secondary_base:
            self._endpoints.append((secondary_base, secondary_key))
        # keep legacy attrs for backward compat (logging, warnings)
        self.api_base = primary_base
        self.api_key = primary_key
        self.model = os.environ.get('ZOOMIN_COT_RM_MODEL', 'deepseek-chat')
        self.concurrency = max(1, int(os.environ.get('ZOOMIN_COT_RM_CONCURRENCY', '8')))
        self.timeout = float(os.environ.get('ZOOMIN_COT_RM_TIMEOUT', '30'))
        self.max_retries = max(0, int(os.environ.get('ZOOMIN_COT_RM_RETRIES', '1')))
        self.request_max_tokens = int(os.environ.get('ZOOMIN_COT_RM_MAX_TOKENS', '1024'))
        self.temperature = float(os.environ.get('ZOOMIN_COT_RM_TEMPERATURE', '0.0'))
        self.neutral_score = min(1.0, max(0.0, float(os.environ.get('ZOOMIN_COT_RM_NEUTRAL', '0.5'))))
        self.gate_accuracy = os.environ.get('ZOOMIN_COT_RM_GATE_ACCURACY', '1') not in ('0', 'false', 'False')
        self.oracle = os.environ.get('ZOOMIN_COT_RM_ORACLE', '0') in ('1', 'true', 'True')
        self.max_segment_chars = int(
            os.environ.get('ZOOMIN_COT_RM_MAX_SEGMENT_CHARS', str(DEFAULT_COT_MAX_SEGMENT_CHARS)))
        self.max_action_chars = int(
            os.environ.get('ZOOMIN_COT_RM_MAX_ACTION_CHARS', str(DEFAULT_COT_MAX_ACTION_CHARS)))
        self.max_total_chars = int(
            os.environ.get('ZOOMIN_COT_RM_MAX_TOTAL_CHARS', str(DEFAULT_COT_MAX_TOTAL_CHARS)))
        criteria = ORACLE_CRITERIA if self.oracle else BLIND_CRITERIA
        self.system_prompt = JUDGE_SYSTEM_PROMPT.format(criteria=criteria)
        # observability counters; a single warning is emitted once per rank
        # when the cumulative judge failure rate crosses 10%.
        self._n_scored = 0
        self._n_failed = 0
        self._failure_warned = False
        # One OpenAI SDK client per endpoint. The SDK requires a non-empty
        # api_key even for unauthenticated endpoints, so a 'EMPTY' placeholder
        # is sent when no key is configured. Retry/timeout policy is managed
        # here (SDK max_retries=0) so cross-endpoint failover keeps full
        # control of the retry and failure accounting.
        self._clients = []
        if self._endpoints:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                warnings.warn('zoomin_cot_rm: the `openai` package is required for '
                              'judge calls (pip install openai); neutral scores are '
                              'returned until it is installed.')
            else:
                self._clients = [
                    AsyncOpenAI(base_url=base, api_key=key or 'EMPTY',
                                timeout=self.timeout, max_retries=0)
                    for base, key in self._endpoints
                ]
        if not self._endpoints:
            warnings.warn('zoomin_cot_rm: ZOOMIN_COT_RM_API_BASE is empty; the '
                          'judge is disabled and every trajectory receives the '
                          f'neutral score ({self.neutral_score}).')

    def _build_judge_messages(self, task, transcript, ground_truth):
        sections = [
            f'任务指令: {task}',
            '',
            'agent 的完整轨迹（think=思考, action=动作决策, feedback=裁剪执行结果）:',
            transcript,
        ]
        if self.oracle:
            sections += [
                '',
                ('参考标注（[0,1000] 归一化坐标下的 GT bbox；'
                 f'[-1,-1,-1,-1] 表示任务不可行）: {ground_truth}'),
            ]
        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': '\n'.join(sections)},
        ]

    async def _score_one(self, semaphore, task, transcript, ground_truth,
                         endpoint_idx=0):
        """Single judge call with timeout + bounded retries + cross-endpoint failover.

        Tries the assigned endpoint first; on exhaustion of retries, fails over
        to the other endpoint (if configured).  Returns neutral score only when
        all endpoints are exhausted.
        """
        messages = self._build_judge_messages(task, transcript, ground_truth)
        # build ordered endpoint list: assigned first, then the rest
        indices = list(range(len(self._endpoints)))
        if endpoint_idx < len(indices):
            indices.remove(endpoint_idx)
            indices.insert(0, endpoint_idx)
        async with semaphore:
            for ep_idx in indices:
                client = self._clients[ep_idx]
                for attempt in range(self.max_retries + 1):
                    if attempt > 0:
                        await asyncio.sleep(min(2 ** attempt, 8))
                    try:
                        response = await client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=self.request_max_tokens,
                        )
                        content = response.choices[0].message.content or ''
                        score = extract_cot_score(content)
                        if score is not None:
                            return score
                    except Exception:
                        continue
        self._n_failed += 1
        return self.neutral_score

    def _maybe_warn_failure_rate(self):
        if self._failure_warned or self._n_scored == 0:
            return
        if self._n_failed / self._n_scored > 0.1:
            self._failure_warned = True
            endpoints_str = ', '.join(base for base, _ in self._endpoints)
            warnings.warn(
                f'ZoomInCotRM: {self._n_failed}/{self._n_scored} cumulative judge calls '
                f'failed (neutral={self.neutral_score}). Check endpoints: {endpoints_str}')

    async def __call__(self, completions, solution, messages=None, rollout_infos=None,
                       **kwargs) -> List[float]:
        del kwargs
        n = len(completions)
        if messages is None:
            messages = [None] * n
        if rollout_infos is None:
            rollout_infos = [None] * n

        judge_inputs = []
        skip_flags = []
        for sample_messages, sample_infos in zip(messages, rollout_infos):
            task = ''
            if sample_messages and isinstance(sample_messages[0], dict) \
                    and sample_messages[0].get('role') == 'system':
                task = extract_task_from_system_prompt(
                    message_content_text(sample_messages[0].get('content', '')))[:400]
            tool_events = sample_infos.get('tool_events') if isinstance(sample_infos, dict) else None
            transcript = build_cot_transcript(
                sample_messages or [],
                tool_events,
                max_segment_chars=self.max_segment_chars,
                max_action_chars=self.max_action_chars,
                max_total_chars=self.max_total_chars,
            )
            # no visible reasoning anywhere -> nothing for the judge to score
            skip_flags.append(not has_think_blocks(sample_messages or []) or not transcript.strip())
            judge_inputs.append((task, transcript))

        n_endpoints = len(self._endpoints)
        if n_endpoints == 0:
            # judge disabled (no endpoint configured): neutral score for every
            # non-skipped trajectory; training continues with the other rewards.
            raw_scores = [0.0 if skip else self.neutral_score for skip in skip_flags]
        else:
            import random
            semaphore = asyncio.Semaphore(self.concurrency)
            coros = [
                self._score_one(semaphore, task, transcript, ground_truth,
                                endpoint_idx=random.randrange(n_endpoints))
                if not skip else self._noop_zero()
                for skip, (task, transcript), ground_truth in zip(skip_flags, judge_inputs, solution)
            ]
            raw_scores = list(await asyncio.gather(*coros))
            self._n_scored += sum(1 for skip in skip_flags if not skip)
        self._maybe_warn_failure_rate()

        if self.gate_accuracy:
            raw_scores = [
                score * accuracy_reward(completion, ground_truth)
                for score, completion, ground_truth in zip(raw_scores, completions, solution)
            ]
        return raw_scores

    @staticmethod
    async def _noop_zero():
        return 0.0


orms['zoomin_cot_rm'] = ZoomInCotRM


class ZoomInGroundingScheduler(MultiTurnScheduler):
    """Run Zoom-In assistant-text -> user-crop-image trajectories."""

    _STATE_KEY = '_zoomin_grounding_state'

    def __init__(self, infer_engine=None, max_turns=None, *args, **kwargs):
        super().__init__(infer_engine, max_turns, *args, **kwargs)
        requested_max_tool_calls = int(
            os.environ.get('ZOOMIN_MAX_TOOL_CALLS', str(MAX_CROP_IMAGE_CALLS)))
        self.max_action_corrections = int(os.environ.get('ZOOMIN_MAX_ACTION_CORRECTIONS', '2'))
        self.max_completion_tokens = int(os.environ.get('ZOOMIN_MAX_COMPLETION_TOKENS', '2000'))
        self.crop_upscale_factor = float(
            os.environ.get('ZOOMIN_CROP_UPSCALE_FACTOR', str(DEFAULT_CROP_UPSCALE_FACTOR)))
        self.crop_max_pixels = int(
            os.environ.get('ZOOMIN_CROP_MAX_PIXELS', str(DEFAULT_CROP_MAX_PIXELS)))
        self.max_crop_aspect_ratio = float(
            os.environ.get('ZOOMIN_MAX_CROP_ASPECT_RATIO', str(DEFAULT_MAX_CROP_ASPECT_RATIO)))
        self.crop_root = Path(os.environ.get('ZOOMIN_CROP_ROOT', '~/.cache/zoomin_crops')).expanduser().resolve()

        # ---- turn-level filtering (PRM quality + per-turn uncertainty) ----
        self.prm_filter = os.environ.get('ZOOMIN_PRM_FILTER', '1') not in ('0', 'false', 'False')
        self.prm_filter_mode = os.environ.get('ZOOMIN_PRM_FILTER_MODE', 'success_only').lower()
        self.prm_min_quality = float(os.environ.get('ZOOMIN_PRM_MIN_QUALITY', '0.05'))
        self.prm_iou_weight = float(os.environ.get('ZOOMIN_PRM_IOU_WEIGHT', str(DEFAULT_PRM_IOU_WEIGHT)))
        self.entropy_filter = os.environ.get('ZOOMIN_ENTROPY_FILTER', '1') not in ('0', 'false', 'False')
        self.entropy_ratio = float(os.environ.get('ZOOMIN_ENTROPY_RATIO', '1.0'))
        if self.prm_filter_mode not in ('success_only', 'always'):
            raise ValueError("ZOOMIN_PRM_FILTER_MODE must be 'success_only' or 'always'.")
        if not 0.0 <= self.prm_min_quality <= 1.0:
            raise ValueError('ZOOMIN_PRM_MIN_QUALITY must be in [0, 1].')
        if not 0.0 <= self.prm_iou_weight <= 1.0:
            raise ValueError('ZOOMIN_PRM_IOU_WEIGHT must be in [0, 1].')
        if self.entropy_ratio <= 0:
            raise ValueError('ZOOMIN_ENTROPY_RATIO must be positive.')

        if requested_max_tool_calls < 0 or self.max_action_corrections < 0:
            raise ValueError('Zoom-In tool and correction limits must be nonnegative.')
        self.max_tool_calls = min(requested_max_tool_calls, MAX_CROP_IMAGE_CALLS)
        if self.max_completion_tokens < 1:
            raise ValueError('ZOOMIN_MAX_COMPLETION_TOKENS must be positive.')
        if self.crop_upscale_factor <= 0 or self.crop_max_pixels < 1:
            raise ValueError('Zoom-In crop resize settings must be positive.')

        required_turns = self.max_tool_calls + self.max_action_corrections + 1
        if self.max_turns is not None and self.max_turns < required_turns:
            warnings.warn(
                f'max_turns ({self.max_turns}) is smaller than the Zoom-In trajectory '
                f'budget ({required_turns}); bumping max_turns to {required_turns}.')
            self.max_turns = required_turns

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content or '')
        return ''.join(
            str(item.get('text', ''))
            for item in content
            if isinstance(item, dict) and item.get('type') == 'text')

    @classmethod
    def _problem_from_request(cls, infer_request) -> str:
        problem = infer_request.data_dict.get('problem')
        if problem:
            return str(problem)
        for message in infer_request.messages:
            if message.get('role') == 'user':
                return cls._message_text(message.get('content'))
        return ''

    @staticmethod
    def _original_image_path(infer_request) -> str:
        if not infer_request.images:
            raise ValueError('Grounding sample does not contain an image path.')
        value = infer_request.images[0]
        if isinstance(value, dict):
            value = value.get('path') or value.get('image') or value.get('url')
        if not isinstance(value, (str, os.PathLike)):
            raise ValueError('Zoom-In rollout requires the original image as a filesystem path.')
        path = registered_path(os.fspath(value))
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        return path

    def _initialize_request(self, infer_request) -> None:
        if self._STATE_KEY in infer_request.data_dict:
            return

        problem = self._problem_from_request(infer_request)
        instruction = extract_instruction(problem)
        original_path = self._original_image_path(infer_request)
        with Image.open(original_path) as image:
            width, height = image.size

        sample_index = infer_request.data_dict.get('index', infer_request.data_dict.get('__index__', 0))
        task_id = f'{sample_index}_{Path(original_path).stem}'
        conversation_id = secrets.token_hex(8)
        crop_dir = self.crop_root / safe_task_component(task_id) / f'conversation_{conversation_id}'
        system_prompt = build_system_prompt(
            instruction,
            original_path,
            width,
            height,
            max_tool_calls=self.max_tool_calls,
        )

        # This is the string-form equivalent of Zoom-In' text-then-image user content.
        # Keeping image paths in ``infer_request.images`` makes the trajectory directly
        # serializable through Swift's rollout-server POST API.
        infer_request.messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'image_path: {original_path}<image>'},
        ]
        infer_request.images = [original_path]
        infer_request.data_dict[self._STATE_KEY] = {
            'conversation_id': conversation_id,
            'crop_dir': str(crop_dir),
            'registry': {
                original_path: {
                    'path': original_path,
                    'width': width,
                    'height': height,
                    'original_region': [0.0, 0.0, float(width), float(height)],
                    'original_width': width,
                    'original_height': height,
                }
            },
            'current_image_path': original_path,
            'tool_count': 0,
            'tool_attempt_count': 0,
            'action_correction_count': 0,
            'crop_image_paths': [],
            'crop_data_uri_map': {},
            'tool_events': [],
            'assistant_response_token_ids': [],
            'assistant_rollout_logprobs': [],
            'error': None,
        }

    async def run(self, infer_request, request_config, **kwargs):
        self._initialize_request(infer_request)
        per_turn_max = self.max_completion_tokens
        if request_config.max_tokens is not None:
            per_turn_max = min(per_turn_max, request_config.max_tokens)
        request_config = replace(request_config, max_tokens=per_turn_max)
        output = await super().run(infer_request, request_config, **kwargs)

        # Swift's base server scheduler only keeps token IDs for the final assistant
        # response unless step() injects IDs. Restore every generated assistant turn so
        # training uses the exact rollout tokens while user crop-result turns remain
        # template-encoded and loss-masked.
        state = infer_request.data_dict[self._STATE_KEY]
        response_token_ids = state['assistant_response_token_ids']
        if response_token_ids and all(response_token_ids):
            # Per-turn binary train mask (heuristic PRM + uncertainty filter).
            turn_mask = self._compute_turn_train_mask(infer_request, state)
            output.response_token_ids = response_token_ids
            output.response_loss_mask = [
                [mask_value] * len(token_ids)
                for mask_value, token_ids in zip(turn_mask, response_token_ids)
            ]
            rollout_logprobs = state['assistant_rollout_logprobs']
            if (len(rollout_logprobs) == len(response_token_ids)
                    and all(len(logprobs) == len(token_ids)
                            for logprobs, token_ids in zip(rollout_logprobs, response_token_ids))):
                # Drop logprobs of masked-out turns so that
                # sum(response_loss_mask) == len(rollout_logprobs) still holds on the
                # trainer side and rollout importance sampling stays enabled.
                output.rollout_logprobs = [
                    logprobs
                    for logprobs, mask_value in zip(rollout_logprobs, turn_mask)
                    if mask_value
                ]
            else:
                output.rollout_logprobs = []
        output.rollout_infos = {
            **(output.rollout_infos or {}),
            **self._rollout_infos(infer_request, state),
        }
        return self._sanitize_output(output)

    @staticmethod
    def _sanitize_output(output):
        """Ensure the rollout output survives FastAPI's ``allow_nan=False`` serialization.

        Starlette's ``json.dumps`` rejects NaN/Inf floats. bf16 numeric issues
        (e.g. vision-encoder overflow on very large images) can leave NaN in
        engine logprobs; a single bad token would otherwise abort the whole
        ``/infer/`` HTTP call with a 500, failing every rollout in that batch.
        Drop or null the offending float channels instead of crashing:

        * ``response.choices[*].logprobs`` -> None (client falls back to no logprobs)
        * ``rollout_logprobs`` -> [] (trainer falls back to no IS correction)
        * ``rollout_infos``    -> NaN/Inf floats replaced by None
        """
        try:
            response = getattr(output, 'response', None)
            for choice in getattr(response, 'choices', None) or []:
                if _has_non_finite(getattr(choice, 'logprobs', None)):
                    choice.logprobs = None
                    warnings.warn('[zoomin] dropped NaN/Inf request logprobs from engine output')
            if getattr(output, 'rollout_logprobs', None) and _has_non_finite(output.rollout_logprobs):
                output.rollout_logprobs = []
                warnings.warn('[zoomin] dropped NaN/Inf rollout_logprobs; trainer falls back to no IS correction')
            if getattr(output, 'rollout_infos', None) and _has_non_finite(output.rollout_infos):
                output.rollout_infos = _mask_non_finite(output.rollout_infos)
                warnings.warn('[zoomin] masked NaN/Inf floats in rollout_infos with null')
            if getattr(output, 'messages', None) and _has_non_finite(output.messages):
                output.messages = _mask_non_finite(output.messages)
                warnings.warn('[zoomin] masked NaN/Inf floats in output messages with null')
        except Exception as exc:  # sanitization must never break the rollout itself
            warnings.warn(f'[zoomin] output sanitization skipped: {exc}')
        return output

    def _restore_visible_assistant_content(self, infer_request, response_choice) -> str:
        """Rebuild the visible think/action text before parsing or storing it."""
        content = response_choice.message.content or ''
        token_ids = response_choice.token_ids or []
        if token_ids and self.tokenizer is not None:
            try:
                decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception:
                decoded = ''
            if decoded:
                content = assistant_visible_content(
                    decoded,
                    response_choice.message.reasoning_content or '',
                )
        elif response_choice.message.reasoning_content:
            content = assistant_visible_content(
                content,
                response_choice.message.reasoning_content,
            )

        response_choice.message.content = content
        if infer_request.messages and infer_request.messages[-1].get('role') == 'assistant':
            # MultiTurnScheduler appended the parser-stripped content just before this hook.
            infer_request.messages[-1]['content'] = content
        return content

    def check_finished(self, infer_request, response_choice, current_turn):
        content = self._restore_visible_assistant_content(infer_request, response_choice)
        state = infer_request.data_dict[self._STATE_KEY]
        token_ids = list(response_choice.token_ids or [])
        state['assistant_response_token_ids'].append(token_ids)
        state['assistant_rollout_logprobs'].append(self._extract_logprobs_from_choice(response_choice))
        if super().check_finished(infer_request, response_choice, current_turn):
            return True

        parsed_action = parse_text_action(content)
        if parsed_action is None:
            if state['action_correction_count'] >= self.max_action_corrections:
                state['error'] = (
                    'Model did not return a parseable crop action or final coordinate '
                    f'after {state["action_correction_count"] + 1} attempts.')
                return True
            return False
        if parsed_action['type'] == 'final':
            state['final_point'] = parsed_action['point']
            return True
        return False

    def step(self, infer_request, response_choice, current_turn) -> Dict[str, Any]:
        state = infer_request.data_dict[self._STATE_KEY]
        completion = response_choice.message.content or ''
        parsed_action = parse_text_action(completion)

        if parsed_action is None:
            state['action_correction_count'] += 1
            infer_request.messages.append({'role': 'user', 'content': INVALID_ACTION_CORRECTION})
            return {
                'infer_request': infer_request,
                'rollout_infos': self._rollout_infos(infer_request, state),
            }

        if parsed_action['type'] == 'crop_batch':
            crop_actions = list(parsed_action['actions'])
        else:
            crop_actions = [parsed_action]

        batch_source_image_path = state['current_image_path']
        batch_events = []
        for batch_index, crop_action in enumerate(crop_actions):
            state['tool_attempt_count'] += 1
            arguments = dict(crop_action.get('arguments') or {})
            arguments.setdefault('image_path', batch_source_image_path)
            budget_index = None

            if state['tool_count'] >= self.max_tool_calls:
                result = {
                    'ok': False,
                    'error': (
                        f'Crop budget exhausted: maximum is {self.max_tool_calls}. '
                        'Return <think>...</think> followed by the final '
                        '[x.xx,y.yy] now, using exactly two decimal places.'
                    ),
                }
            else:
                state['tool_count'] += 1
                budget_index = state['tool_count']
                result = execute_crop(
                    arguments,
                    state['registry'],
                    Path(state['crop_dir']),
                    state['tool_count'],
                    crop_upscale_factor=self.crop_upscale_factor,
                    crop_max_pixels=self.crop_max_pixels,
                    max_aspect_ratio=self.max_crop_aspect_ratio,
                )

            event = {
                'tool_call_index': state['tool_attempt_count'],
                'tool_budget_index': budget_index,
                'model_turn': current_turn,
                'batch_index': batch_index,
                'protocol': 'text_action',
                'name': 'crop_image',
                'raw_action': crop_action.get('source'),
                'arguments': _safe_json_value(arguments),
                'result': _safe_json_value(result),
            }
            # data_uri can be large (base64 PNG); keep it only in the
            # dedicated crop_data_uri_map, not in tool_events.
            event['result'].pop('data_uri', None)
            state['tool_events'].append(event)
            batch_events.append(event)
            if result.get('ok'):
                image_path = str(result['image_path'])
                state['current_image_path'] = image_path
                state['crop_image_paths'].append(image_path)
                state['crop_data_uri_map'][image_path] = result.get('data_uri', '')

        user_content, image_paths = crop_results_user_text(batch_events)
        infer_request.messages.append({'role': 'user', 'content': user_content})
        infer_request.images.extend(image_paths)

        return {
            'infer_request': infer_request,
            'rollout_infos': self._rollout_infos(infer_request, state),
        }

    def _compute_turn_train_mask(self, infer_request, state: Dict[str, Any]) -> List[int]:
        """Per-turn binary train mask for multi-turn trajectories.

        Two filters over intermediate turns (the final assistant turn is always
        trained, since it carries the answer that the reward scores):

        * PRM filter — a turn whose crop(s) score below ``prm_min_quality``
          (GT center outside the crop AND near-zero IoU) is a bad zoom decision.
          - mode ``success_only`` (default): only mask it when the trajectory's
            final answer still lands in the GT bbox, i.e. block the noisy
            "bad crop + lucky answer" positive-advantage reinforcement while
            keeping the correct negative signal for genuinely failed ones.
          - mode ``always``: mask regardless of the final outcome (SE-GUI style).
        * Entropy filter — intermediate turns whose uncertainty
          (-mean logprob, a sequence-level proxy for step entropy) falls below
          ``entropy_ratio * median`` across filter-eligible turns are steps the
          policy is already sure about; masking them focuses learning on the
          critical high-uncertainty decisions (DART step selection).

        Masked-out turns keep their tokens in ``response_token_ids`` but with a
        zero loss mask, and their logprobs are dropped by the caller so rollout
        importance sampling stays consistent. Results are stashed in ``state``
        for ``_rollout_infos`` observability.
        """
        token_ids_per_turn = state['assistant_response_token_ids']
        n_turns = len(token_ids_per_turn)
        mask = [1] * n_turns
        state['turn_prm_scores'] = {}
        state['turn_nll'] = {}
        if n_turns <= 1:
            state['turn_train_mask'] = mask
            return mask

        solution = infer_request.data_dict.get('solution')
        tool_events = state.get('tool_events') or []

        if self.prm_filter:
            prm_scores = turn_prm_scores(tool_events, solution, iou_weight=self.prm_iou_weight)
            state['turn_prm_scores'] = {str(k): v for k, v in prm_scores.items()}
            if prm_scores:
                final_correct = point_in_bbox_accuracy(state.get('final_point'), solution) >= 1.0
                for model_turn, score in prm_scores.items():
                    turn_index = model_turn - 1  # model_turn is 1-based, like current_turn
                    if not 0 <= turn_index < n_turns - 1:
                        continue
                    if score >= self.prm_min_quality:
                        continue
                    if self.prm_filter_mode == 'always' or final_correct:
                        mask[turn_index] = 0

        if self.entropy_filter:
            logprobs_per_turn = state.get('assistant_rollout_logprobs') or []
            nlls: Dict[int, float] = {}
            for turn_index in range(n_turns - 1):  # intermediate turns only
                if mask[turn_index] == 0:
                    continue
                if turn_index >= len(logprobs_per_turn):
                    continue
                turn_logprobs = logprobs_per_turn[turn_index]
                if turn_logprobs:
                    nlls[turn_index] = -sum(turn_logprobs) / len(turn_logprobs)
            state['turn_nll'] = {str(k + 1): v for k, v in nlls.items()}
            if len(nlls) >= 2:
                median_nll = statistics.median(nlls.values())
                for turn_index, nll in nlls.items():
                    if nll < median_nll * self.entropy_ratio:
                        mask[turn_index] = 0

        mask[-1] = 1  # the answer-bearing final turn is always trained
        state['turn_train_mask'] = mask
        return mask

    @staticmethod
    def _rollout_infos(infer_request, state: Dict[str, Any]) -> Dict[str, Any]:
        # Replace crop image file paths with base64 data URIs so the
        # training node can access crop images without a shared filesystem.
        # Original (non-crop) images remain as file paths — they exist on
        # both rollout and training nodes.
        uri_map = state.get('crop_data_uri_map', {})
        images = [
            uri_map.get(str(img), img) if isinstance(img, str) else img
            for img in infer_request.images
        ]
        return {
            'images': images,
            'action_protocol': 'assistant_text_then_user_crop_image',
            'conversation_id': state['conversation_id'],
            'crop_directory': state['crop_dir'],
            'crop_image_paths': list(state['crop_image_paths']),
            'error': state['error'],
            'final_content': (
                infer_request.messages[-1].get('content')
                if infer_request.messages and infer_request.messages[-1].get('role') == 'assistant'
                else ''
            ),
            'prediction_relative_0_1000': state.get('final_point'),
            'tool_call_counts': state['tool_count'],
            'tool_attempt_count': state['tool_attempt_count'],
            'tool_events': list(state['tool_events']),
            'turn_train_mask': list(state.get('turn_train_mask') or []),
            'turn_prm_scores': dict(state.get('turn_prm_scores') or {}),
            'turn_nll': dict(state.get('turn_nll') or {}),
        }


multi_turns['zoomin_grounding_scheduler'] = ZoomInGroundingScheduler
