#!/usr/bin/env python3
"""Live LLM pairwise judge for human-alignment annotations."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data_generation.llm_utils import call_llm_json
from benchmark.human_alignment.common import (
    METRIC_BY_CODE,
    METRIC_CODES,
    normalize_preference,
    read_json,
    write_json,
)
from benchmark.human_alignment.summarize_annotations import summarize_annotations

DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_JUDGE_CONCURRENCY = 8
LIVE_JUDGE_METRIC_RUBRIC = {
    "SF": (
        "Source faithfulness. Prefer the system that best preserves and teaches the "
        "provided source-backed concepts during the teaching dialog while avoiding "
        "contradictions. This is a tutoring metric, not a quotation-containment metric: "
        "faithful tutoring can include concise "
        "session summaries, student-friendly analogies, implementation examples, broader "
        "names for related ideas, and application scenarios when they are anchored to the "
        "target gaps and do not contradict the excerpts. Do not treat "
        "outside but domain-plausible elaboration as hallucination by default. Penalize "
        "extra material only when it displaces the source concepts, makes a factual claim "
        "that conflicts with the excerpts, or causes the teaching dialog to mainly focus on "
        "unrelated content. In technical domains, implementation-level terms such as "
        "retrieval, vector databases, knowledge graphs, context injection, dropout, "
        "underfitting, optimization, adversarial training, backpropagation, GPU processing, "
        "or other standard mechanisms can be faithful application examples in the teaching "
        "dialog when they explain the source-backed concept being taught. "
        "Explicit knowledge-source markers are a core signal for this metric. Page/source "
        "references, attribution phrases, or tags such as [rag-1] show that the tutor is "
        "separating retrieved/source-backed knowledge from its own explanation. When those "
        "markers are attached to source-compatible claims, treat them as strong positive "
        "evidence and prefer that system over one that gives similar content without clear "
        "source marking. Do not use SF to reward completeness, pedagogy, or coverage of "
        "more target gaps unless the missing or added material changes source faithfulness; "
        "those belong to other metrics. Prefer the richer tutor if it uses source-compatible "
        "explanations, summaries, and explicit source markers that "
        "make the source concepts operational, especially when the other system only stays "
        "close to excerpt wording or lacks clear knowledge-source marking. "
        "Choose tie when both systems are broadly faithful and the difference is mostly "
        "coverage breadth, literal wording, citation style, or harmless extra explanation."
    ),
    "PER": (
        "Personalization. Prefer the system that adapts better to the student's profile, "
        "knowledge state, current confusion, and communication style during the teaching "
        "dialog. Strong positive evidence includes concise session summaries, explicitly "
        "tracking what the student has understood or corrected, calibrating detail level "
        "to the student's background, and connecting the explanation to the student's goal "
        "without losing the source-backed learning objective. Up-front summary blocks and "
        "answer-first scaffolds are strong personalization signals because they help the "
        "student keep state across the dialog. "
        "Mirroring the student's tone or wording is useful only when it supports learning; "
        "do not over-reward slang, cheerleading, repeated catchphrases, or simply echoing "
        "the student's invented metaphors if they do not improve adaptation."
    ),
    "APP": (
        "Applicability. Prefer the system that better helps the student make progress on "
        "the task and success criteria during the teaching dialog. Strong positive evidence "
        "includes source-aware worked examples, concrete transfer scenarios, checkpoints "
        "where the student applies the idea, and summaries that turn the source concept into "
        "an actionable rule. Strong applicability often appears inside each tutor response "
        "as an answer-first takeaway, a portable rule, a step-by-step procedure, a decision "
        "criterion, a formula/template, or an explicit next action the student can reuse "
        "outside the conversation. Up-front summaries and answer-first takeaways are strong "
        "evidence that the tutor is converting the source-backed concept into usable "
        "guidance. Do not treat the number of in-dialog exercises or checkpoints as the "
        "main evidence; checkpoints help only when they produce a clearly reusable rule. "
        "Prefer direct, transferable guidance over a longer sequence of local exercises "
        "when both address the same target gap. Do not reward off-task activities, "
        "excessive roleplay, or long detours merely because they are interactive. Do not "
        "treat more worked examples as better if they move beyond the current success "
        "criteria or add adjacent topics that the student did not need. Prefer focused, "
        "portable guidance that helps the student apply the target concept now, anchored "
        "to the student's stated goal and the target gaps. Choose tie when both systems "
        "give usable, task-grounded guidance and the difference is mainly direct rule-first "
        "guidance versus more interactive local practice."
    ),
    "VID": (
        "Vividness. Prefer the system with more concrete, vivid, and example-supported "
        "teaching-dialog explanations. Strong positive evidence includes inspectable "
        "structure, clear headings, concise summaries, tables or comparisons, equations or "
        "code/data snippets when appropriate, and concrete examples that make the source "
        "concept easier to see. Up-front summary sections and clean markdown organization "
        "are vividness positives because they make the lesson easier to inspect. Do not "
        "over-reward decorative slang, emoji, theatrical "
        "metaphors, or sheer quantity of formatting. Prefer vividness that improves clarity "
        "and preserves the learning objective."
    ),
    "LD": (
        "Logical depth. Prefer the system with deeper, more coherent conceptual reasoning. "
        "Strong positive evidence includes explicit concept chains, clear causal or "
        "mathematical links between steps, source-aware distinctions, worked derivations, "
        "and summaries that integrate the student's corrected understanding. Also consider "
        "the logical length and density inside individual tutor responses: a response is "
        "deeper when it connects several necessary intermediate steps within the same answer "
        "(premise -> mechanism -> consequence -> general rule or transfer), rather than "
        "stating only isolated facts or examples. The steps must be necessary for the target "
        "gap; do not count adjacent-topic derivations as depth. Up-front "
        "summaries and answer-first conceptual scaffolds are positive when they make the "
        "reasoning easier to follow. Do not reward verbosity, extra complexity, or long "
        "chains by itself. Penalize conceptual sprawl: a dialog that moves into adjacent "
        "or new topics can be less logically deep for the requested task even if each "
        "individual step is coherent. Prefer reasoning that remains focused on the target "
        "gaps, makes the needed distinctions cleanly, and is useful for the student's task. "
        "Choose tie when both systems have coherent reasoning and the difference is mainly "
        "more compact answer-first chains versus longer interactive derivations."
    ),
    "FIT": (
        "Practice question fitness. Prefer the practice set that better fits the "
        "student after the teaching dialog: their profile, original target gaps, "
        "misconceptions revealed in the exchange, and the concepts the tutor actually "
        "scaffolded. Strong positive evidence includes questions that use the student's "
        "project context or examples from the dialog, target lingering confusions rather "
        "than only retesting basics, and stretch the student just beyond their current "
        "state without becoming inaccessible. Do not automatically prefer the set that "
        "matches the initial gap labels most literally if it ignores the student's "
        "end-of-dialog progress, stated goals, or the applied scenarios used to teach "
        "the concept. Conversely, penalize advanced or tangential questions when they "
        "were not prepared by the dialog or would overwhelm the student's profile. In "
        "technical or applied subjects, well-scaffolded pipeline, mechanism, edge-case, "
        "or implementation questions can be a better fit than basic taxonomy recall "
        "when the dialog has prepared the student to use those ideas for their project "
        "or next learning goal."
    ),
    "GND": (
        "Practice question groundedness. Prefer the practice set that is more traceable "
        "to the provided source-backed concepts as taught in the dialog. Grounding does "
        "not require copying source wording or using only examples named in the excerpts: "
        "a new realistic scenario, project application, standard technical term, or "
        "dialog-created analogy can be well grounded when it accurately exercises a "
        "source-backed concept. Penalize questions that contradict the excerpts or make "
        "unsupported outside material the main thing being tested. Do not over-penalize "
        "practice questions for using implementation details or transfer examples that "
        "the teaching dialog explicitly introduced to explain the source concept. For "
        "source excerpts written as surveys, taxonomies, or high-level mechanisms, "
        "representative techniques and pipeline details introduced in the dialog can be "
        "grounded when they instantiate the excerpt's category, process, limitation, or "
        "design tradeoff. Treat a practice item as ungrounded only when its central "
        "tested knowledge cannot be connected through the dialog to any source-backed "
        "category, mechanism, limitation, example class, or success criterion, or when "
        "it contradicts the source-backed lesson. When both sets are source-compatible, "
        "prefer the set whose questions make the source-backed concepts more operational "
        "and assessable through accurate mechanisms, workflows, project applications, "
        "edge cases, or transfer scenarios. Choose tie only when both sets are similarly "
        "traceable and similarly operational, or when the difference is mainly literal "
        "closeness or coverage breadth. GND is not a coverage metric: a practice set that "
        "assesses a narrower subset of clearly source-backed concepts can be more grounded "
        "than one that covers more of the dialog but relies on concepts whose source link "
        "is weaker or only implicit. GND is also not a shortest-trace metric: a longer "
        "source -> taught mechanism -> assessed application chain can be strongly grounded "
        "when the dialog explicitly built the intermediate mechanism and the final question "
        "accurately preserves the source-backed relationship. At the same time, do not "
        "reward adjacent topics merely because a dialog taught them; if a practice set "
        "centers material outside the shared source excerpts, target gaps, and task success "
        "criteria without a clear source-backed bridge, it is less grounded."
    ),
    "DIV": (
        "Practice question diversity. Prefer the practice set that tests genuinely varied "
        "cognitive operations, concept angles, representations, and transfer situations. "
        "Diversity is not just topic breadth: a coherent pipeline or single applied "
        "scenario can be diverse when different questions test definition, mechanism, "
        "tradeoff, calculation, diagnosis, transfer, and misconception repair. Do not "
        "prefer a broad checklist of source terms if most questions use the same recall "
        "pattern. Also do not penalize reuse of one anchor scenario when each question "
        "requires a distinct reasoning move."
    ),
    "ANS": "Practice question answer quality. Prefer the practice set with better options, answers, and non-trivial distractors.",
    "CC": (
        "Practice question cross-concept. Prefer the practice set that better connects "
        "related concepts where appropriate for the task. Strong positive evidence "
        "includes questions that require the student to coordinate two or more ideas in "
        "a coherent workflow, mechanism, proof, model, or project scenario. Cross-concept "
        "quality is not the same as covering many unrelated topics: do not reward detours "
        "or adjacent concepts merely because they are numerous. Prefer connections that "
        "serve the original task and the dialog's learning trajectory, such as linking "
        "setup to mechanism to consequence, or connecting a source principle to its "
        "application and limitation."
    ),
}
TRANSCRIPT_METRIC_CODES = {"SF", "PER", "APP", "VID", "LD"}
SYSTEM_PROMPT = """You are an expert blind evaluator for TutorBench.

You compare System A and System B for the same student profile, task, source excerpts,
dialog, and practice questions. For each metric, choose exactly one of:

- A: System A is better.
- B: System B is better.
- tie: the two systems are comparable or evidence is insufficient.

Use the metric definitions exactly. Do not infer backend identity. Return only valid JSON.
"""


def parse_metric_codes(raw: str | None = None) -> list[str]:
    if not raw:
        return list(METRIC_CODES)
    requested = [part.strip().upper() for part in raw.split(",") if part.strip()]
    unknown = [code for code in requested if code not in METRIC_CODES]
    if unknown:
        raise ValueError(f"Unknown metric code(s): {', '.join(unknown)}. Valid: {', '.join(METRIC_CODES)}")
    return requested


def _load_package(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows[str(item["pair_id"])] = item
    return rows


def _load_key(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    items = data.get("items", data if isinstance(data, list) else [])
    return {str(item["pair_id"]): item for item in items}


def _load_annotated_pair_ids(path: Path) -> list[str]:
    import csv

    ids: set[str] = set()
    def has_human_label(row: dict[str, Any]) -> bool:
        return any(normalize_preference(row.get(code)) is not None for code in METRIC_CODES)

    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    pair_id = str(row.get("pair_id", "")).strip()
                    if pair_id and has_human_label(row):
                        ids.add(pair_id)
        return sorted(ids)

    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pair_id = str(row.get("pair_id", "")).strip()
            if pair_id and has_human_label(row):
                ids.add(pair_id)
    return sorted(ids)


def _select_pair_ids(
    *,
    annotations_path: Path,
    key_by_pair: dict[str, dict[str, Any]],
    package_by_pair: dict[str, dict[str, Any]],
    judge_all_pairs: bool,
) -> list[str]:
    if judge_all_pairs:
        return sorted(set(key_by_pair) & set(package_by_pair))
    return [
        pair_id
        for pair_id in _load_annotated_pair_ids(annotations_path)
        if pair_id in key_by_pair and pair_id in package_by_pair
    ]


def _short(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def _format_dialog(dialog: list[dict[str, Any]], max_chars: int) -> str:
    blocks = []
    for idx, msg in enumerate(dialog, start=1):
        role = str(msg.get("role", "")).upper()
        content = _short(msg.get("content", ""), max_chars)
        blocks.append(f"{idx}. {role}\n{content}")
    return "\n\n".join(blocks) or "(none)"


def _format_questions(questions: list[Any], max_chars: int) -> str:
    blocks = []
    for idx, question in enumerate(questions, start=1):
        blocks.append(f"Q{idx}. {_short(question, max_chars)}")
    return "\n\n".join(blocks) or "(none)"


def _format_gaps(gaps: list[dict[str, Any]], max_chars: int) -> str:
    blocks = []
    for gap in gaps:
        source = "\n".join(
            f"Page {page}: {_short(text, max_chars // 2)}"
            for page, text in (gap.get("source_excerpts") or {}).items()
        )
        blocks.append(
            "\n".join(
                [
                    f"Gap: {gap.get('gap_id', '')} {gap.get('target_concept', '')}",
                    f"Type: {gap.get('gap_type', '')}",
                    f"Description: {_short(gap.get('description', ''), max_chars)}",
                    f"Correct understanding: {_short(gap.get('correct_understanding', ''), max_chars)}",
                    f"Source excerpts:\n{source or '(none)'}",
                ]
            )
        )
    return "\n\n".join(blocks) or "(none)"


def _judge_prompt(item: dict[str, Any], metric_code: str) -> str:
    profile = item.get("profile", {}) or {}
    task = item.get("task", {}) or {}
    system_a = item.get("system_a", {}) or {}
    system_b = item.get("system_b", {}) or {}
    metric_guidance = ""
    if metric_code in TRANSCRIPT_METRIC_CODES:
        metric_guidance = """
Transcript metric scope:
- Evaluate only the teaching dialog for this metric.
- Ignore practice questions; they are evaluated by FIT/GND/DIV/ANS/CC.
- Prefer source-aware summaries, task-grounded scaffolding, and learning progress over surface entertainment or verbosity.
- Treat up-front "Summary" sections, answer-first takeaways, and clean markdown organization as positive signals when they support the metric.
- For non-SF transcript metrics, do not reward citation markers or [rag-1] tags by themselves; use them only if they concretely improve the metric being judged.
- For APP and LD, do not reward extra breadth, extra worked examples, or longer concept chains unless they directly advance the current success criteria.
- For APP, value reusable action structure inside the teaching dialog: answer-first takeaways, step procedures, formulas/templates, decision criteria, and explicit next actions.
- For APP, do not use the number of local exercises/checkpoints as the main evidence unless they leave the student with a portable rule or decision procedure.
- For LD, value logical length inside individual tutor responses when the intermediate steps are necessary and connected, not just a long dialog overall.
- For APP/LD close cases, use tie when both systems are substantively useful and the contrast is mainly compact rule-first guidance versus longer interactive derivation.
- Avoid position bias: do not prefer System A or System B just because it is first, longer, more theatrical, or repeats more of the student's slang.
- Choose tie when the difference is mostly style, wording, or amount of enthusiasm.
"""
    if metric_code == "SF":
        metric_guidance = """
Transcript metric scope:
- Evaluate only the teaching dialog for this metric.
- Ignore practice questions; they are evaluated by FIT/GND/DIV/ANS/CC.
- Prefer source-aware summaries, task-grounded scaffolding, and learning progress over surface entertainment or verbosity.

SF decision procedure:
1. Evaluate only the teaching dialog for SF. Ignore practice questions; they are evaluated by separate metrics.
2. Check for clear contradictions with the excerpts or major unsupported concepts that drive the teaching dialog.
3. If one system has such major issues and the other does not, prefer the more faithful system.
4. Treat source faithfulness as compatible with tutoring: examples, analogies, summaries, implementation scenarios, and broader labels can be faithful when they illuminate the excerpt-backed concepts.
5. Do not prefer a system just because it paraphrases the excerpts more literally or avoids useful explanation.
6. Give strong positive SF credit when a system explicitly links its explanation to retrieved/source-backed content with relevant tags such as [rag-1], page/source references, or attribution phrases. If the linked claim is source-compatible, this should usually break an otherwise close tie.
7. Do not turn SF into a completeness or teaching-quality metric: covering more gaps or giving more detailed explanations should not decide SF unless it changes faithfulness.
8. If both systems mostly teach concepts compatible with the excerpts, choose tie even if one has closer wording or covers more excerpt details.
9. If one system better connects the student's misconception to source-backed concepts and the other is only loosely grounded, that is positive SF evidence.
10. When deciding whether extra content is harmful, ask whether a student could still trace the main teaching dialog back to the provided source-backed ideas. If yes, the extra content should not decide against that system.
11. Treat standard implementation terms in AI, economics, computer science, philosophy, or calculus as acceptable teaching examples when they support the source-backed idea; do not require those terms to appear verbatim in the excerpts.
12. A concise session summary that accurately frames the source-backed lesson is positive SF evidence because it reduces drift across a long dialog.
"""
    if metric_code not in TRANSCRIPT_METRIC_CODES:
        metric_guidance = f"""
Practice question metric scope:
- Evaluate only the practice questions and their explanations for {metric_code}; use the shared profile, task, source excerpts, and each system's own teaching dialog as context for what that system's practice set should assess.
- Judge System A practice questions against System A's teaching dialog, and System B practice questions against System B's teaching dialog. Do not penalize one system because the other system did not teach the same scaffold, analogy, worked example, or implementation detail.
- Think of two parallel versions of the same student: one completed System A's dialog before seeing System A's practice, and the other completed System B's dialog before seeing System B's practice. Do not ask whether System B's questions would fit a student who learned through System A, or vice versa.
- When the two dialogs taught different but source-compatible routes toward the same task, compare how well each practice set assesses its own taught route while still serving the shared success criteria.
- Do not use a practice-question metric to re-judge whether the teaching dialog should have chosen a different route through the lesson; teaching quality and scope are evaluated by other metrics. For practice metrics, ask whether the questions are a good assessment of the learning route that actually occurred.
- Treat the student as being at the end of the teaching dialog, not at the first turn. Good practice may target corrected misconceptions, transfer from worked examples, or concepts the tutor explicitly scaffolded during the session.
- Prefer questions that are answerable from the source-backed lesson and dialog. New scenarios are allowed when they exercise the same source-backed concept rather than becoming a new lesson.
- Do not prefer a set merely because it repeats more source words, gap labels, or dialog examples. Also do not prefer a set merely because it is longer, more advanced, or more polished.
- Penalize questions that mostly test concepts outside the stated task or outside what the dialog prepared the student to practice.
- When one set is a literal checklist of initial gaps and the other is a coherent end-of-session assessment, choose based on the metric: fit to the learned state for FIT, traceability for GND, distinct reasoning moves for DIV, answer/option quality for ANS, and meaningful conceptual integration for CC.
- Choose tie when both practice sets would be similarly useful and the difference is mostly wording, scenario flavor, or question difficulty.
"""
    if metric_code == "FIT":
        metric_guidance += """
FIT decision procedure:
1. Identify the student's profile, original gaps, success criteria, and where the dialog ended.
2. For each system separately, identify what its own dialog actively scaffolded and where that student would be at the end of that dialog.
3. Prefer the practice set that best targets remaining confusion and useful transfer for that end state.
4. Give positive credit for using the student's project context, analogies, or scenarios from that system's dialog when they make the exercise more relevant.
5. Do not over-reward simple recall of already-mastered concepts or definitions listed as known_well.
6. Do not over-penalize an applied or slightly advanced question if that system's dialog built the needed scaffold and the question remains answerable.
7. Do not decide FIT by saying one dialog over-reached the initial task if its practice questions fairly assess the student's end-of-dialog learned state and remain connected to the shared success criteria.
8. Prefer the set that better supports transfer to the student's stated project or next learning goal when both sets cover answerable material.
9. For students who prefer hands-on, advanced, project-based, or technical learning, give FIT credit to questions that assess mechanism use, pipeline choices, edge cases, or design tradeoffs prepared by the dialog; do not default to easier foundational recall as the better fit.
10. Do not prefer a set solely because it covers more initial gap labels or repeats the earliest examples. If the other set better matches what its own dialog prepared the parallel student to do next, and remains source-compatible, that is strong FIT evidence.
11. Treat coverage of every initial gap as secondary to appropriate end-of-session assessment quality. A practice set can fit well by assessing a narrower or more applied route when that is where the dialog and student goal naturally ended.
"""
    elif metric_code == "GND":
        metric_guidance += """
GND decision procedure:
1. Ask whether each question's tested idea can be traced to a source excerpt or to its own system's dialog explanation of a source-backed idea.
2. Treat faithful transfer examples as grounded when the underlying concept is source-backed and accurately preserved.
3. Penalize unsupported outside concepts only when they become the main tested knowledge or conflict with the excerpts.
4. Do not require exact source wording, page references, or [rag-1] markers in practice questions.
5. Do not treat a well-scaffolded implementation detail as ungrounded merely because it is a modern name or mechanism for applying the source concept.
6. When a question uses a named technique, implementation term, or applied scenario that was taught in the dialog, evaluate whether the underlying tested relationship is source-compatible rather than whether the exact term appears in the excerpt.
7. If both sets are source-compatible, prefer the set whose questions make the source-backed concepts operational through accurate transfer, not the set that merely stays closest to excerpt vocabulary.
8. In survey-like or taxonomy-like source material, examples of compression, retrieval, context-window management, graph storage, optimization, robustness, heuristics, or workflow design may be grounded even when the exact label is absent, if the dialog connected them to the excerpt's stated category or limitation.
9. Do not downgrade standard disciplinary mechanisms such as backpropagation, dropout, marginal rules, utility-based demand reasoning, retrieval reordering, or prompt compression merely because the exact term is absent. Downgrade them only if the question tests the mechanism as disconnected outside knowledge rather than as an explanation or application of the source-backed idea.
10. If one set is literal and another is an accurate operationalization, choose the operationalized set when it better tests whether the student can use the source-backed concept; otherwise choose tie instead of baseline.
11. Use a conservative threshold for choosing a winner on GND, but do not flatten meaningful operational grounding into style. A set that turns source-backed concepts into accurate mechanisms, workflows, edge cases, project applications, or transfer scenarios can be more grounded than a set that only repeats source taxonomy or familiar examples.
12. Prefer one set on GND only for a clear grounding difference: contradiction, unsupported central knowledge in multiple items, or practice questions that cannot be answered from the source-backed lesson as taught.
13. When both sets are consistent, the stronger GND set is the one whose questions provide clearer traceability from source idea -> taught mechanism -> assessed application. Prefer tie only if that traceability is comparably strong in both.
14. Do not reward broader coverage of the teaching dialog, more source headings, or more original gaps by itself. If a narrower set stays inside explicitly supported source concepts while a broader set depends on less visible derivations or loosely related extensions, prefer the narrower grounded set.
15. Do not let one debatable or weakly grounded question decide the metric if the rest of that practice set is strongly source-compatible and operational. Prefer tie, or prefer that set if its overall source idea -> taught mechanism -> assessed application chain is stronger.
16. Do not prefer a system merely because its traceability chain is shorter or more literal. If the other system's questions preserve the source-backed relationship through a dialog-built mechanism and test it in a concrete application, that can be stronger GND evidence.
17. Use the shared task scenario and success criteria as part of grounding context. In debate tasks, ethical frameworks mentioned in the task scenario can be grounded practice material when used to assess source-backed dialectical tracking; in technical tasks, implementation mechanisms can be grounded when they instantiate a source-backed process, limitation, or design tradeoff.
18. For AI and agent-memory lessons, mechanisms such as adversarial training, backpropagation, dropout, embeddings, vector retrieval, top-K selection, context injection, reordering, or compression can be grounded if the dialog explicitly connected them to source-backed pattern learning, robustness, bias, context-window limits, retrieval quality, or memory organization.
"""
    elif metric_code == "DIV":
        metric_guidance += """
DIV decision procedure:
1. Count distinct reasoning moves, not just distinct nouns or source headings.
2. Prefer sets that vary across recall, mechanism, calculation, comparison, diagnosis, prediction, transfer, and misconception repair.
3. A single case study can be diverse if each question probes a different part of the mechanism or decision process.
4. A broad set can still be repetitive if each item is the same definitional multiple-choice pattern.
"""
    elif metric_code == "CC":
        metric_guidance += """
CC decision procedure:
1. Prefer questions that require coordinating related ideas, not merely naming them separately.
2. Reward links that mirror the learning trajectory: concept -> mechanism -> consequence -> application or limitation.
3. Do not reward unrelated breadth, topic sprawl, or questions that connect concepts the student was not prepared to combine.
4. A coherent applied scenario can be strong cross-concept evidence when it makes the student integrate multiple taught ideas.
"""
    practice_sections = ""
    gap_chars = 1200
    dialog_chars = 1800
    question_chars = 1600
    if metric_code not in TRANSCRIPT_METRIC_CODES:
        gap_chars = 2200
        dialog_chars = 2600
        question_chars = 1800
        practice_sections = f"""
SYSTEM A PRACTICE QUESTIONS:
{_format_questions(system_a.get("practice_questions", []) or [], question_chars)}

SYSTEM B PRACTICE QUESTIONS:
{_format_questions(system_b.get("practice_questions", []) or [], question_chars)}
"""
    return f"""Compare System A and System B for this pair on exactly ONE metric: {metric_code}.

Return JSON with exactly this shape:
{{
  "preference": "A|B|tie",
  "rationale": "brief reason"
}}
The response must be parseable JSON. In the rationale string, avoid double quotes;
use single quotes or paraphrase quoted terms instead.

Metric definition:
- {metric_code}: {LIVE_JUDGE_METRIC_RUBRIC[metric_code]}
{metric_guidance}

Shared profile:
{json.dumps(profile, ensure_ascii=False, indent=2)}

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Gaps and source excerpts:
{_format_gaps(item.get("gaps", []) or [], gap_chars)}

SYSTEM A DIALOG:
{_format_dialog(system_a.get("dialog", []) or [], dialog_chars)}

SYSTEM B DIALOG:
{_format_dialog(system_b.get("dialog", []) or [], dialog_chars)}
{practice_sections}
"""


def _side_to_backend_pref(side_pref: str, key: dict[str, Any]) -> str | None:
    value = str(side_pref or "").strip().lower()
    if value == "tie":
        return "tie"
    if value == "a":
        backend = key.get("system_a_backend")
    elif value == "b":
        backend = key.get("system_b_backend")
    else:
        return None
    if backend == key.get("target_backend"):
        return "target"
    if backend == key.get("baseline_backend"):
        return "baseline"
    return None


def _extract_metric_preference(response: dict[str, Any], metric_code: str) -> tuple[str, str]:
    preference = response.get("preference")
    rationale = response.get("rationale")
    if preference is None and isinstance(response.get("preferences"), dict):
        preference = response["preferences"].get(metric_code)
    if not isinstance(rationale, str) and isinstance(response.get("rationale"), dict):
        rationale = response["rationale"].get(metric_code)
    return str(preference or ""), str(rationale or "")


async def _judge_one_metric(
    *,
    pair_id: str,
    metric_code: str,
    item: dict[str, Any],
    key: dict[str, Any],
    model: str,
    binding: str | None,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        response = await call_llm_json(
            user_prompt=_judge_prompt(item, metric_code),
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            response_format={"type": "json_object"},
            **{k: v for k, v in {"binding": binding, "base_url": base_url, "api_key": api_key}.items() if v},
        )
    preference, rationale = _extract_metric_preference(response, metric_code)
    backend_preference = _side_to_backend_pref(preference, key)
    return {
        "pair_id": pair_id,
        "metric_code": metric_code,
        "model": model,
        "binding": binding,
        "side_preference": preference,
        "backend_preference": backend_preference if backend_preference in {"target", "baseline", "tie"} else "",
        "rationale": rationale,
        "raw_response": response,
    }


async def run_live_judge(
    *,
    annotations_path: Path,
    key_path: Path,
    package_path: Path,
    output_path: Path,
    model: str = DEFAULT_JUDGE_MODEL,
    binding: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    temperature: float = 0.0,
    max_tokens: int = 1800,
    limit_pairs: int = 0,
    metric_codes: list[str] | None = None,
    judge_all_pairs: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    key_by_pair = _load_key(key_path)
    package_by_pair = _load_package(package_path)
    pair_ids = _select_pair_ids(
        annotations_path=annotations_path,
        key_by_pair=key_by_pair,
        package_by_pair=package_by_pair,
        judge_all_pairs=judge_all_pairs,
    )
    if limit_pairs > 0:
        pair_ids = pair_ids[:limit_pairs]
    metric_codes = metric_codes or list(METRIC_CODES)

    if verbose:
        print("Live LLM judge starting")
        print(f"  model       : {model}")
        print(f"  binding     : {binding or '(from existing LLM config)'}")
        print(f"  base_url    : {base_url or '(from existing LLM config)'}")
        print(f"  annotations : {annotations_path}")
        print(f"  package     : {package_path}")
        print(f"  pair mode   : {'all package pairs' if judge_all_pairs else 'human-annotated pairs only'}")
        print(f"  pairs       : {len(pair_ids)}")
        print(f"  metrics     : {','.join(metric_codes)}")
        print(f"  judge calls : {len(pair_ids) * len(metric_codes)} ({len(metric_codes)} metrics per pair)")
        print(f"  concurrency : {max(1, concurrency)}")
        print(f"  max_tokens  : {max_tokens}")

    semaphore = asyncio.Semaphore(max(1, concurrency))
    started = time.monotonic()
    tasks = {}
    for pair_id in pair_ids:
        for metric_code in metric_codes:
            task = asyncio.create_task(
                _judge_one_metric(
                    pair_id=pair_id,
                    metric_code=metric_code,
                    item=package_by_pair[pair_id],
                    key=key_by_pair[pair_id],
                    model=model,
                    binding=binding,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    semaphore=semaphore,
                )
            )
            tasks[task] = (pair_id, metric_code)

    metric_items = []
    completed = 0
    for task in asyncio.as_completed(tasks):
        metric_item = await task
        metric_items.append(metric_item)
        completed += 1
        if verbose:
            elapsed = time.monotonic() - started
            print(
                f"  [{completed}/{len(tasks)}] judged "
                f"{metric_item['pair_id']} {metric_item['metric_code']} ({elapsed:.1f}s elapsed)"
            )

    by_pair: dict[str, dict[str, Any]] = {}
    for metric_item in metric_items:
        pair_id = str(metric_item["pair_id"])
        record = by_pair.setdefault(
            pair_id,
            {
                "pair_id": pair_id,
                "model": model,
                "binding": binding,
                "side_preferences": {},
                "backend_preferences": {},
                "rationale": {},
                "raw_response": {},
            },
        )
        code = str(metric_item["metric_code"])
        record["side_preferences"][code] = metric_item.get("side_preference", "")
        if metric_item.get("backend_preference") in {"target", "baseline", "tie"}:
            record["backend_preferences"][code] = metric_item["backend_preference"]
        record["rationale"][code] = metric_item.get("rationale", "")
        record["raw_response"][code] = metric_item.get("raw_response", {})

    items = [by_pair[pair_id] for pair_id in sorted(by_pair)]
    result = {
        "step": "human_alignment_live_llm_judge",
        "judge_granularity": "per_metric",
        "annotations_path": str(annotations_path),
        "annotation_key_path": str(key_path),
        "annotation_package_path": str(package_path),
        "model": model,
        "binding": binding or "",
        "base_url": base_url or "",
        "metric_codes": metric_codes,
        "pair_selection": "all_pairs" if judge_all_pairs else "human_annotated_pairs",
        "num_pairs_judged": len(items),
        "num_metric_judgments": len(metric_items),
        "items": items,
    }
    write_json(output_path, result)
    if verbose:
        elapsed = time.monotonic() - started
        print(f"Live LLM judge done: {len(items)} pairs / {len(metric_items)} metric judgments in {elapsed:.1f}s")
        print(f"Live judge JSON: {output_path}")
    return result


def live_preferences_by_pair(judge_result: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        str(item["pair_id"]): dict(item.get("backend_preferences", {}) or {})
        for item in judge_result.get("items", [])
    }


def summarize_with_live_judge(
    *,
    annotations_path: Path,
    key_path: Path,
    package_path: Path,
    summary_output_path: Path,
    judge_output_path: Path,
    model: str = DEFAULT_JUDGE_MODEL,
    binding: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    temperature: float = 0.0,
    max_tokens: int = 1800,
    limit_pairs: int = 0,
    metric_codes: list[str] | None = None,
    judge_all_pairs: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    judge_result = asyncio.run(
        run_live_judge(
            annotations_path=annotations_path,
            key_path=key_path,
            package_path=package_path,
            output_path=judge_output_path,
            model=model,
            binding=binding,
            base_url=base_url,
            api_key=api_key,
            concurrency=concurrency,
            temperature=temperature,
            max_tokens=max_tokens,
            limit_pairs=limit_pairs,
            metric_codes=metric_codes,
            judge_all_pairs=judge_all_pairs,
            verbose=verbose,
        )
    )
    summary = summarize_annotations(
        annotations_path=annotations_path,
        key_path=key_path,
        output_path=summary_output_path,
        live_llm_preferences=live_preferences_by_pair(judge_result),
        live_metric_codes=metric_codes or list(METRIC_CODES),
        judge_metadata={
            "judge_output_path": str(judge_output_path),
            "model": model,
            "binding": binding or "",
            "base_url": base_url or "",
            "metric_codes": metric_codes or list(METRIC_CODES),
            "pair_selection": judge_result.get("pair_selection", ""),
            "num_pairs_judged": judge_result.get("num_pairs_judged", 0),
            "num_metric_judgments": judge_result.get("num_metric_judgments", 0),
            "judge_granularity": judge_result.get("judge_granularity", ""),
        },
    )
    if verbose:
        print(f"Summary JSON: {summary_output_path}")
        print(f"Summary MD  : {summary_output_path.with_suffix('.md')}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live Claude pairwise judge for human alignment")
    parser.add_argument("--annotations", required=True, help="Completed annotation CSV/JSONL")
    parser.add_argument("--key", required=True, help="annotation_key.json")
    parser.add_argument("--package", default="", help="annotation_package.jsonl (default: next to key)")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="Judge model")
    parser.add_argument("--binding", default="", help="Override LLM provider binding; default uses existing LLM config")
    parser.add_argument("--base-url", default="", help="Override judge API base URL; default uses existing LLM config")
    parser.add_argument("--api-key", default=None, help="Judge API key (default: provider env var)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_JUDGE_CONCURRENCY, help="Concurrent judge calls")
    parser.add_argument("--max-tokens", type=int, default=1800, help="Max tokens per judge response")
    parser.add_argument("--limit-pairs", type=int, default=0, help="Debug: judge only first N selected pairs")
    parser.add_argument("--metrics", default="", help="Comma-separated metric codes to judge, e.g. SF or SF,PER")
    parser.add_argument("--judge-all-pairs", action="store_true", help="Judge every pair in annotation_package.jsonl, including pairs without human labels")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    parser.add_argument("--judge-output", default="", help="Live judge JSON output")
    parser.add_argument("--summary-output", default="", help="Summary JSON output")
    args = parser.parse_args()

    key_path = Path(args.key)
    package_path = Path(args.package) if args.package else key_path.parent / "annotation_package.jsonl"
    default_judge_name = "live_llm_judgments_all_pairs.json" if args.judge_all_pairs else "live_llm_judgments.json"
    judge_output_path = Path(args.judge_output) if args.judge_output else key_path.parent / default_judge_name
    summary_output_path = Path(args.summary_output) if args.summary_output else key_path.parent / "human_alignment_summary.json"
    summarize_with_live_judge(
        annotations_path=Path(args.annotations),
        key_path=key_path,
        package_path=package_path,
        summary_output_path=summary_output_path,
        judge_output_path=judge_output_path,
        model=args.model,
        binding=args.binding or None,
        base_url=args.base_url or None,
        api_key=args.api_key,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        limit_pairs=args.limit_pairs,
        metric_codes=parse_metric_codes(args.metrics),
        judge_all_pairs=args.judge_all_pairs,
        verbose=not args.quiet,
    )
    print(f"Live judge: {judge_output_path}")
    print(f"Summary: {summary_output_path}")
    print(f"Markdown: {summary_output_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
