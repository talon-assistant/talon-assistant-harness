"""Shared plan-step execution engine.

Used by both PlannerTalent (multi-step routines) and TaskAssistTalent
(agentic mode).  Extracted to avoid duplicating the step-execution loop.
"""

from typing import Callable

import logging
log = logging.getLogger(__name__)


def execute_plan_steps(
    steps: list[str],
    assistant,
    speak: bool = False,
    notify: Callable | None = None,
    command_source: str = "local",
    on_step_complete: Callable | None = None,
) -> list[dict]:
    """Execute a list of plan steps through the assistant pipeline.

    Parameters
    ----------
    steps : list[str]
        Ordered list of sub-commands to execute.
    assistant : TalonAssistant
        The assistant instance (used to call process_command).
    speak : bool
        Whether talents should speak their responses.
    notify : callable, optional
        Toast notification callback(title, message).
    command_source : str
        Command source tag for routing/logging.
    on_step_complete : callable, optional
        Called after each step with (step_index, step_text, result_dict).

    Returns
    -------
    list[dict]
        One dict per step: {step, command, talent, response, success}.
    """
    step_results = []
    last_user_input = ""
    last_result = ""

    for i, step in enumerate(steps, 1):
        # Substitute placeholders
        if last_user_input and "{user_input}" in step:
            step = step.replace("{user_input}", last_user_input)
        if last_result and "{last_result}" in step:
            step = step.replace("{last_result}", last_result)

        log.info(f"[PlanExecutor] Step {i}/{len(steps)}: {step}")

        # ── ask_user: step ─────────────────────────────────────────────
        if step.strip().lower().startswith("ask_user:"):
            question = step.split(":", 1)[1].strip()
            if notify:
                notify(f"Step {i}/{len(steps)}", f"Needs input: {question}")

            if hasattr(assistant, "request_human_input"):
                last_user_input = assistant.request_human_input(question)
            else:
                last_user_input = ""

            result_dict = {
                "step": i,
                "command": step,
                "talent": "human_input",
                "response": (f"User answered: {last_user_input}"
                             if last_user_input else "(no answer)"),
                "success": bool(last_user_input),
            }
            step_results.append(result_dict)
            if on_step_complete:
                on_step_complete(i, step, result_dict)
            continue

        # Toast for progress
        if notify:
            notify(f"Step {i}/{len(steps)}", step)

        try:
            result = assistant.process_command(
                step,
                speak_response=speak,
                _executing_rule=True,
                _planner_substep=True,
                command_source=command_source,
            )
            if result:
                ok = result.get("success", True)
                resp = result.get("response", "").strip()
                talent_used = result.get("talent", "conversation")
            else:
                ok = False
                resp = "No response."
                talent_used = ""
        except Exception as e:
            ok = False
            resp = f"Error: {e}"
            talent_used = ""
            log.info(f"[PlanExecutor] Step {i} raised exception: {e}")

        result_dict = {
            "step": i,
            "command": step,
            "talent": talent_used,
            "response": resp,
            "success": ok,
        }
        step_results.append(result_dict)

        if resp:
            last_result = resp
        if not ok:
            log.error(f"[PlanExecutor] Step {i} failed: {resp}")

        if on_step_complete:
            on_step_complete(i, step, result_dict)

    return step_results
