import openai
from llama_index.core.workflow import WorkflowTimeoutError
from loguru import logger

from wahabot.core.transport import Transport
from wahabot.status.notify import notify_operator
from wahabot.status.state import state


def llm_endpoint_down(exc: Exception) -> bool:
    """Whether *exc* means the LLM endpoint (or its proxy) is unreachable.

    ``APIConnectionError`` covers DNS, refused and dropped connections.
    A reverse proxy in front of the provider answers instead when it is
    the one that is down — a 502/503/504 ``APIStatusError`` — which must
    classify the same: the runs it fails are just as lost, and the
    operator notification exists for exactly this class of outage.

    Timeouts do NOT belong here: ``APITimeoutError`` subclasses
    ``APIConnectionError`` but means the endpoint answered nothing
    within the per-request budget — the provider may be healthy and
    still generating (non-streaming calls send zero bytes until
    completion). A timeout is a slow generation or a hung endpoint,
    never a proved outage, so it gets its own classifier
    (:func:`llm_call_timed_out`) and its own operator message; treating
    it as "unreachable" would pause-replies-narrate an outage that
    never happened.
    """
    if isinstance(exc, openai.APITimeoutError):
        return False
    if isinstance(exc, openai.APIConnectionError):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


def llm_call_timed_out(exc: Exception) -> bool:
    """Whether *exc* is the per-request LLM timeout firing.

    Distinct from an outage (:func:`llm_endpoint_down`): a timeout
    says the request lived out its budget without an answer — the
    model may still be generating on a healthy endpoint. The failure
    is just as fatal to the run, so the chat paths give it the same
    seen-marker drop (WAHA's redelivery retries), but the operator
    message must say what actually happened: a slow/hung generation,
    not an unreachable endpoint.
    """
    return isinstance(exc, openai.APITimeoutError)


def run_timed_out(exc: Exception) -> bool:
    """Whether *exc* is the run-level timeout firing.

    Distinct budget from :func:`llm_call_timed_out`: the run cap kills a
    run whose *total* duration crossed ``WAHABOT_RUN_TIMEOUT`` — several
    rounds each within the per-call budget can add up past the cap, or
    one slow call can leave no room for the wrap-up round. Still a slow
    generation, never an outage (the endpoint may be healthy) and never
    a bug (the cap did its job): the health flag stays up and the
    operator message names the run knob, not the per-request one.

    Two exception shapes reach here: builtin ``TimeoutError`` — the
    ``asyncio.wait_for`` cap around ``agent.run`` in
    :func:`wahabot.ai.context.handle_message`, the enforcement point
    since the library's own ``timeout=`` proved cumulative across a
    reused Context — and
    ``WorkflowTimeoutError``, should a library timeout ever fire again.
    ``openai.APITimeoutError`` does not subclass ``TimeoutError``, so
    the per-call budget class keeps its own, earlier branch.
    """
    return isinstance(exc, (WorkflowTimeoutError, TimeoutError))


async def mark_llm_unreachable(waha: Transport, session: str, exc: Exception) -> None:
    """Flip the LLM health flag down and notify the operator once.

    Called from the failure handlers of the chat and command paths on
    an outage-class exception: the first failure of an outage
    transitions the flag and sends the alert notification; later failures
    (every message that arrives while the provider is down) only log —
    the operator already knows, and the chat path's dropped seen
    markers let WAHA's redelivery retry once the provider is back.
    """
    logger.warning(
        "LLM endpoint unreachable ({exc}); operator notified, messages await redelivery",
        exc=exc,
    )
    if state.llm_healthy:
        state.llm_healthy = False
        message = (
            "LLM provider is unreachable — replies are paused until it"
            " responds again. Incoming messages are held and retried"
            " automatically."
        )
        await notify_operator(waha, session, message)


def llm_failure_class(exc: Exception) -> str:
    """The run outcome class for a model failure, without notifying."""
    if llm_call_timed_out(exc) or run_timed_out(exc):
        return "timeout"
    if llm_endpoint_down(exc):
        return "outage"
    return "bug"


async def classify_llm_failure(
    waha: Transport,
    session: str,
    exc: Exception,
    timeout_s: float,
    run_timeout_s: int,
) -> str:
    """Notify per an LLM failure's class; returns the class for the caller.

    One shared classifier for every run path (single message, burst,
    album, operator command): a timeout — per-request
    (:func:`llm_call_timed_out`) or run-level (:func:`run_timed_out`) —
    notifies as a timeout (the endpoint may be healthy and still
    generating), an outage-class failure flips the health flag and
    notifies "unreachable", and anything else is a bug — no notify at
    all, the caller's full traceback is the report. Callers still own
    the seen-marker drop; this only decides who gets told what.
    """
    failure_class = llm_failure_class(exc)
    if failure_class == "timeout":
        if llm_call_timed_out(exc):
            await mark_llm_timed_out(waha, session, exc, timeout_s)
        else:
            await mark_run_timed_out(waha, session, exc, run_timeout_s)
        return "timeout"
    if failure_class == "outage":
        await mark_llm_unreachable(waha, session, exc)
        return "outage"
    return "bug"


async def mark_llm_timed_out(
    waha: Transport, session: str, exc: Exception, timeout_s: float
) -> None:
    """Notify the operator that an LLM call timed out (once per slow stretch).

    A timeout is not an outage (:func:`llm_endpoint_down`): the outage
    health flag stays untouched — the very next run may succeed against
    the same healthy-but-slow endpoint, and "replies paused" would be
    the wrong message. The operator gets one alert naming the actual
    budget and the knob (``WAHABOT_LLM_TIMEOUT``) on the first timeout
    of a stretch; repeats stay quiet until a successful run resets the
    latch (alongside :func:`mark_llm_recovered`). Callers drop seen
    markers exactly like an outage — a timed-out run is just as lost.
    """
    logger.warning(
        "LLM call timed out after {timeout}s ({exc}); messages await redelivery",
        timeout=timeout_s,
        exc=exc,
    )
    if not state.llm_timeout_notified:
        state.llm_timeout_notified = True
        message = (
            f"LLM call timed out after {timeout_s:.0f}s — the model was still"
            f" generating (or the provider hung). If your provider regularly"
            f" needs longer, raise WAHABOT_LLM_TIMEOUT; replies continue on"
            f" the next message."
        )
        await notify_operator(waha, session, message)


async def mark_run_timed_out(
    waha: Transport, session: str, exc: Exception, run_timeout_s: int
) -> None:
    """Notify the operator that a run hit the run-timeout cap.

    Same failure class as a per-request timeout
    (:func:`mark_llm_timed_out`): the endpoint may be healthy and still
    generating — the *run* outlived its total budget, not one request
    its own — so the health flag stays up and the shared latch keeps it
    to one alert per slow stretch, re-armed by the next successful run.
    The message names the run's knob (``WAHABOT_RUN_TIMEOUT``): the
    per-call budget may be blameless here, and the run cap must leave
    room above it for several rounds.
    """
    logger.warning(
        "Agent run timed out after {timeout}s ({exc}); messages await redelivery",
        timeout=run_timeout_s,
        exc=exc,
    )
    if not state.llm_timeout_notified:
        state.llm_timeout_notified = True
        message = (
            f"Agent run hit the {run_timeout_s:.0f}s cap while the model was"
            f" still working — messages await redelivery. If long runs are"
            f" normal for your provider, raise WAHABOT_RUN_TIMEOUT (it must"
            f" stay above WAHABOT_LLM_TIMEOUT, with room for several rounds)."
        )
        await notify_operator(waha, session, message)


async def mark_llm_recovered(waha: Transport, session: str) -> None:
    """Flip the LLM health flag back up and notify the operator once.

    Called after any run completes against the provider: the first
    success after an outage transitions the flag and sends the recovery
    notification; the steady state stays silent. A success also
    re-arms the timeout notify latch (:func:`mark_llm_timed_out`) —
    the endpoint answering again means the next slow call is a fresh
    slow-stretch worth one new alert.
    """
    if not state.llm_healthy:
        state.llm_healthy = True
        await notify_operator(
            waha,
            session,
            "LLM provider is back — replies resumed.",
        )
    state.llm_timeout_notified = False
