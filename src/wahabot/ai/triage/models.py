from dataclasses import dataclass

from wahabot.ai.triage.side_thread import is_side_thread


@dataclass(frozen=True, slots=True)
class TriageVerdict:
    """The gate's labels, reason, ledger gist and hints."""

    adversarial: bool = False
    escalate: bool = False
    participate: bool = False
    why: str = ""
    gist: str = ""
    needs: tuple[str, ...] = ()

    @property
    def fact_keys(self) -> tuple[str, ...]:
        """Notebook keys the gate asked the run to pin."""
        return tuple(
            key
            for need in self.needs
            if need.startswith("note:")
            and ":" in need
            and (key := need.split(":", 1)[1].strip())
        )


@dataclass(frozen=True, slots=True)
class TriageResult:
    """One verdict with its timing and failure state."""

    verdict: TriageVerdict
    latency_ms: int = 0
    timed_out: bool = False
    failed: bool = False

    @classmethod
    def fallback(cls, *, timed_out: bool = False, failed: bool = False) -> TriageResult:
        """The verdict recorded when the gate produced none.

        ``participate`` defaults to False so a missing label is never
        read as permission; the caller separately treats a timeout or
        failure as an unavailable gate and degrades to running the agent.
        """
        return cls(TriageVerdict(), timed_out=timed_out, failed=failed)


@dataclass(frozen=True, slots=True)
class TriageContext:
    """The small, bounded input the gate is allowed to see."""

    session: str
    chat_id: str
    message: str
    turns: tuple[str, ...] = ()
    mute: str = ""
    rules: str = ""
    group: bool = False
    addressed: bool = True
    quoted_other: bool = False
    quoted_bot: bool = False
    quoted_operator: bool = False

    @property
    def side_thread(self) -> bool:
        """True for an unaddressed group reply to another member's message."""
        return is_side_thread(
            self.group,
            self.addressed,
            self.quoted_other,
            self.quoted_bot,
            self.quoted_operator,
        )
