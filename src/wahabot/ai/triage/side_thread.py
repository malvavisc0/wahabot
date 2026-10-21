def is_side_thread(
    group: bool,
    addressed: bool,
    quoted_other: bool,
    quoted_bot: bool,
    quoted_operator: bool,
) -> bool:
    """True for an unaddressed group reply to another member's message."""
    return (
        group
        and not addressed
        and quoted_other
        and not quoted_bot
        and not quoted_operator
    )
