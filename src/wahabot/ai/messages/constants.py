import re

NON_REPLYABLE_SUFFIXES = ("@broadcast", "@newsletter")

REACTION_TARGET_KWARG = "reaction_target_id"

TURN_HANDLED_KWARG = "turn_handled"

MERGED_TURNS_KWARG = "merged_turns"

WRAP_UP_NOTE_KWARG = "wrap_up_note"

OPERATOR_FOLD_KWARG = "operator_fold"

TURN_SERIALS_KWARG = "turn_serials"

EMOJI_CHARS = (
    r"\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF"
    r"\U00002702-\U000027B0"
    r"\U0000FE00-\U0000FE0F"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF"
    r"\U00002600-\U000026FF"
    r"\U0000200D"
    r"\U0000FE0F"
)

REACTION_ONLY_RE = re.compile(rf"^(?:[{EMOJI_CHARS}]|[^\w\s])+$")

OPERATOR_PAYLOAD_FLAG = "_wahabot_operator"

OPERATOR_NOTE_PREFIX = "[operator message] "

UNADDRESSED_NOTE = (
    "\n[you were not addressed: this is other members talking in the group."
    " Do not answer another member's question, a remark, or a shared link as"
    " if it were yours. Reply only if you can add something genuinely useful;"
    " otherwise call stay_silent]"
)

GENERIC_ADDRESSED_NOTE = "\n[you were addressed: this message is for you]"

BLAME_RE = re.compile(
    r"\b(?:te has equivocado|you(?:'re| are) (?:wrong|mistaken)|"
    + r"du (?:hast dich geirrt|liegst falsch))\b",
    re.IGNORECASE,
)
