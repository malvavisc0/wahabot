"""Access lists (whitelist/blacklist) loaded from a JSON config file."""

import json
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


class AccessLists(BaseModel):
    """Chat access control lists.

    An empty whitelist means answer everybody; a blacklist entry is
    always ignored, regardless of the whitelist.
    """

    whitelist: set[str] = set()
    blacklist: set[str] = set()


def load_access_lists(path: Path) -> AccessLists:
    """Read whitelist/blacklist from a JSON file.

    The file may be absent or list nothing; both mean "no filtering".
    """
    if not path.exists():
        logger.info("No access config at {path}; answering everybody", path=path)
        return AccessLists()
    lists = AccessLists.model_validate(json.loads(path.read_text()))
    logger.info(
        "Loaded access lists from {path}: {n_white} whitelisted, {n_black} blacklisted",
        path=path,
        n_white=len(lists.whitelist),
        n_black=len(lists.blacklist),
    )
    return lists
