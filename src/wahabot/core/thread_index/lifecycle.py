from pathlib import Path

from loguru import logger

from wahabot.core.persistence import persistable
from wahabot.core.thread_index.models import index_file, vector_file


def forget_thread_index(data_dir: Path, session: str, chat_id: str) -> bool:
    """Delete a chat's thread index files; True when one was removed."""
    if not persistable(chat_id):
        logger.warning("Not forgetting thread index of bad chat id {id!r}", id=chat_id)
        return False
    removed = False
    for path in (
        index_file(data_dir, session, chat_id),
        vector_file(data_dir, session, chat_id),
    ):
        removed = unlink(path) or removed
        removed = unlink(path.with_name(path.name + ".bad")) or removed
        removed = unlink(path.with_name(path.name + ".tmp")) or removed
    return removed


def unlink(path: Path) -> bool:
    """Delete one file; False when it is already absent."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning(
            "Failed to delete thread index file {path}: {exc}", path=path, exc=exc
        )
        return False
    return True
