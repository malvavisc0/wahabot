"""Uvicorn HTTP protocol logging detail for unparseable requests.

Stock uvicorn logs a bare "Invalid HTTP request received." when h11
raises RemoteProtocolError (e.g. TLS bytes sent to plain HTTP) and
discards the error. This subclass probes a copy of the connection
before handing data to uvicorn, so a failing request is logged with
the h11 error text, the remote peer, and the offending byte prefix.
"""

from typing import override

import h11
from loguru import logger
from uvicorn.protocols.http.h11_impl import H11Protocol


class LoggingH11Protocol(H11Protocol):
    """H11Protocol that reports why a request failed to parse."""

    @override
    def data_received(self, data: bytes) -> None:
        """Log a diagnostic if the chunk fails HTTP parsing, then defer."""
        error = parse_error(data) if self.fresh_connection() else ""
        if error:
            logger.warning(
                "Invalid HTTP request from {client}: {error}; bytes: {chunk!r}",
                client=self.client or None,
                error=error,
                chunk=data[:64],
            )
            self.send_400_response("Invalid HTTP request received.")
            return
        super().data_received(data)

    def fresh_connection(self) -> bool:
        """Whether the connection is idle and awaiting its first request."""
        return (
            self.conn.our_state is h11.IDLE
            and not self.conn.they_are_waiting_for_100_continue
        )


def parse_error(data: bytes) -> str:
    """The h11 parse error for a chunk sent to a fresh server, empty when valid."""
    try:
        probe = h11.Connection(our_role=h11.SERVER)
        probe.receive_data(data)
        probe.next_event()
    except h11.RemoteProtocolError as error:
        return str(error)
    return ""
