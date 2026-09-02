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
        if (
            self.conn.our_state is h11.IDLE
            and not self.conn.they_are_waiting_for_100_continue
        ):
            probe = h11.Connection(our_role=h11.SERVER)
            try:
                probe.receive_data(data)
                probe.next_event()
            except h11.RemoteProtocolError as error:
                client = self.client if self.client else None
                logger.warning(
                    "Invalid HTTP request from {client}: {error}; bytes: {chunk!r}",
                    client=client,
                    error=error,
                    chunk=data[:64],
                )
                self.send_400_response("Invalid HTTP request received.")
                return
        super().data_received(data)
