"""Simple projection handlers — direct writes to SQLite tables and FAISS index.

Phase 1: No Event Store yet. Projections write directly on ingest.
Phase 2+: Will be refactored to consume StoreEvents from EventStore.
"""

from ard.infra.logging import log


class Projections:
    """Registry of projection handlers that update read models.

    In Phase 1, handlers are called directly during ingest.
    In Phase 2+, they subscribe to EventStore.apply().
    """

    def __init__(self):
        self._handlers: dict[str, list[callable]] = {}

    def register(self, stream: str, handler: callable) -> None:
        """Register a handler for a given event stream.

        Args:
            stream: Stream name (e.g. "knowledge.chunk", "knowledge.source").
            handler: Callable(event_payload) that updates the read model.
        """
        self._handlers.setdefault(stream, []).append(handler)

    def apply(self, stream: str, payload: dict) -> None:
        """Apply all registered handlers for a stream to the given payload."""
        for handler in self._handlers.get(stream, []):
            try:
                handler(payload)
            except Exception as e:
                log.error("projection_handler_failed", stream=stream, error=str(e))
                raise
