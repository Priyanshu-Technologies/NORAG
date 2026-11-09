from dataclasses import dataclass
from threading import Lock


@dataclass
class ServiceCounters:
    """Snapshot of service-level counters exposed through /stats."""

    queries: int = 0
    uploads: int = 0
    deletions: int = 0
    errors: int = 0


class ServiceMetrics:
    """Thread-safe counters describing what the service has processed."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters = ServiceCounters()

    def record_query(self) -> None:
        with self._lock:
            self._counters.queries += 1

    def record_upload(self) -> None:
        with self._lock:
            self._counters.uploads += 1

    def record_deletion(self) -> None:
        with self._lock:
            self._counters.deletions += 1

    def record_error(self) -> None:
        with self._lock:
            self._counters.errors += 1

    def snapshot(self) -> ServiceCounters:
        with self._lock:
            return ServiceCounters(
                queries=self._counters.queries,
                uploads=self._counters.uploads,
                deletions=self._counters.deletions,
                errors=self._counters.errors,
            )
