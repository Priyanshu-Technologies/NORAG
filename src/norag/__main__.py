import uvicorn

from norag.config import get_settings
from norag.logging import configure_logging


def main() -> None:
    """Run the NORAG API using the configured host and port."""

    settings = get_settings()
    configure_logging(settings.log_level)

    uvicorn.run(
        "norag.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
