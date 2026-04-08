# -*- coding: utf-8 -*-
"""
main.py — Application entry point for LLM API Prism.
"""

import logging

import uvicorn
from fastapi import FastAPI

from app.routers.proxy import PROXY_ROUTERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="LLM API Prism",
    description="Multi-protocol LLM API proxy with Hub-and-Spoke architecture",
    version="0.1.0",
)

# Register all N×N proxy router combinations (every protocol × every backend).
# Each combination gets its own base_url prefix: /{input_protocol}/{backend}/
# Client only needs to configure base_url, no other parameters needed.
for route_name, router in PROXY_ROUTERS.items():
    app.include_router(router, prefix=f"/{route_name}")


@app.get("/health")
async def health_check() -> dict:
    """Return service health status."""
    return {"status": "ok"}


def main() -> None:
    """Start the uvicorn ASGI server."""
    uvicorn.run("app.main:app", host="0.0.0.0", port=9876)


if __name__ == "__main__":
    main()