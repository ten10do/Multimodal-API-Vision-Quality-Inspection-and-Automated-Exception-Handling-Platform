"""PLC Adapter interface + implementations (Phase 7).

Core business code depends only on the PlcAdapter Protocol; the transport
(HTTP or OPC UA) is injected. Fail-safe: a communication failure surfaces as
PlcUnreachable and the caller must enter SAFE_HOLD.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .commands import IndustrialCommand

logger = logging.getLogger(__name__)


class PlcError(Exception):
    """Base for PLC transport failures."""


class PlcUnreachable(PlcError):
    """Connection-level failure (offline / timeout). Caller must SAFE_HOLD."""


class PlcNack(PlcError):
    """PLC returned NACK (fault / rejected the command)."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class PlcCommandResult:
    acked: bool
    response: dict
    latency_ms: float
    duplicate: bool = False


class PlcAdapter(Protocol):
    name: str

    async def send_command(self, command: IndustrialCommand) -> PlcCommandResult:
        """Send one command; raises PlcUnreachable / PlcNack on failure."""


class HttpPlcAdapter:
    """HTTP adapter for the PLC Simulator (or a real PLC gateway with an HTTP
    API). Retries are the caller's concern; the adapter itself is one-shot."""

    name = "http"

    def __init__(self, base_url: str, timeout_seconds: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds

    async def send_command(self, command: IndustrialCommand) -> PlcCommandResult:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/v1/command", json=command.to_payload())
        except httpx.TimeoutException as exc:
            raise PlcUnreachable(f"plc timeout after {self.timeout}s") from exc
        except httpx.HTTPError as exc:
            raise PlcUnreachable(f"plc unreachable: {exc}") from exc
        latency = (time.perf_counter() - started) * 1000.0
        if resp.status_code != 200:
            raise PlcUnreachable(f"plc http {resp.status_code}")
        body = resp.json()
        if body.get("ack") != "ACK":
            raise PlcNack(str(body))
        return PlcCommandResult(
            acked=True, response=body, latency_ms=round(latency, 2), duplicate=bool(body.get("duplicate", False))
        )


class OpcUaPlcAdapter:
    """OPC UA adapter talking to a simulated PLC server (Phase 7).

    Uses asyncua; the server exposes a `Plc/execute` method that is
    idempotent by command_id. A connection failure raises PlcUnreachable.
    """

    name = "opcua"

    # Stable contract namespace URI of the IVQC PLC simulator. The namespace
    # INDEX is allocated by the server at startup (currently 2) and must never
    # be hard-coded; the adapter resolves URI -> index dynamically at connect
    # time and falls back to browsing by browse name for foreign servers.
    NS_URI = "urn:ivqc:plc"

    def __init__(self, endpoint: str = "opc.tcp://127.0.0.1:8503", timeout_seconds: float = 3.0,
                 ns_uri: str = NS_URI) -> None:
        self.endpoint = endpoint
        self.timeout = timeout_seconds
        self.ns_uri = ns_uri

    async def send_command(self, command: IndustrialCommand) -> PlcCommandResult:
        import json

        from asyncua import Client

        started = time.perf_counter()
        try:
            client = Client(self.endpoint, timeout=self.timeout)
            await client.connect()
            try:
                # Resolve the Plc object WITHOUT any namespace-index
                # hard-coding: a fixed "0:Plc" browse path (or a fixed ns=2
                # index) raises BadNoMatch whenever the server registers the
                # object under a different index. Strategy:
                #   1. preferred: declared namespace URI -> index (the index
                #      is server-allocated and read at connect time)
                #   2. fallback: any object whose browse name is "Plc"
                uri_index = None
                try:
                    namespaces = await client.get_namespace_array()
                    if self.ns_uri in namespaces:
                        uri_index = namespaces.index(self.ns_uri)
                except Exception:  # noqa: BLE001 - URI resolution is best-effort
                    uri_index = None
                candidates = []
                for obj in await client.nodes.objects.get_children():
                    bn = await obj.read_browse_name()
                    if bn.Name == "Plc":
                        candidates.append(obj)
                if not candidates:
                    raise RuntimeError("Plc object not found in address space")
                plc = None
                if uri_index is not None:
                    for cand in candidates:
                        if cand.nodeid.NamespaceIndex == uri_index:
                            plc = cand
                            break
                plc = plc or candidates[0]
                execute = None
                for m in await plc.get_methods():
                    mb = await m.read_browse_name()
                    if mb.Name == "execute":
                        execute = m
                        break
                if execute is None:
                    raise RuntimeError("execute method not found on Plc")
                result = await plc.call_method(execute, json.dumps(command.to_payload()))
                text = str(result)
            finally:
                await client.disconnect()
        except PlcNack:
            raise
        except Exception as exc:  # noqa: BLE001 - transport failure -> fail-safe
            raise PlcUnreachable(f"opcua unreachable: {exc}") from exc
        latency = (time.perf_counter() - started) * 1000.0
        if text.startswith("NACK"):
            raise PlcNack(text)
        duplicate = text == "ACK:duplicate"
        return PlcCommandResult(acked=True, response={"ack": "ACK", "detail": text}, latency_ms=round(latency, 2), duplicate=duplicate)


def get_plc_adapter() -> PlcAdapter:
    """Factory bound to settings (defaults to the HTTP PLC simulator)."""
    from ..config import get_settings

    settings = get_settings()
    if settings.plc_adapter_type == "opcua":
        return OpcUaPlcAdapter(endpoint=settings.plc_opcua_endpoint)
    return HttpPlcAdapter(base_url=settings.plc_url, timeout_seconds=settings.plc_timeout_seconds)
