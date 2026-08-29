"""MCP (Model Context Protocol) client for Talon.

Lets Talon act as an MCP host: connect to external MCP servers, pull their
tools into the native tool-calling loop, and route the model's calls back out
to the right server. Talents stay first-class; MCP tools are just additional
entries in the same tools[] list the agentic loop already builds.

Config lives in config/mcp_servers.json (gitignored), Claude-Desktop format::

    {
      "mcpServers": {
        "filesystem": {"command": "npx", "args": ["-y",
            "@modelcontextprotocol/server-filesystem", "C:\\\\path"]},
        "remote":     {"url": "https://host/mcp"}
      }
    }

The SDK is async; Talon's talents are sync. We run one asyncio loop in a
daemon thread and give each server a worker coroutine that owns its session
and services a request queue, so every session call stays in the task that
opened the session (this avoids anyio cross-task cancellation problems). The
sync API (tool_schemas / call_tool) hands work to that loop and blocks on the
result.

Fail-soft throughout: a missing SDK, a missing or empty config, or a server
that won't start just means fewer tools, never a crashed assistant.
"""
import os
import json
import atexit
import asyncio
import threading

import logging
log = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _HAS_MCP = True
except Exception:                      # pragma: no cover - import guard
    _HAS_MCP = False

_CONFIG_PATH = os.path.join("config", "mcp_servers.json")
_PREFIX = "mcp__"


class MCPManager:
    """Owns connections to configured MCP servers and exposes their tools."""

    def __init__(self, config_path=_CONFIG_PATH, connect_timeout=25):
        self._servers = {}
        self._loop = None
        self._thread = None
        self._sessions = {}        # server name -> ClientSession
        self._queues = {}          # server name -> asyncio.Queue
        self._tool_schemas = []    # OpenAI-format function schemas
        self._tool_routes = {}     # qualified tool name -> (server, raw tool)
        self._tasks = []           # worker tasks (for clean shutdown)
        self._connect_timeout = connect_timeout

        if not _HAS_MCP:
            log.info("[MCP] python 'mcp' SDK not installed; MCP disabled.")
            return
        self._servers = self._load_config(config_path)
        if not self._servers:
            log.info("[MCP] no servers configured; MCP disabled.")
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="mcp-loop", daemon=True)
        self._thread.start()
        try:
            asyncio.run_coroutine_threadsafe(
                self._connect_all(), self._loop
            ).result(timeout=self._connect_timeout + 10)
        except Exception as e:
            log.warning(f"[MCP] connect phase error: {e}")
        log.info(f"[MCP] {len(self._sessions)} server(s) connected, "
                 f"{len(self._tool_schemas)} tool(s) available.")
        # Clean up subprocess transports at interpreter exit (before the daemon
        # loop thread is killed) so we don't segfault tearing them down on exit.
        atexit.register(self.shutdown)

    # ── config ─────────────────────────────────────────────────────
    @staticmethod
    def _load_config(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        servers = data.get("mcpServers") or data.get("servers") or {}
        clean = {}
        for name, cfg in servers.items():
            if not isinstance(cfg, dict) or cfg.get("disabled"):
                continue
            if not (cfg.get("command") or cfg.get("url")):
                log.warning(f"[MCP] server '{name}' has no command or url; skipping.")
                continue
            # Sanitize to a safe identifier with no '__' (used as a name delimiter).
            safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
            while "__" in safe:
                safe = safe.replace("__", "_")
            safe = safe.strip("_") or "server"
            clean[safe] = cfg
        return clean

    # ── asyncio loop thread ────────────────────────────────────────
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_all(self):
        readies = []
        for name, cfg in self._servers.items():
            ev = asyncio.Event()
            readies.append(ev)
            self._tasks.append(self._loop.create_task(self._worker(name, cfg, ev)))
        if readies:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(e.wait() for e in readies)),
                    timeout=self._connect_timeout)
            except asyncio.TimeoutError:
                log.warning("[MCP] some servers did not report ready in time "
                            "(they may still connect in the background).")

    def _open_transport(self, cfg):
        if cfg.get("url"):
            from mcp.client.streamable_http import streamablehttp_client
            return streamablehttp_client(cfg["url"])
        env = dict(os.environ)
        env.update(cfg.get("env") or {})
        params = StdioServerParameters(
            command=cfg["command"], args=cfg.get("args") or [], env=env)
        return stdio_client(params)

    async def _worker(self, name, cfg, ready):
        """Own one server's session for the life of the loop; serve a queue."""
        try:
            async with self._open_transport(cfg) as conn:
                read, write = conn[0], conn[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self._register(name, listed.tools)
                    self._sessions[name] = session
                    q = asyncio.Queue()
                    self._queues[name] = q
                    ready.set()
                    while True:
                        item = await q.get()
                        if item is None:        # shutdown sentinel
                            break
                        tool, args, fut = item
                        try:
                            res = await session.call_tool(tool, args)
                            if not fut.done():
                                fut.set_result(res)
                        except Exception as e:
                            if not fut.done():
                                fut.set_exception(e)
        except Exception as e:
            log.warning(f"[MCP] server '{name}' failed: {e}")
        finally:
            self._sessions.pop(name, None)
            self._queues.pop(name, None)
            if not ready.is_set():
                ready.set()

    # Tools that mutate the server's underlying resource. A server config with
    # "readOnly": true registers none of these, so the model is never offered a
    # write path — the only reliable enforcement point, since MCP servers
    # (e.g. server-filesystem) expose writes with no read-only flag of their own.
    _WRITE_TOOL_NAMES = frozenset({
        "write_file", "edit_file", "move_file", "create_directory",
        "delete_file", "delete_directory", "remove_file", "copy_file",
        "rename_file", "put_file", "append_file", "patch_file",
    })
    _WRITE_TOOL_HINTS = ("write", "edit", "delete", "remove", "create",
                         "move", "rename", "append", "patch", "put", "upload")

    def _is_write_tool(self, tool_name):
        low = (tool_name or "").lower()
        if low in self._WRITE_TOOL_NAMES:
            return True
        return any(h in low for h in self._WRITE_TOOL_HINTS)

    def _register(self, server, tools):
        cfg = self._servers.get(server, {}) or {}
        read_only = bool(cfg.get("readOnly") or cfg.get("read_only"))
        allow = {str(x) for x in (cfg.get("allowTools") or [])}
        deny = {str(x) for x in (cfg.get("denyTools") or [])}

        for t in (tools or []):
            if allow and t.name not in allow:
                log.info(f"[MCP] '{server}': skipping {t.name} "
                         f"(not in allowTools)")
                continue
            if t.name in deny:
                log.info(f"[MCP] '{server}': skipping {t.name} (denyTools)")
                continue
            if read_only and self._is_write_tool(t.name):
                log.info(f"[MCP] '{server}': skipping {t.name} "
                         f"(server is readOnly)")
                continue
            qname = f"{_PREFIX}{server}__{t.name}"
            schema = t.inputSchema if isinstance(t.inputSchema, dict) else None
            if not schema or schema.get("type") != "object":
                schema = {"type": "object", "properties": {}}
            self._tool_schemas.append({
                "type": "function",
                "function": {
                    "name": qname,
                    "description": (t.description or t.name or qname)[:1024],
                    "parameters": schema,
                },
            })
            self._tool_routes[qname] = (server, t.name)
            log.info(f"[MCP] registered tool {qname}")

    # ── public sync API ────────────────────────────────────────────
    @property
    def enabled(self):
        return bool(self._tool_schemas)

    def connected_servers(self):
        return sorted(self._sessions.keys())

    def tool_schemas(self):
        return list(self._tool_schemas)

    def is_mcp_tool(self, name):
        return isinstance(name, str) and name in self._tool_routes

    def tool_is_mutating(self, qualified_name):
        """Return whether a registered tool appears to change external state.

        This shares the same conservative name classifier used by readOnly
        registration.  The capability broker is the second enforcement layer:
        writable tools may be configured, but they are not silently executed.
        """
        route = self._tool_routes.get(qualified_name)
        return bool(route and self._is_write_tool(route[1]))

    def capability_inventory(self):
        """Return normalized coverage records for all registered MCP tools."""
        records = []
        for qualified_name, (server, raw_name) in sorted(self._tool_routes.items()):
            mutating = self._is_write_tool(raw_name)
            records.append({
                "owner": qualified_name,
                "owner_type": "mcp",
                "access": "brokered" if mutating else "read_only",
                "capabilities": ("mcp_write",) if mutating else (),
                "enforcement": "host" if mutating else "none",
                "status": "protected" if mutating else "read_only",
                "detail": f"{server}:{raw_name}",
                "sandbox": "host",
            })
        return records

    def call_tool(self, qualified_name, arguments, timeout=60):
        route = self._tool_routes.get(qualified_name)
        if not route or not self._loop:
            return f"MCP tool '{qualified_name}' is not available."
        server, tool = route

        async def _do():
            q = self._queues.get(server)
            if q is None:
                raise RuntimeError(f"server '{server}' is not connected")
            fut = self._loop.create_future()
            await q.put((tool, arguments or {}, fut))
            return await fut

        try:
            result = asyncio.run_coroutine_threadsafe(
                _do(), self._loop).result(timeout=timeout)
        except Exception as e:
            return f"MCP call to '{qualified_name}' failed: {e}"
        return self._result_to_text(result)

    @staticmethod
    def _result_to_text(result):
        parts = []
        for block in (getattr(result, "content", None) or []):
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
            else:
                parts.append(f"[{getattr(block, 'type', 'content')}]")
        out = "\n".join(parts).strip() or "(no output)"
        if getattr(result, "isError", False):
            out = f"[tool error] {out}"
        return out

    def shutdown(self, timeout=5):
        """Stop cleanly: drain workers so each exits its session and subprocess
        contexts, then stop the loop and join the thread. Doing this in order
        avoids a Windows asyncio segfault from tearing down subprocess
        transports while the loop is still being stopped."""
        if not self._loop or not self._loop.is_running():
            return

        async def _drain():
            for q in list(self._queues.values()):
                await q.put(None)            # tell each worker to break
            if self._tasks:
                # Let workers exit their async-with blocks (terminates the
                # stdio subprocesses cleanly) before we stop the loop.
                await asyncio.wait(self._tasks, timeout=timeout)

        try:
            asyncio.run_coroutine_threadsafe(
                _drain(), self._loop).result(timeout=timeout + 2)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3)
