"""
Author: Rutger van Haasteren -- rutger@vhaasteren.com
Date:   2025-10-10

Process sandbox for libstempo/tempo2 that keeps each pulsar in its own clean
subprocess. A segfault in tempo2/libstempo only kills the worker, not your kernel.

Usage (drop-in):
    from sandbox import tempopulsar
    psr = tempopulsar(parfile="J1713.par", timfile="J1713.tim", dofit=False)
    r = psr.residuals()

Advanced with logging:
    from sandbox import tempopulsar, configure_logging, Policy
    configure_logging(level="DEBUG", log_file="tempo2.log")
    policy = Policy(ctor_retry=5, call_timeout_s=300.0)
    psr = tempopulsar(parfile="J1713.par", timfile="J1713.tim", policy=policy)

With specific environment:
    psr = tempopulsar(parfile="J1713.par", timfile="J1713.tim", env_name="myenv")
    # or for conda: env_name="mycondaenv"
    # or explicit path: env_name="python:/path/to/python"

With persistent workers (no recycling/timeouts):
    policy = Policy(
        call_timeout_s=None,        # No RPC timeouts
        max_calls_per_worker=None,  # Never recycle by call count
        max_age_s=None,            # Never recycle by age
        rss_soft_limit_mb=None     # Never recycle by memory
    )
    psr = tempopulsar(parfile="J1713.par", timfile="J1713.tim", policy=policy)

Advanced:
    from sandbox import load_many, Policy
    ok, retried, failed = load_many([("J1713.par","J1713.tim"), ...], policy=Policy())

Environment selection (Apple Silicon + Rosetta etc.):
    psr = tempopulsar(..., env_name="tempo2_intel")       # conda env
    psr = tempopulsar(..., env_name="myvenv")             # venv (~/.venvs/myvenv, etc.)
    psr = tempopulsar(..., env_name="arch")               # system python via Rosetta (arch -x86_64)
    psr = tempopulsar(..., env_name="python:/abs/python") # explicit Python path

You can force Rosetta prefix via env var:
    TEMPO2_SANDBOX_WORKER_ARCH_PREFIX="arch -x86_64"

Logging:
    The sandbox includes comprehensive loguru logging for debugging and monitoring.
    Use configure_logging() to set up logging levels and outputs. Logs include:
    - Worker process lifecycle (creation, recycling, termination)
    - RPC call details and timing
    - Constructor retry attempts and failures
    - Memory usage and recycling decisions
    - Error details and recovery attempts

Robustness:
    The sandbox redirects stdout to stderr in worker processes before importing libstempo,
    preventing interference with the JSON-RPC protocol. This ensures reliable
    communication even when libstempo or other libraries print diagnostic messages.
    The redirection works at the OS file descriptor level to catch output from C libraries.
"""

from __future__ import annotations

import base64
import collections
import contextlib
import dataclasses
import json
import os
import pickle
import platform
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections.abc import Mapping

# Import TimFileAnalyzer for proactive TOA counting
from .tim_file_analyzer import TimFileAnalyzer

# Standard logging
import logging

logger = logging.getLogger(__name__)

# ---------------------------- Public Exceptions ---------------------------- #


class Tempo2Error(Exception):
    """Base class for sandbox errors."""


class Tempo2Crashed(Tempo2Error):
    """The worker process crashed or died unexpectedly (likely a segfault)."""


class Tempo2Timeout(Tempo2Error):
    """The worker did not reply in time; it was terminated."""


class Tempo2ProtocolError(Tempo2Error):
    """Malformed RPC request/response or other IPC failure."""


class Tempo2ConstructorFailed(Tempo2Error):
    """Constructor failed even after retries."""


# ------------------------------- Policy knobs ----------------------------- #
@dataclass(frozen=True)
class Policy:
    """Configuration policy for sandbox worker behavior and lifecycle management.

    Controls retry behavior, timeouts, and worker recycling policies.
    """

    # Constructor protection
    ctor_retry: int = 5  # number of extra tries after the first
    ctor_backoff: float = 0.75  # seconds between ctor retries
    preload_residuals: bool = False  # call residuals() once after ctor
    preload_designmatrix: bool = False  # call designmatrix() once after ctor
    preload_toas: bool = False  # call toas() once after ctor
    preload_fit: bool = False  # call fit() once after ctor

    # RPC protection
    call_timeout_s: Optional[float] = None  # per-call timeout (seconds), None = no timeout
    kill_grace_s: float = 2.0  # after timeout, wait before SIGKILL

    # Recycling / hygiene
    max_calls_per_worker: Optional[int] = None  # recycle after this many good calls, None = never recycle by calls
    max_age_s: Optional[float] = None  # recycle after this many seconds, None = never recycle by age
    rss_soft_limit_mb: Optional[int] = None  # if provided, recycle when beaten

    # Proactive TOA handling for large files
    auto_nobs_retry: bool = True  # automatically add nobs parameter for large TOA files
    nobs_threshold: int = 10000  # add nobs parameter if TOA count exceeds this threshold
    nobs_safety_margin: float = 1.1  # multiplier for nobs parameter (e.g., 1.1 = 10% more than actual count)

    # Logging / stderr capture
    stderr_ring_max_lines: int = 20000
    stderr_log_file: Optional[str] = None
    include_stderr_tail_in_errors: int = 200  # 0 disables tail inclusion


# -------------------------- Wire serialization helpers --------------------- #

# We send JSON-RPC 2.0 frames. To avoid JSON-encoding numpy arrays and
# cross-arch issues, params/result travel as base64-encoded cloudpickle blobs.

try:
    import cloudpickle as _cp  # best-effort; falls back to pickle if missing
except Exception:
    _cp = pickle


def _b64_dumps_py(obj: Any) -> str:
    """Serialize Python object to base64-encoded string using cloudpickle."""
    return base64.b64encode(_cp.dumps(obj)).decode("ascii")


def _b64_loads_py(s: str) -> Any:
    """Deserialize base64-encoded string to Python object using cloudpickle."""
    if hasattr(s, 'encode'):
        s_str = s
    else:
        s_str = str(s)
    return _cp.loads(base64.b64decode(s_str.encode("ascii")))


def _format_exc_tuple() -> Tuple[str, str, str]:
    """Format current exception info as tuple of (type_name, message, traceback)."""
    et, ev, tb = sys.exc_info()
    name = et.__name__ if et else "Exception"
    return (name, str(ev), "".join(traceback.format_exception(et, ev, tb)))


def _current_rss_mb_portable() -> Optional[int]:
    """Get current process RSS memory usage in MB, portable across platforms."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm") as f:
                pages = int(f.read().split()[1])
            rss = pages * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
            return rss
    except Exception:
        pass
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        return None


# ----------------------------- Worker (stdio) ------------------------------ #

# --- Worker-side helpers (called inside the subprocess) ---


def _numpy_safe_copy(value):
    """Convert numpy arrays/scalars to serialization-safe Python objects."""
    try:
        import numpy as _np  # noqa: F811
        if isinstance(value, _np.ndarray):
            return value.copy(order="C")
        elif isinstance(value, _np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _extract_param_fallback(cur):
    """Best-effort extraction of val/err/fit from unpicklable libstempo parameter objects."""
    try:
        has_val = hasattr(cur, "val") or hasattr(cur, "_val")
        has_err = hasattr(cur, "err") or hasattr(cur, "_err")
        has_fit = (
            hasattr(cur, "fit") or hasattr(cur, "fitFlag") or hasattr(cur, "_fitFlag")
        )
        if has_val or has_err or has_fit:
            val = err = fit = None
            with contextlib.suppress(Exception):
                val = _numpy_safe_copy(
                    getattr(cur, "val", getattr(cur, "_val", None))
                )
            with contextlib.suppress(Exception):
                err = _numpy_safe_copy(
                    getattr(cur, "err", getattr(cur, "_err", None))
                )
            with contextlib.suppress(Exception):
                f = getattr(cur, "fit", None)
                if f is None:
                    f = getattr(cur, "fitFlag", getattr(cur, "_fitFlag", None))
                fit = bool(f) if isinstance(f, (int, bool)) else f
            name_guess = None
            with contextlib.suppress(Exception):
                name_guess = getattr(cur, "name", None) or getattr(cur, "label", None)
            return {
                "__libstempo_param__": True,
                "name": name_guess,
                "val": val,
                "err": err,
                "fit": fit,
            }
    except Exception:
        pass
    return {"__repr__": repr(cur)}


def _worker_handle_get(obj, params):
    """Handle 'get' RPC: resolve dotted attribute path, return {kind, value} dict."""
    name = params["name"]
    parts = str(name).split(".") if isinstance(name, str) else [name]
    cur = obj
    for idx, part in enumerate(parts):
        if idx == 0:
            try:
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                elif isinstance(cur, Mapping) or hasattr(cur, "__getitem__"):
                    cur = cur[part]
                else:
                    raise AttributeError
            except Exception:
                return {"kind": "missing", "value": None}
        else:
            try:
                cur = getattr(cur, part)
            except Exception:
                return {"kind": "missing", "value": None}

    if callable(cur):
        return {"kind": "callable", "value": None}

    cur = _numpy_safe_copy(cur)

    try:
        _b64_dumps_py({"kind": "value", "value": cur})
        return {"kind": "value", "value": cur}
    except Exception:
        return {"kind": "value", "value": _extract_param_fallback(cur)}


def _worker_handle_set(obj, params):
    """Handle 'set' RPC: set attribute via dotted path."""
    name, value = params["name"], params["value"]
    parts = str(name).split(".") if isinstance(name, str) else [name]
    cur = obj
    for idx, part in enumerate(parts[:-1]):
        try:
            if idx == 0:
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                elif isinstance(cur, Mapping) or hasattr(cur, "__getitem__"):
                    cur = cur[part]
                else:
                    raise AttributeError
            else:
                cur = getattr(cur, part)
        except Exception:
            raise AttributeError(f"cannot resolve path for set: {name}")  # noqa: B904
    setattr(cur, parts[-1], value)


def _worker_handle_setitem(obj, params):
    """Handle 'setitem' RPC: set slice on numpy array attribute."""
    arr = getattr(obj, params["name"])
    value = params["value"]
    try:
        import numpy as _np  # noqa: F811
        if isinstance(value, _np.ndarray) and not _np.can_cast(
            value.dtype, arr.dtype, casting="safe"
        ):
            value = value.astype(arr.dtype, copy=False)
    except Exception:
        pass
    arr[params["index"]] = value


def _worker_handle_get_slice(obj, params):
    """Handle 'get_slice' RPC: get slice from numpy array attribute."""
    arr = getattr(obj, params["name"])
    return _numpy_safe_copy(arr[params["index"]])


def _worker_handle_call(obj, params):
    """Handle 'call' RPC: call a method on the pulsar object."""
    meth = getattr(obj, params["name"])
    out = meth(*tuple(params.get("args", ())), **dict(params.get("kwargs", {})))
    return _numpy_safe_copy(out)


# --- Main worker entry point ---


def _worker_stdio_main() -> None:  # noqa: E501
    """Runs inside the worker subprocess.

    Protocol:
      1) Print a single 'hello' JSON line with environment info.
      2) Serve JSON-RPC 2.0 requests on stdin/stdout.

    stdout is redirected to stderr before importing libstempo so that
    C-level printf output cannot corrupt the JSON-RPC protocol channel.
    """
    import os as _os_for_fds

    _proto_out = None  # type: ignore
    _env_fd = os.environ.get("TEMPO2_SANDBOX_PROTO_FD")
    if _env_fd is not None:
        try:
            _proto_fd = int(_env_fd)
            _proto_out = _os_for_fds.fdopen(_proto_fd, "w", buffering=1)
        except Exception:
            _proto_out = None
        finally:
            with contextlib.suppress(Exception):
                os.environ.pop("TEMPO2_SANDBOX_PROTO_FD", None)
    if _proto_out is None:
        _proto_fd = _os_for_fds.dup(1)
        _os_for_fds.dup2(2, 1)
        sys.stdout = sys.stderr
        _proto_out = _os_for_fds.fdopen(_proto_fd, "w", buffering=1)

    # --- Hello handshake ---
    hello = {
        "hello": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "has_libstempo": False,
            "tempo2_version": None,
            "proto_version": "1.2",
            "capabilities": {
                "get_kind": True, "dir": True, "setitem": True,
                "get_slice": True, "path_access": True,
            },
        }
    }
    try:
        try:
            from libstempo import tempopulsar as _lib_tempopulsar  # noqa: F811
            import numpy  # noqa: F811
            hello["hello"]["has_libstempo"] = True
            try:
                from libstempo import tempo2  # type: ignore
                hello["hello"]["tempo2_version"] = getattr(
                    tempo2, "TEMPO2_VERSION", None
                )
            except Exception:
                pass
        except Exception:
            pass
    finally:
        _proto_out.write(json.dumps(hello) + "\n")
        _proto_out.flush()

    try:
        from libstempo import tempopulsar as _lib_tempopulsar  # noqa: F811
        import numpy  # noqa: F811, F401
    except Exception:
        _lib_tempopulsar = None  # type: ignore[assignment]

    obj = None

    def _ok(rid, result):
        _proto_out.write(
            json.dumps({"jsonrpc": "2.0", "id": rid, "result_b64": _b64_dumps_py(result)}) + "\n"
        )
        _proto_out.flush()

    def _err(rid, code, message, data=""):
        _proto_out.write(
            json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message, "data": data}}) + "\n"  # noqa: E501
        )
        _proto_out.flush()

    # --- JSON-RPC loop ---
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            _err(None, -32700, "parse error")
            continue

        rid = req.get("id")
        method = req.get("method", "")
        params_b64 = req.get("params_b64")

        params = {}
        if params_b64 is not None:
            try:
                params = _b64_loads_py(params_b64)
                if not isinstance(params, dict):
                    raise TypeError("params_b64 must decode to dict")
            except Exception:
                et, ev, tb = _format_exc_tuple()
                _err(rid, -32602, f"invalid params: {ev}", tb)
                continue

        try:
            if method == "bye":
                _ok(rid, "bye")
                return

            if method == "rss":
                _ok(rid, _current_rss_mb_portable())
                continue

            if method == "ctor":
                if _lib_tempopulsar is None:
                    raise ImportError("libstempo not available in worker")
                obj = _lib_tempopulsar(**params["kwargs"])
                if params.get("preload_residuals", True):
                    obj.residuals(updatebats=True, formresiduals=True)
                _ok(rid, "constructed")
                continue

            if obj is None:
                raise RuntimeError("object not constructed")

            if method == "get":
                _ok(rid, _worker_handle_get(obj, params))
            elif method == "set":
                _worker_handle_set(obj, params)
                _ok(rid, None)
            elif method == "setitem":
                _worker_handle_setitem(obj, params)
                _ok(rid, None)
            elif method == "get_slice":
                _ok(rid, _worker_handle_get_slice(obj, params))
            elif method == "call":
                _ok(rid, _worker_handle_call(obj, params))
            elif method == "del":
                obj = None
                _ok(rid, None)
            elif method == "dir":
                names = sorted(n for n in dir(obj) if not n.startswith("_"))
                _ok(rid, names)
            else:
                _err(rid, -32601, f"method not found: {method}")
        except Exception:
            et, ev, tb = _format_exc_tuple()
            _err(rid, -32000, f"{et}: {ev}", tb)


# ------------------------------ Subprocess client -------------------------- #


class _WorkerProc:
    """
    JSON-RPC over stdio subprocess.
    Launches the worker in the requested environment (conda/venv/arch/system).
    """

    def __init__(self, policy: Policy, cmd: List[str], require_x86_64: bool = False):
        self.policy = policy
        self.cmd = cmd
        self.proc: Optional[subprocess.Popen] = None
        self._id = 0
        self._rpc_lock = threading.Lock()
        logger.info(f"Creating worker process with command: {' '.join(cmd)}")
        logger.info(f"Require x86_64 architecture: {require_x86_64}")
        self._start(require_x86_64=require_x86_64)

    # ---------- process management ----------

    def _start(self, require_x86_64: bool = False):
        logger.debug("Starting worker subprocess...")
        self._hard_kill()  # just in case

        # Ensure unbuffered text I/O
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        logger.debug(f"Launching subprocess with environment: PYTHONUNBUFFERED={env.get('PYTHONUNBUFFERED')}")
        logger.debug(f"Subprocess working directory: {os.getcwd()}")
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=os.getcwd(),
            env=env,
            close_fds=True,
            start_new_session=True,
        )

        logger.debug(f"Worker process started with PID: {self.proc.pid}")

        self._log_buf = collections.deque(maxlen=self.policy.stderr_ring_max_lines)
        log_file = None
        if self.policy.stderr_log_file:
            try:
                log_file = open(self.policy.stderr_log_file, "a", buffering=1, encoding="utf-8")
            except Exception:
                log_file = None

        def _drain_stderr(pipe, sink_deque, sink_file):
            try:
                for line in iter(pipe.readline, ""):
                    line = line.rstrip("\n")
                    sink_deque.append(line)

                    # Write to real stderr for real-time output (native-like behavior)
                    print(line, file=sys.stderr, flush=True)

                    if sink_file:
                        try:
                            sink_file.write(line + "\n")
                        except Exception:
                            pass
                    logger.debug("[tempo2-stderr] %s", line)
            finally:
                with contextlib.suppress(Exception):
                    pipe.close()
                if sink_file:
                    with contextlib.suppress(Exception):
                        sink_file.flush()
                        sink_file.close()

        if self.proc.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=_drain_stderr, args=(self.proc.stderr, self._log_buf, log_file), daemon=True
            )
            self._stderr_thread.start()

        # Hello handshake (one line of JSON)
        logger.debug("Waiting for worker hello handshake...")
        hello = self._readline_with_timeout(self.policy.call_timeout_s)
        if hello is None:
            if self.policy.call_timeout_s is None:
                logger.error("Worker did not send hello - worker disconnected")
                self._hard_kill()
                raise Tempo2Crashed("worker did not send hello - worker disconnected")
            else:
                logger.error("Worker did not send hello in time")
                self._hard_kill()
                raise Tempo2Timeout("worker did not send hello in time")

        try:
            hello_obj = json.loads(hello)
        except Exception as e:
            logger.error(f"Failed to parse worker hello: {e}")
            self._hard_kill()
            raise Tempo2ProtocolError(f"malformed hello: {hello!r}")

        info = hello_obj.get("hello", {})
        logger.info(f"Worker hello received: {info}")
        self._proto_version = info.get("proto_version", "1.0")

        if require_x86_64:
            if str(info.get("machine", "")).lower() != "x86_64":
                logger.error(f"Architecture mismatch: worker is {info.get('machine')}, but x86_64 required")
                self._hard_kill()
                raise Tempo2Error(f"worker arch is {info.get('machine')}, but x86_64 is required for quad precision")

        if not info.get("has_libstempo", False):
            logger.error("libstempo not available in worker environment")
            # Keep the worker up; subsequent ctor will return a clean error,
            # but we can already warn here to fail fast.
            self._hard_kill()
            raise Tempo2Error(
                "libstempo is not importable inside the selected environment. "
                f"Worker executable: {info.get('executable')}"
            )

        self.birth = time.time()
        self.calls_ok = 0
        logger.info(f"Worker ready and initialized (PID: {self.proc.pid})")

    def _readline_with_timeout(self, timeout: Optional[float]) -> Optional[str]:
        if self.proc is None or self.proc.stdout is None:
            return None

        if timeout is None:
            # No timeout - wait indefinitely
            while True:
                rlist, _, _ = select.select([self.proc.stdout], [], [])
                if rlist:
                    line = self.proc.stdout.readline()
                    if not line:  # EOF
                        return None
                    return line.rstrip("\n")
        else:
            # With timeout
            end = time.time() + timeout
            while time.time() < end:
                rlist, _, _ = select.select([self.proc.stdout], [], [], max(0.01, end - time.time()))
                if rlist:
                    line = self.proc.stdout.readline()
                    if not line:  # EOF
                        return None
                    return line.rstrip("\n")
            return None

    def _hard_kill(self):
        if self.proc and self.proc.poll() is None:
            logger.debug(f"Hard killing worker process (PID: {self.proc.pid})")
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception as e:
                logger.warning(f"Failed to terminate process group: {e}; falling back to terminate()")
                with contextlib.suppress(Exception):
                    self.proc.terminate()
            t0 = time.time()
            while self.proc.poll() is None and (time.time() - t0) < self.policy.kill_grace_s:
                time.sleep(0.01)
            if self.proc.poll() is None:
                logger.debug(f"Sending SIGKILL to worker process (PID: {self.proc.pid})")
                with contextlib.suppress(Exception):
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except Exception:
                        os.kill(self.proc.pid, signal.SIGKILL)
        self.proc = None

    def close(self):
        logger.debug("Closing worker process...")
        if self.proc and self.proc.poll() is None:
            try:
                logger.debug("Sending bye RPC to worker")
                self._send_rpc("bye", {})
                # ignore response; we're closing anyway
            except Exception as e:
                logger.debug(f"Bye RPC failed (expected): {e}")
                pass
        self._hard_kill()
        logger.debug("Worker process closed")

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    # ---------- JSON-RPC helpers ----------

    def _send_rpc(self, method: str, params: Dict[str, Any], timeout: Optional[float] = None) -> Any:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            logger.error("Worker not running, cannot send RPC")
            raise Tempo2Crashed("worker not running")

        self._id += 1
        rid = self._id
        # Only log debug for non-get methods to reduce noise
        if method != "get":
            logger.debug(f"Sending RPC {method} (id: {rid})")

        frame = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
            "params_b64": _b64_dumps_py(params),
        }
        line = json.dumps(frame) + "\n"

        try:
            with self._rpc_lock:
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
        except Exception as e:
            logger.error(f"Failed to send RPC {method}: {e}")
            self._hard_kill()
            raise Tempo2Crashed(f"send failed: {e!r}")

        # Wait for response
        t = self.policy.call_timeout_s if timeout is None else timeout
        if t is None:
            logger.debug(f"Waiting for RPC {method} response (no timeout)")
        else:
            logger.debug(f"Waiting for RPC {method} response (timeout: {t}s)")
        resp_line = self._readline_with_timeout(t)
        if resp_line is None:
            if t is None:
                logger.error(f"RPC {method} failed - worker disconnected")
                self._hard_kill()
                raise Tempo2Crashed(f"RPC '{method}' failed - worker disconnected")
            else:
                logger.error(f"RPC {method} timed out after {t}s")
                self._hard_kill()
                raise Tempo2Timeout(f"RPC '{method}' timed out")

        try:
            resp = json.loads(resp_line)
        except Exception as e:
            logger.error(f"Failed to parse RPC {method} response: {e}")
            self._hard_kill()
            raise Tempo2ProtocolError(f"malformed response: {resp_line!r}")

        if resp.get("id") != rid:
            logger.error(f"RPC {method} id mismatch: expected {rid}, got {resp.get('id')}")
            self._hard_kill()
            raise Tempo2ProtocolError(f"mismatched id in response: {resp.get('id')} vs {rid}")

        if "error" in resp and resp["error"] is not None:
            err = resp["error"]
            msg = err.get("message", "error")
            data = err.get("data", "")
            logger.error(f"RPC {method} failed: {msg}")
            tail = ""
            if getattr(self, "_log_buf", None) and (self.policy.include_stderr_tail_in_errors or 0) > 0:
                try:
                    tail_lines = list(self._log_buf)[-self.policy.include_stderr_tail_in_errors :]
                    if tail_lines:
                        blob = "\n".join(tail_lines)
                        max_bytes = 16384
                        if len(blob) > max_bytes:
                            blob = blob[-max_bytes:]
                        tail = "\n--- tempo2 stderr (tail) ---\n" + blob
                except Exception:
                    tail = ""
            raise Tempo2Error(f"{msg}\n{data}{tail}")

        # Only log debug for non-get methods to reduce noise
        if method != "get":
            logger.debug(f"RPC {method} completed successfully")
        result_b64 = resp.get("result_b64", None)
        return _b64_loads_py(result_b64) if result_b64 is not None else None

    # Public RPCs
    def ctor(self, kwargs: Dict[str, Any], preload_residuals: bool):
        logger.info(f"Constructing tempopulsar with kwargs: {kwargs}")
        logger.info(f"Preload residuals: {preload_residuals}")
        return self._send_rpc("ctor", {"kwargs": kwargs, "preload_residuals": preload_residuals})

    def get(self, name: str):
        logger.debug(f"Getting attribute: {name}")
        return self._send_rpc("get", {"name": name})

    def get_kind(self, name: str):
        """Return (kind, value) where kind in {"value","callable","missing"}.
        For legacy workers without get-kind support, assumes raw value => ("value", value).
        """
        resp = self._send_rpc("get", {"name": name})
        if isinstance(resp, dict) and "kind" in resp:
            return (resp.get("kind"), resp.get("value"))
        return ("value", resp)

    def dir(self):
        """Return list of public attribute names."""
        return self._send_rpc("dir", {})

    def set(self, name: str, value: Any):
        logger.debug(f"Setting attribute: {name}")
        return self._send_rpc("set", {"name": name, "value": value})

    def call(self, name: str, args=(), kwargs=None):
        logger.debug(f"Calling method: {name} with args={args}, kwargs={kwargs}")
        return self._send_rpc("call", {"name": name, "args": tuple(args), "kwargs": dict(kwargs or {})})

    def setitem(self, name: str, index, value: Any):
        logger.debug(f"Setting array slice: {name}[{index}] = <value>")
        return self._send_rpc("setitem", {"name": name, "index": index, "value": value})

    def get_slice(self, name: str, index):
        logger.debug(f"Getting array slice: {name}[{index}]")
        return self._send_rpc("get_slice", {"name": name, "index": index})

    def rss(self) -> Optional[int]:
        try:
            logger.debug("Getting worker RSS memory usage")
            return self._send_rpc("rss", {})
        except Exception as e:
            logger.warning(f"Failed to get RSS: {e}")
            return None

    def logs(self, tail: int = 500) -> str:
        try:
            return "\n".join(list(self._log_buf)[-max(0, tail) :])
        except Exception:
            return ""


# ------------------------- Command resolution (env_name) -------------------- #


def _venv_search_paths(env_name: str) -> List[Path]:
    """Return candidate Python executable paths for a given venv name."""
    return [
        Path.home() / ".venvs" / env_name / "bin" / "python",
        Path.home() / "venvs" / env_name / "bin" / "python",
        Path.home() / ".virtualenvs" / env_name / "bin" / "python",
        Path.cwd() / env_name / "bin" / "python",
        Path.cwd() / ".venv" / "bin" / "python",
        Path("/opt/venvs") / env_name / "bin" / "python",
        Path("/opt/virtualenvs") / env_name / "bin" / "python",
        Path("/usr/local/venvs") / env_name / "bin" / "python",
        Path("/home") / "venvs" / env_name / "bin" / "python",
    ]


def _detect_environment_type(env_name: str) -> str:
    """Return "conda:<tool>", "venv", "arch", "python", or "unknown"."""
    if env_name.startswith("python:"):
        return "python"

    for tool in ("conda", "mamba", "micromamba"):
        try:
            r = subprocess.run(
                [tool, "run", "-n", env_name, "python", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                return "conda:" + tool
        except Exception:
            pass

    for p in _venv_search_paths(env_name):
        if p.exists():
            return "venv"

    if env_name in ("arch", "rosetta", "system"):
        return "arch"

    return "unknown"


def _find_venv_python_path(env_name: str) -> Optional[str]:
    """Find the Python executable for a venv by name, or None."""
    for p in _venv_search_paths(env_name):
        if p.exists():
            return str(p)
    return None


def _resolve_worker_cmd(env_name: Optional[str]) -> Tuple[List[str], bool]:
    """
    Build the subprocess command to run the worker and whether we require x86_64.
    Returns (cmd, require_x86_64)
    """

    # Base invocation that runs this file in worker mode.
    # Find the src directory dynamically
    current_file = Path(__file__).resolve()
    src_dir = current_file.parent.parent  # Go up from libstempo/sandbox.py to src/
    src_path = str(src_dir)

    def python_to_worker_cmd(python_exe: str) -> List[str]:
        """Build command to run worker with given Python executable."""
        return [
            python_exe,
            "-c",
            (
                f"import os, sys; "
                f"os.environ['TEMPO2_SANDBOX_PROTO_FD'] = str(os.dup(1)); "
                f"os.dup2(2, 1); "
                f"sys.stdout = sys.stderr; "
                f"sys.path.insert(0, '{src_path}'); "
                f"import libstempo.sandbox as m; "
                f"m._worker_stdio_main()"
            ),
        ]

    arch_prefix_env = os.environ.get("TEMPO2_SANDBOX_WORKER_ARCH_PREFIX", "").strip()
    require_x86_64 = False

    # No env_name -> use current python (no Rosetta)
    if env_name is None:
        py = sys.executable
        return (python_to_worker_cmd(py), False)

    # Explicit python path
    if env_name.startswith("python:"):
        py = env_name.split(":", 1)[1]
        return (python_to_worker_cmd(py), False)

    etype = _detect_environment_type(env_name)

    # conda/mamba/micromamba
    if etype.startswith("conda:"):
        tool = etype.split(":", 1)[1]
        cmd = [tool, "run", "-n", env_name] + python_to_worker_cmd("python")
        # Choosing to require x86_64 only if user *explicitly* asks via arch prefix or env_name == "arch"
        require_x86_64 = "arch" in env_name.lower()
        if arch_prefix_env:
            cmd = arch_prefix_env.split() + cmd
            require_x86_64 = True
        return (cmd, require_x86_64)

    # venv
    if etype == "venv":
        py = _find_venv_python_path(env_name)
        if not py:
            raise Tempo2Error(f"virtualenv '{env_name}' not found in common locations")
        cmd = python_to_worker_cmd(py)
        if arch_prefix_env:
            cmd = arch_prefix_env.split() + cmd
            require_x86_64 = True
        return (cmd, require_x86_64)

    # system Rosetta
    if etype == "arch":
        # try system python (python3 or python)
        py = shutil.which("python3") or shutil.which("python")
        if not py:
            raise Tempo2Error("could not find system python for arch mode")
        arch = arch_prefix_env.split() if arch_prefix_env else ["arch", "-x86_64"]
        require_x86_64 = True
        return (arch + python_to_worker_cmd(py), require_x86_64)

    raise Tempo2Error(
        f"Environment '{env_name}' not found. " "Use a conda env name, a venv name, 'arch', or 'python:/abs/python'."
    )


# ------------------------------ Public proxy ------------------------------- #


@dataclasses.dataclass
class _State:
    """Internal state tracking for tempopulsar proxy instances."""

    created_at: float
    calls_ok: int
    # State cache for crash recovery
    param_cache: Dict[str, Dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )  # {'RAJ': {'val': 5.016, 'fit': True}, ...}
    array_cache: Dict[str, Any] = dataclasses.field(
        default_factory=dict
    )  # {'stoas': modified_array, 'toaerrs': modified_array}
    # Crash recovery statistics
    crash_count: int = 0
    last_crash_at: Optional[float] = None


class tempopulsar:
    """
    Proxy for libstempo.tempopulsar living inside an isolated subprocess.

    This class provides a drop-in replacement for libstempo.tempopulsar that runs
    in a separate process to prevent crashes from affecting the main kernel.
    All constructor arguments are forwarded to libstempo.tempopulsar unchanged.

    The proxy automatically handles:
    - Worker process lifecycle management
    - Automatic retry on failures
    - Worker recycling based on age, call count, or memory usage
    - JSON-RPC communication over stdio

    Args:
        env_name: Environment name (conda env or venv name, 'arch', or 'python:/abs/python').
                 If None (default), uses the current Python environment.
        policy: Optional Policy instance to configure worker behavior
        **kwargs: Additional arguments passed to libstempo.tempopulsar

    Example:
        >>> psr = tempopulsar(parfile="J1713.par", timfile="J1713.tim", dofit=False)
        >>> residuals = psr.residuals()
        >>> design_matrix = psr.designmatrix()
    """

    __slots__ = (
        "_policy",
        "_wp",
        "_state",
        "_ctor_kwargs",
        "_env_name",
        "_require_x86",
    )

    def __init__(self, env_name: Optional[str] = None, policy: Optional[Policy] = None, **kwargs):
        self._policy: Policy = policy if isinstance(policy, Policy) else Policy()
        self._env_name = env_name
        self._ctor_kwargs = dict(kwargs)
        self._wp: Optional[_WorkerProc] = None
        self._state = _State(created_at=time.time(), calls_ok=0)
        self._require_x86 = False

        logger.info(f"Creating tempopulsar with env_name='{env_name}', kwargs={self._ctor_kwargs}")
        logger.info(f"Using policy: ctor_retry={self._policy.ctor_retry}, ctor_backoff={self._policy.ctor_backoff}s")
        self._construct_with_retries()

    # --------------- construction / reconstruction with retries --------------- #

    def _construct_with_retries(self):
        logger.info(f"Starting construction with {self._policy.ctor_retry + 1} total attempts")

        # Fast-fail on missing input files to avoid noisy retries
        try:
            parfile = self._ctor_kwargs.get("parfile")
            timfile = self._ctor_kwargs.get("timfile")
            cwd = os.getcwd()
            if parfile and not Path(parfile).exists():
                raise Tempo2ConstructorFailed(
                    f"parfile not found: {parfile} (cwd: {cwd}). Provide an absolute path or correct relative path."
                )
            if timfile and not Path(timfile).exists():
                raise Tempo2ConstructorFailed(
                    f"timfile not found: {timfile} (cwd: {cwd}). Provide an absolute path or correct relative path."
                )
        except Tempo2ConstructorFailed:
            # Re-raise to surface a clean single error without retries
            raise
        except Exception:
            # Do not block construction for unexpected preflight errors
            logger.debug("Preflight path check skipped due to unexpected error:", exc_info=True)

        # Proactive TOA counting to avoid "Too many TOAs" errors
        if self._policy.auto_nobs_retry:
            self._proactive_nobs_setup()

        last_exc: Optional[Exception] = None
        for attempt in range(1 + self._policy.ctor_retry):
            logger.info(f"Construction attempt {attempt + 1}/{self._policy.ctor_retry + 1}")
            try:
                cmd, require_x86 = _resolve_worker_cmd(self._env_name)
                self._require_x86 = require_x86
                logger.debug(f"Resolved worker command: {' '.join(cmd)}")
                logger.debug(f"Require x86_64: {require_x86}")

                self._wp = _WorkerProc(self._policy, cmd, require_x86_64=require_x86)
                # ctor on the worker (libstempo.tempopulsar)
                logger.info("Calling constructor on worker...")
                self._wp.ctor(self._ctor_kwargs, preload_residuals=self._policy.preload_residuals)
                self._state.created_at = time.time()
                self._state.calls_ok = 0
                logger.info(f"Construction successful on attempt {attempt + 1}")

                # Restore state after successful reconstruction
                self._restore_state_after_reconstruction()
                return
            except Exception as e:
                logger.warning(f"Construction attempt {attempt + 1} failed: {e}")
                last_exc = e

                # Record crash if this is a retry (not the first attempt)
                if attempt > 0:
                    self._record_crash()

                # If it's a file-not-found style error, fail fast without retries
                msg = str(e)
                if any(
                    t in msg
                    for t in ("Cannot find parfile", "Cannot find timfile", "parfile not found", "timfile not found")
                ):
                    logger.error("Input file missing; not retrying constructor.")
                    break
                # kill and retry
                try:
                    if self._wp:
                        logger.debug("Cleaning up failed worker")
                        self._wp.close()
                except Exception as cleanup_e:
                    logger.warning(f"Cleanup failed: {cleanup_e}")
                    pass
                self._wp = None
                if attempt < self._policy.ctor_retry:  # Don't sleep after last attempt
                    logger.info(f"Waiting {self._policy.ctor_backoff}s before retry...")
                    time.sleep(self._policy.ctor_backoff)
        logger.error(f"All construction attempts failed. Last error: {last_exc}")
        raise Tempo2ConstructorFailed(f"tempopulsar ctor failed after retries: {last_exc}")

    def _proactive_nobs_setup(self):
        """Proactively count TOAs and add nobs parameter if needed to avoid 'Too many TOAs' errors."""
        try:
            timfile = self._ctor_kwargs.get("timfile")
            if not timfile:
                logger.debug("No timfile specified, skipping proactive nobs setup")
                return

            timfile_path = Path(timfile)
            if not timfile_path.exists():
                logger.warning(f"TIM file does not exist: {timfile_path}")
                return

            logger.info(f"Proactively counting TOAs in {timfile_path}")
            analyzer = TimFileAnalyzer()
            toa_count = analyzer.count_toas(timfile_path)

            if toa_count > self._policy.nobs_threshold:
                maxobs_with_margin = int(toa_count * self._policy.nobs_safety_margin)
                self._ctor_kwargs["maxobs"] = maxobs_with_margin
                logger.info(
                    f"Proactively added maxobs={maxobs_with_margin} parameter "
                    f"(TOAs: {toa_count}, threshold: {self._policy.nobs_threshold}, "
                    f"margin: {self._policy.nobs_safety_margin})"
                )
            else:
                logger.debug(
                    f"TOA count {toa_count} below threshold {self._policy.nobs_threshold}, no maxobs parameter needed"
                )

        except Exception as e:
            logger.warning(f"Proactive nobs setup failed: {e}")
            # Don't raise - this is just optimization, construction should still work

    # ----------------------------- state management ----------------------------- #

    def _capture_param_state(self, param_name: str, field: str, value: Any) -> None:
        """Capture parameter state for crash recovery."""
        if param_name not in self._state.param_cache:
            self._state.param_cache[param_name] = {}
        self._state.param_cache[param_name][field] = value
        logger.debug(f"Captured param state: {param_name}.{field} = {value}")

    def _capture_array_state(self, array_name: str, value: Any) -> None:
        """Capture array state for crash recovery."""
        self._state.array_cache[array_name] = value
        logger.debug(f"Captured array state: {array_name}")

    def _restore_state_after_reconstruction(self) -> None:
        """Restore parameter values, fit flags, and array modifications after worker reconstruction."""
        if not self._state.param_cache and not self._state.array_cache:
            logger.debug("No state to restore")
            return

        logger.info(f"Restoring state: {len(self._state.param_cache)} params, {len(self._state.array_cache)} arrays")

        # Restore parameter values and fit flags
        for param_name, param_state in self._state.param_cache.items():
            for field, value in param_state.items():
                try:
                    self._wp.set(f"{param_name}.{field}", value)
                    logger.debug(f"Restored param: {param_name}.{field} = {value}")
                except Exception as e:
                    logger.warning(f"Failed to restore param {param_name}.{field}: {e}")

        # Restore array modifications
        for array_name, array_data in self._state.array_cache.items():
            try:
                self._wp.setitem(array_name, slice(None), array_data)
                logger.debug(f"Restored array: {array_name}")
            except Exception as e:
                logger.warning(f"Failed to restore array {array_name}: {e}")

        logger.info("State restoration completed")

    def _record_crash(self) -> None:
        """Record crash statistics."""
        self._state.crash_count += 1
        self._state.last_crash_at = time.time()
        logger.info(f"Worker crash recorded (total crashes: {self._state.crash_count})")

    def get_crash_stats(self) -> Dict[str, Any]:
        """Get crash recovery statistics."""
        return {
            "crash_count": self._state.crash_count,
            "last_crash_at": self._state.last_crash_at,
            "worker_age_s": time.time() - self._state.created_at if self._wp else None,
            "calls_since_creation": self._state.calls_ok,
        }

    # ----------------------------- recycling policy --------------------------- #

    def _should_recycle(self) -> bool:
        if self._wp is None:
            logger.debug("Should recycle: worker is None")
            return True

        age = time.time() - self._state.created_at

        # Check age limit (if set)
        if self._policy.max_age_s is not None and age > self._policy.max_age_s:
            logger.info(f"Should recycle: worker age {age:.1f}s exceeds max_age_s {self._policy.max_age_s}")
            return True

        # Check call limit (if set)
        if self._policy.max_calls_per_worker is not None and self._state.calls_ok >= self._policy.max_calls_per_worker:
            logger.info(
                f"Should recycle: calls_ok {self._state.calls_ok} exceeds "
                f"max_calls_per_worker {self._policy.max_calls_per_worker}"
            )
            return True

        # Check RSS limit (if set)
        if self._policy.rss_soft_limit_mb is not None:
            rss = self._wp.rss()
            if rss and rss > self._policy.rss_soft_limit_mb:
                logger.info(f"Should recycle: RSS {rss}MB exceeds limit {self._policy.rss_soft_limit_mb}MB")
                return True

        logger.debug(f"Worker still healthy: age={age:.1f}s, calls={self._state.calls_ok}")
        return False

    def _recycle(self):
        logger.info("Recycling worker (creating new one)")
        if self._wp is not None:
            logger.debug("Closing old worker")
            with contextlib.suppress(Exception):
                self._wp.close()
            self._wp = None
        logger.debug("Constructing new worker")
        self._construct_with_retries()

    # ---------------------------- RPC convenience ----------------------------- #

    def _get_kind_safe(self, name: str):
        """Call get_kind with automatic crash recovery and recycling."""
        if self._wp is None:
            self._construct_with_retries()
        if self._should_recycle():
            self._recycle()
        try:
            return self._wp.get_kind(name)
        except (Tempo2Crashed, Tempo2Timeout) as e:
            logger.warning(f"get_kind({name!r}) failed: {e}; recycling worker")
            self._record_crash()
            self._recycle()
            return self._wp.get_kind(name)

    def _dispatch_rpc(self, call: str, payload: Dict[str, Any]) -> Any:
        """Send a single RPC call to the worker. No retry logic."""
        assert self._wp is not None
        if call == "get":
            return self._wp.get(payload["name"])
        elif call == "set":
            return self._wp.set(payload["name"], payload["value"])
        elif call == "call":
            return self._wp.call(payload["name"], payload.get("args", ()), payload.get("kwargs", {}))
        elif call == "setitem":
            return self._wp.setitem(payload["name"], payload.get("index"), payload.get("value"))
        else:
            raise Tempo2ProtocolError(f"unknown call {call}")

    def _rpc(self, call: str, **payload):
        if self._wp is None:
            logger.debug("Worker is None, constructing...")
            self._construct_with_retries()
        if self._should_recycle():
            logger.info("Worker needs recycling")
            self._recycle()
        try:
            out = self._dispatch_rpc(call, payload)
            self._state.calls_ok += 1
            logger.debug(f"RPC {call} successful, total calls: {self._state.calls_ok}")
            return out
        except (Tempo2Timeout, Tempo2Crashed) as e:
            logger.warning(f"RPC {call} failed with {type(e).__name__}: {e}")
            logger.info("Attempting automatic worker recycle and retry")
            self._record_crash()
            self._recycle()
            out = self._dispatch_rpc(call, payload)
            self._state.calls_ok += 1
            logger.info(f"RPC {call} succeeded after recycle, total calls: {self._state.calls_ok}")
            return out

    # ------------------------ Attribute proxying magic ------------------------ #

    def __getattr__(self, name: str):
        # Filter out IPython-specific attributes to prevent infinite loops
        if name.startswith("_ipython_") or name in {
            "_ipython_canary_method_should_not_exist_",
            "_repr_mimebundle_",
            "_repr_html_",
            "_repr_json_",
            "_repr_latex_",
            "_repr_png_",
            "_repr_jpeg_",
            "_repr_svg_",
            "_repr_pdf_",
        }:
            logger.debug(f"Filtering out IPython attribute: {name}")
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        def _remote_method(*args, **kwargs):
            return self._rpc("call", name=name, args=args, kwargs=kwargs)

        kind, payload = self._get_kind_safe(name)
        if kind == "value":
            # If worker returned a safe libstempo param marker, expose a proxy
            if isinstance(payload, dict) and payload.get("__libstempo_param__"):
                return _ParamProxy(self, name)
            return payload
        if kind == "callable":
            return _remote_method
        if kind == "missing":
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        raise Tempo2ProtocolError(f"unexpected get-kind '{kind}' for attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        if name in tempopulsar.__slots__:
            return object.__setattr__(self, name, value)

        # Capture state for crash recovery
        self._capture_array_state(name, value)

        _ = self._rpc("set", name=name, value=value)
        return None

    def __dir__(self):
        """Return a list of available attributes for dir() function."""
        try:
            return list(self._wp.dir())
        except Exception:
            pass

        # Fallback: return a basic set of common tempopulsar attributes
        return [
            "name",
            "nobs",
            "stoas",
            "toaerrs",
            "freqs",
            "ndim",
            "residuals",
            "designmatrix",
            "toas",
            "fit",
            "fittable",
            "identifiable",
            "vals",
            "errs",
            "pars",
            "flags",
            "flagvals",
            "savepar",
            "savetim",
            "chisq",
            "rms",
            "ssbfreqs",
            "logs",
        ]

    # Explicit helpers for common call shapes
    def residuals(self, **kwargs):
        return self._rpc("call", name="residuals", kwargs=kwargs)

    def designmatrix(self, **kwargs):
        return self._rpc("call", name="designmatrix", kwargs=kwargs)

    def toas(self, **kwargs):
        return self._rpc("call", name="toas", kwargs=kwargs)

    def fit(self, **kwargs):
        return self._rpc("call", name="fit", kwargs=kwargs)

    def logs(self, tail: int = 500) -> str:
        return self._wp.logs(tail) if self._wp else ""

    # Mapping-style access to parameters, proxied to libstempo
    def __getitem__(self, key: str):
        return _ParamProxy(self, key)

    def __setitem__(self, key: str, value: Any):
        raise TypeError("Direct assignment to parameters is not supported; set fields like psr['RAJ'].val = x")

    # Expose array-like attributes as write-through proxies
    @property
    def stoas(self):
        return _ArrayProxy(self, "stoas")

    @property
    def toaerrs(self):
        return _ArrayProxy(self, "toaerrs")

    @property
    def freqs(self):
        return _ArrayProxy(self, "freqs")


class _ParamProxy:
    __slots__ = ("_parent", "_name")

    def __init__(self, parent: tempopulsar, name: str) -> None:
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, attr: str):
        kind, payload = self._parent._get_kind_safe(f"{self._name}.{attr}")
        if kind == "value":
            return payload
        if kind == "callable":
            # Expose a callable that routes via call with dotted name
            def _remote_method(*args, **kwargs):
                return self._parent._rpc("call", name=f"{self._name}.{attr}", args=args, kwargs=kwargs)

            return _remote_method
        if kind == "missing":
            raise AttributeError(f"'{self._name}' has no attribute '{attr}'")
        raise Tempo2ProtocolError(f"unexpected get-kind '{kind}' for attribute '{self._name}.{attr}'")

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr in _ParamProxy.__slots__:
            return object.__setattr__(self, attr, value)

        # Capture parameter state for crash recovery
        self._parent._capture_param_state(self._name, attr, value)

        _ = self._parent._rpc("set", name=f"{self._name}.{attr}", value=value)
        return None

    def __repr__(self) -> str:
        try:
            v = self.__getattr__("val")
            e = self.__getattr__("err")
            return f"<ParamProxy {self._name}: val={v!r}, err={e!r}>"
        except Exception:
            return f"<ParamProxy {self._name}>"


class _ArrayProxy:
    __slots__ = ("_parent", "_name")

    def __init__(self, parent: tempopulsar, name: str) -> None:
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_name", name)

    # numpy reads
    def __array__(self, dtype=None):
        import numpy as _np

        kind, payload = self._parent._get_kind_safe(self._name)
        if kind == "value":
            arr = payload
        elif kind == "callable":
            # Arrays are not callable; treat as empty
            arr = []
        else:
            arr = []

        a = _np.asarray(arr)
        if dtype is not None:
            a = a.astype(dtype, copy=False)
        return a

    def __len__(self):
        a = self.__array__()
        return int(a.shape[0]) if getattr(a, "ndim", 1) > 0 else 1

    @property
    def shape(self):
        return self.__array__().shape

    @property
    def dtype(self):
        return self.__array__().dtype

    # python indexing
    def __getitem__(self, idx):
        return self._parent._wp.get_slice(self._name, idx)

    def __setitem__(self, idx, value):
        # Capture array state for crash recovery
        # For full array replacement (slice(None)), capture the entire array
        if idx == slice(None):
            self._parent._capture_array_state(self._name, value)
        else:
            # For partial updates, we need to get the current array and apply the change
            # This is more complex, so for now we'll capture the full array after the change
            try:
                current_array = self.__array__()
                if hasattr(current_array, "copy"):
                    new_array = current_array.copy()
                    new_array[idx] = value
                    self._parent._capture_array_state(self._name, new_array)
            except Exception:
                # If we can't capture the state, continue anyway
                pass

        _ = self._parent._rpc("setitem", name=self._name, index=idx, value=value)
        return None

    def __repr__(self) -> str:
        try:
            return repr(self.__array__())
        except Exception:
            return f"<_ArrayProxy {self._name}>"

    def __str__(self) -> str:
        try:
            return str(self.__array__())
        except Exception:
            return self.__repr__()

    # Array arithmetic so that e.g. psr.toaerrs * 1e-6 works in plots
    def __mul__(self, other):
        return self.__array__() * other

    def __rmul__(self, other):
        return other * self.__array__()

    def __add__(self, other):
        return self.__array__() + other

    def __radd__(self, other):
        return other + self.__array__()

    def __sub__(self, other):
        return self.__array__() - other

    def __rsub__(self, other):
        return other - self.__array__()

    def __truediv__(self, other):
        return self.__array__() / other

    def __rtruediv__(self, other):
        return other / self.__array__()

    # Delegate unknown attributes/methods to the numpy array
    def __getattr__(self, name: str):
        arr = self.__array__()
        return getattr(arr, name)


# -------------------------- Bulk loader (optional) -------------------------- #


@dataclass
class LoadReport:
    par: str
    tim: Optional[str]
    attempts: int
    ok: bool
    error: Optional[str] = None
    retried: bool = False


def load_many(
    pairs: Iterable[Tuple[str, Optional[str]]],
    policy: Optional[Policy] = None,
    parallel: int = 8,
) -> Tuple[Dict[str, tempopulsar], Dict[str, LoadReport], List[LoadReport]]:
    """Bulk-load many pulsars with bounded parallelism.

    Each pulsar is constructed via ``tempopulsar()``, which already has its own
    internal retry logic (controlled by ``policy.ctor_retry``).  This function
    only provides parallelism; it does **not** add an extra retry layer.

    Returns (ok_by_name, retried_by_name, failed_list).
    """
    pol = policy if isinstance(policy, Policy) else Policy()
    pairs_list = list(pairs)
    logger.info(f"Starting bulk load of {len(pairs_list)} pulsars with {parallel} parallel workers")
    logger.info(f"Using policy: ctor_retry={pol.ctor_retry}, ctor_backoff={pol.ctor_backoff}s")

    def _one(par, tim):
        """Load a single pulsar (retries are handled inside tempopulsar)."""
        logger.debug(f"Loading pulsar: par={par}, tim={tim}")
        report = LoadReport(par=par, tim=tim, attempts=1, ok=False)
        try:
            psr = tempopulsar(parfile=par, timfile=tim, policy=pol)
            name = getattr(psr, "name")
            report.ok = True
            logger.info(f"Successfully loaded {name}")
            return ("ok", name, psr, report)
        except Exception as e:
            logger.warning(f"Failed to load {par}: {e}")
            report.error = f"{e.__class__.__name__}: {e}"
            logger.error(f"Failed to load {par}: {e}")
            return ("fail", None, None, report)

    ok: Dict[str, tempopulsar] = {}
    retried: Dict[str, LoadReport] = {}
    failed: List[LoadReport] = []

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futs = {ex.submit(_one, par, tim): (par, tim) for (par, tim) in pairs_list}
        for fut in as_completed(futs):
            kind, name, psr, report = fut.result()
            if kind == "ok":
                ok[name] = psr
            else:
                failed.append(report)

    logger.info(
        f"Bulk load completed: {len(ok)} successful, {len(failed)} failed"
    )
    return ok, retried, failed


# ------------------------------- Quick helpers ------------------------------ #


def configure_logging(level: str = "INFO", log_file: Optional[str] = None, enable_console: bool = True):
    """
    Configure standard logging for the sandbox.

    Args:
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional file path to log to
        enable_console: Whether to log to console
    """
    # Get the sandbox logger
    sandbox_logger = logging.getLogger(__name__)

    # Clear existing handlers
    sandbox_logger.handlers.clear()

    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    sandbox_logger.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | tempo2_sandbox | %(message)s", datefmt="%H:%M:%S")

    # Add console handler if requested
    if enable_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        sandbox_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        sandbox_logger.addHandler(file_handler)

    logger.info(f"Logging configured: level={level}, console={enable_console}, file={log_file}")


def setup_instructions(env_name: str = "tempo2_intel"):
    """Print setup instructions for creating a tempo2 environment.

    This is a utility function to help users set up their environment
    for using the sandbox with different Python environments.
    """
    print("Setup instructions for environment '{}':".format(env_name))
    print("\n1. Conda (recommended):")
    print(f"   conda create -n {env_name} python=3.11")
    print(f"   conda activate {env_name}")
    print("   conda install -c conda-forge tempo2 libstempo")
    print(f'   # then just: psr = tempopulsar(..., env_name="{env_name}")')
    print("\n2. Virtual Environment (Rosetta):")
    print(f"   arch -x86_64 /usr/local/bin/python3 -m venv ~/.venvs/{env_name}")
    print(f"   source ~/.venvs/{env_name}/bin/activate")
    print("   pip install tempo2 libstempo")
    print(f'   # then just: psr = tempopulsar(..., env_name="{env_name}")')
    print("\n3. System Python with Rosetta:")
    print("   # Install Intel Python first (or use system one under arch).")
    print('   # You can force Rosetta via TEMPO2_SANDBOX_WORKER_ARCH_PREFIX="arch -x86_64"')
    print('   # then: psr = tempopulsar(..., env_name="arch")')


def detect_and_guide(env_name: str):
    """Detect environment type and provide guidance for setup.

    This is a utility function to help users understand what type
    of environment they have and how to use it with the sandbox.
    """
    et = _detect_environment_type(env_name)
    print(f"Environment detection for '{env_name}': {et}")
    if et.startswith("conda:"):
        print("✅ Conda env detected; just use env_name as given.")
    elif et == "venv":
        p = _find_venv_python_path(env_name)
        if p:
            print(f"✅ venv detected at {p}")
        else:
            print("❌ venv name matched, but python path not resolved.")
    elif et == "arch":
        print("✅ Rosetta/system arch mode will be used.")
    elif et == "python":
        print("✅ Using explicit Python path.")
    else:
        print("❌ Not found. Use conda env name, venv name, 'arch', or 'python:/abs/python'.")


# ------------------------------ Module runner ------------------------------- #

if __name__ == "__main__":
    # If executed directly, act as worker (useful for manual debugging):
    _worker_stdio_main()
