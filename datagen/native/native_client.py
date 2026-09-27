from __future__ import annotations

import ctypes
import json
import os
import time
from typing import Any


GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
ERROR_PIPE_BUSY = 231
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def _kernel32() -> Any:
    if os.name != "nt":
        raise RuntimeError(
            "The Guitar Pro native renderer is only available on Windows"
        )
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _open_pipe(pipe_name: str, timeout: float) -> int:
    kernel32 = _kernel32()
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.WaitNamedPipeW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handle = kernel32.CreateFileW(
            pipe_name,
            GENERIC_READ | GENERIC_WRITE,
            0,
            None,
            OPEN_EXISTING,
            0,
            None,
        )
        if handle not in (None, 0, INVALID_HANDLE_VALUE):
            return int(handle)
        if ctypes.get_last_error() == ERROR_PIPE_BUSY:
            kernel32.WaitNamedPipeW(pipe_name, 2_000)
        else:
            time.sleep(0.25)
    raise TimeoutError(f"Could not connect to Guitar Pro export pipe {pipe_name}")


def _close_pipe(handle: int) -> None:
    kernel32 = _kernel32()
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle(ctypes.c_void_p(handle))


def _send_recv(
    handle: int,
    request: dict[str, Any],
    *,
    response_timeout: float,
) -> dict[str, Any]:
    kernel32 = _kernel32()
    kernel32.WriteFile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p,
    ]
    kernel32.ReadFile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p,
    ]
    kernel32.PeekNamedPipe.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.FlushFileBuffers.argtypes = [ctypes.c_void_p]
    payload = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
    written = ctypes.c_ulong()
    if not kernel32.WriteFile(
        ctypes.c_void_p(handle), payload, len(payload), ctypes.byref(written), None
    ):
        raise OSError(f"WriteFile failed: {ctypes.get_last_error()}")
    kernel32.FlushFileBuffers(ctypes.c_void_p(handle))

    deadline = time.monotonic() + max(0.1, float(response_timeout))
    available = ctypes.c_ulong()
    while True:
        if not kernel32.PeekNamedPipe(
            ctypes.c_void_p(handle),
            None,
            0,
            None,
            ctypes.byref(available),
            None,
        ):
            raise OSError(f"PeekNamedPipe failed: {ctypes.get_last_error()}")
        if available.value:
            break
        if time.monotonic() >= deadline:
            command = str(request.get("cmd") or "request")
            raise TimeoutError(
                f"Guitar Pro native export timed out waiting for {command}"
            )
        time.sleep(0.05)

    buffer = ctypes.create_string_buffer(65_536)
    read = ctypes.c_ulong()
    if not kernel32.ReadFile(
        ctypes.c_void_p(handle), buffer, len(buffer) - 1, ctypes.byref(read), None
    ):
        raise OSError(f"ReadFile failed: {ctypes.get_last_error()}")
    response = buffer.raw[: read.value].decode("utf-8", errors="replace").strip()
    if not response:
        raise OSError("Guitar Pro export pipe returned an empty response")
    return dict(json.loads(response))


class NativeExportClient:
    def __init__(
        self,
        pipe_name: str,
        timeout: float = 10.0,
        *,
        response_timeout: float = 900.0,
    ) -> None:
        self._handle: int | None = _open_pipe(pipe_name, timeout)
        self._response_timeout = max(0.1, float(response_timeout))

    def close(self) -> None:
        if self._handle is not None:
            _close_pipe(self._handle)
            self._handle = None

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._handle is None:
            raise RuntimeError("Guitar Pro export pipe is closed")
        try:
            return _send_recv(
                self._handle,
                request,
                response_timeout=self._response_timeout,
            )
        except TimeoutError:
            self.close()
            raise

    def export(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        *,
        layout_output: str | os.PathLike[str],
        official_score_output: str | os.PathLike[str],
        track_index: int = 0,
        capture_note_geometry: bool = True,
        display_mode: str = "tab",
    ) -> dict[str, Any]:
        if self._handle is None:
            raise RuntimeError("Guitar Pro export pipe is closed")
        if isinstance(track_index, bool) or not isinstance(track_index, int):
            raise TypeError("track_index must be an integer")
        if track_index < 0:
            raise ValueError("track_index must be nonnegative")
        if type(capture_note_geometry) is not bool:
            raise TypeError("capture_note_geometry must be a boolean")
        if display_mode not in {"tab", "notation", "both"}:
            raise ValueError(f"Invalid display mode: {display_mode}")
        request: dict[str, Any] = {
            "cmd": "export",
            "input": os.path.abspath(input_path).replace("\\", "/"),
            "output": os.path.abspath(output_path).replace("\\", "/"),
            "layout_output": os.path.abspath(layout_output).replace("\\", "/"),
            "official_score_output": os.path.abspath(official_score_output).replace(
                "\\", "/"
            ),
            "track_index": track_index,
            "tab_only": display_mode == "tab",
            "display_mode": display_mode,
        }
        if not capture_note_geometry:
            request["capture_note_geometry"] = False
        return self._request(request)

    def list_tracks(
        self,
        input_path: str | os.PathLike[str],
    ) -> dict[str, Any]:
        if self._handle is None:
            raise RuntimeError("Guitar Pro export pipe is closed")
        return self._request(
            {
                "cmd": "list_tracks",
                "input": os.path.abspath(input_path).replace("\\", "/"),
            }
        )

    def release_source(self) -> dict[str, Any]:
        if self._handle is None:
            raise RuntimeError("Guitar Pro export pipe is closed")
        return self._request({"cmd": "release_source"})

    def quit(self) -> dict[str, Any]:
        if self._handle is None:
            return {"ok": True}
        result = self._request({"cmd": "quit"})
        self.close()
        return result
