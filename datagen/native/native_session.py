from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .native_client import NativeExportClient
from .native_injector import find_process_all, inject_dll


DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000
STARTF_USESHOWWINDOW = 0x00000001
SW_HIDE = 0
PROCESS_TERMINATE = 0x0001
SYNCHRONIZE = 0x00100000
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
PROCESS_EXIT_TIMEOUT_MS = 10_000
PROCESS_STARTUP_DELAY_SECONDS = 20.0


class NativeExportSession:
    def __init__(
        self,
        *,
        runtime_dir: Path,
        show_app: bool = False,
        ready_timeout: int = 120,
        startup_delay_seconds: float = PROCESS_STARTUP_DELAY_SECONDS,
    ) -> None:
        self.runtime_dir = runtime_dir.resolve()
        self.executable = self.runtime_dir / "GuitarPro.exe"
        self.dll_path = (
            Path(__file__).resolve().parents[1]
            / "native-bin"
            / "gpomr_native_export.dll"
        )
        self.show_app = show_app
        self.ready_timeout = ready_timeout
        self.startup_delay_seconds = float(startup_delay_seconds)
        if self.startup_delay_seconds < 0:
            raise ValueError("startup_delay_seconds must be non-negative")
        self.process_id: int | None = None
        self.hide_stop: threading.Event | None = None
        self.client: NativeExportClient | None = None
        self.ready_event: int | None = None
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32.CreateEventW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_wchar_p,
        ]
        self.kernel32.CreateEventW.restype = ctypes.c_void_p
        self.kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.kernel32.WaitForSingleObject.restype = ctypes.c_ulong
        self.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel32.CloseHandle.restype = ctypes.c_bool
        self.kernel32.OpenProcess.argtypes = [
            ctypes.c_ulong,
            ctypes.c_bool,
            ctypes.c_ulong,
        ]
        self.kernel32.OpenProcess.restype = ctypes.c_void_p
        self.kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.kernel32.TerminateProcess.restype = ctypes.c_bool

    def _kill_process(self, process_id: int) -> None:
        handle = self.kernel32.OpenProcess(
            PROCESS_TERMINATE | SYNCHRONIZE,
            False,
            process_id,
        )
        if not handle:
            error = ctypes.get_last_error()
            if process_id not in find_process_all("GuitarPro.exe"):
                return
            raise OSError(
                f"OpenProcess failed for Guitar Pro PID {process_id}: {error}"
            )
        try:
            wait_result = self.kernel32.WaitForSingleObject(handle, 0)
            if wait_result == WAIT_OBJECT_0:
                return
            if wait_result != WAIT_TIMEOUT:
                raise OSError(
                    "WaitForSingleObject failed before terminating Guitar Pro "
                    f"PID {process_id}: result={wait_result} "
                    f"error={ctypes.get_last_error()}"
                )
            if not self.kernel32.TerminateProcess(handle, 0):
                raise OSError(
                    f"TerminateProcess failed for Guitar Pro PID {process_id}: "
                    f"{ctypes.get_last_error()}"
                )
            wait_result = self.kernel32.WaitForSingleObject(
                handle,
                PROCESS_EXIT_TIMEOUT_MS,
            )
            if wait_result == WAIT_TIMEOUT:
                raise TimeoutError(
                    f"Guitar Pro PID {process_id} did not terminate within "
                    f"{PROCESS_EXIT_TIMEOUT_MS} ms"
                )
            if wait_result != WAIT_OBJECT_0:
                raise OSError(
                    "WaitForSingleObject failed after terminating Guitar Pro "
                    f"PID {process_id}: result={wait_result} "
                    f"error={ctypes.get_last_error()}"
                )
        finally:
            self.kernel32.CloseHandle(handle)

    def _require_no_existing_processes(self) -> None:
        process_ids = find_process_all("GuitarPro.exe")
        if process_ids:
            formatted = ", ".join(str(value) for value in sorted(process_ids))
            raise RuntimeError(
                "Guitar Pro is already running; close it before exporting "
                f"(PID: {formatted})"
            )

    def _hide_windows(self) -> threading.Event:
        stop = threading.Event()
        process_id = self.process_id
        assert process_id is not None

        def loop() -> None:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            callback_type = ctypes.WINFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p
            )

            def callback(window, _) -> bool:
                window_process_id = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(window, ctypes.byref(window_process_id))
                if window_process_id.value == process_id:
                    user32.ShowWindow(window, SW_HIDE)
                return True

            callback_ref = callback_type(callback)
            while not stop.is_set():
                user32.EnumWindows(callback_ref, 0)
                stop.wait(0.1)

        threading.Thread(
            target=loop, name="guitarpro-window-hider", daemon=True
        ).start()
        return stop

    def __enter__(self) -> NativeExportSession:
        if not self.executable.is_file():
            raise FileNotFoundError(
                f"GuitarPro.exe not found at {self.executable}; "
                "pass its containing directory as runtime_dir or with --runtime"
            )
        if not self.dll_path.is_file():
            raise FileNotFoundError(self.dll_path)

        pipe_name = rf"\\.\pipe\gpomr_export_{os.getpid()}_{int(time.time() * 1000)}"
        ready_event_name = f"gpomr_export_ready_{os.getpid()}_{int(time.time() * 1000)}"
        self.ready_event = self.kernel32.CreateEventW(
            None, True, False, ready_event_name
        )
        if not self.ready_event:
            raise OSError(f"CreateEventW failed: {ctypes.get_last_error()}")

        try:
            self._require_no_existing_processes()
            environment = os.environ.copy()
            environment.update(
                {
                    "GPOMR_EXPORT_PIPE": pipe_name,
                    "GPOMR_EXPORT_READY_EVENT": ready_event_name,
                }
            )
            startup_info = None
            creation_flags = DETACHED_PROCESS
            if not self.show_app:
                startup_info = subprocess.STARTUPINFO()
                startup_info.dwFlags |= STARTF_USESHOWWINDOW
                startup_info.wShowWindow = SW_HIDE
                creation_flags |= CREATE_NO_WINDOW
            process = subprocess.Popen(
                [str(self.executable)],
                cwd=str(self.executable.parent),
                env=environment,
                startupinfo=startup_info,
                creationflags=creation_flags,
            )
            self.process_id = process.pid
            if not self.show_app:
                self.hide_stop = self._hide_windows()

            preloaded = os.environ.get("GPOMR_NATIVE_PRELOADED") == "1"
            # Direct injection must wait for the Windows loader lock. The Wine
            # preload route is already loaded before Guitar Pro signals ready.
            if not preloaded:
                time.sleep(self.startup_delay_seconds)
                inject_dll(
                    self.process_id,
                    self.dll_path,
                    process_handle=int(process._handle),
                )
            for _ in range(max(1, self.ready_timeout)):
                if self.kernel32.WaitForSingleObject(self.ready_event, 1_000) == 0:
                    break
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(
                        f"Guitar Pro exited before the export hook was ready ({exit_code})"
                    )
            else:
                raise RuntimeError(
                    "Guitar Pro native export hook did not report readiness"
                )

            self.client = NativeExportClient(
                pipe_name,
                timeout=45,
                response_timeout=900,
            )
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_: Any) -> None:
        if self.client is not None:
            try:
                self.client.quit()
            except Exception:
                self.client.close()
            self.client = None
        if self.hide_stop is not None:
            self.hide_stop.set()
            self.hide_stop = None
        if self.process_id is not None:
            self._kill_process(self.process_id)
            self.process_id = None
        if self.ready_event is not None:
            self.kernel32.CloseHandle(ctypes.c_void_p(self.ready_event))
            self.ready_event = None

    def export(
        self,
        input_path: Path,
        output_path: Path,
        *,
        layout_output: Path,
        official_score_output: Path,
        track_index: int = 0,
        display_mode: str = "tab",
    ) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Guitar Pro native export session is not open")
        return self.client.export(
            input_path,
            output_path,
            layout_output=layout_output,
            official_score_output=official_score_output,
            track_index=track_index,
            display_mode=display_mode,
            capture_note_geometry=display_mode == "tab",
        )

    def list_tracks(self, input_path: Path) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Guitar Pro native export session is not open")
        return self.client.list_tracks(input_path)

    def release_source(self) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Guitar Pro native export session is not open")
        return self.client.release_source()
