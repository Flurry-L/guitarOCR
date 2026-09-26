from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import time
from pathlib import Path


PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
PROCESS_CREATE_THREAD = 0x0002
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020
PROCESS_INJECTION_ACCESS = (
    PROCESS_CREATE_THREAD
    | PROCESS_QUERY_INFORMATION
    | PROCESS_VM_OPERATION
    | PROCESS_VM_READ
    | PROCESS_VM_WRITE
)
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258


def _windows_libraries():
    if os.name != "nt":
        raise RuntimeError("DLL injection is only available on Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wt.HANDLE
    kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    kernel32.OpenProcess.restype = wt.HANDLE
    kernel32.CloseHandle.argtypes = [wt.HANDLE]
    kernel32.CloseHandle.restype = wt.BOOL
    return kernel32, psapi


def find_process_all(name: str) -> list[int]:
    kernel32, psapi = _windows_libraries()
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    kernel32.OpenProcess.restype = wt.HANDLE
    kernel32.CloseHandle.argtypes = [wt.HANDLE]
    psapi.EnumProcesses.argtypes = [
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    psapi.GetModuleBaseNameW.argtypes = [
        wt.HANDLE,
        wt.HMODULE,
        ctypes.c_wchar_p,
        ctypes.c_ulong,
    ]
    process_ids = (ctypes.c_ulong * 4096)()
    returned_bytes = ctypes.c_ulong()
    if not psapi.EnumProcesses(
        process_ids, ctypes.sizeof(process_ids), ctypes.byref(returned_bytes)
    ):
        return []

    matches: list[int] = []
    count = returned_bytes.value // ctypes.sizeof(ctypes.c_ulong)
    for process_id in process_ids[:count]:
        if not process_id:
            continue
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, process_id
        )
        if not handle:
            continue
        try:
            executable_name = ctypes.create_unicode_buffer(260)
            if psapi.GetModuleBaseNameW(
                handle, None, executable_name, len(executable_name)
            ):
                if executable_name.value.casefold() == name.casefold():
                    matches.append(int(process_id))
        finally:
            kernel32.CloseHandle(handle)
    return matches


def _enable_debug_privilege() -> None:
    try:
        kernel32, _ = _windows_libraries()
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        advapi32.OpenProcessToken.argtypes = [
            wt.HANDLE,
            wt.DWORD,
            ctypes.POINTER(wt.HANDLE),
        ]
        advapi32.OpenProcessToken.restype = wt.BOOL

        class Luid(ctypes.Structure):
            _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_long)]

        class LuidAndAttributes(ctypes.Structure):
            _fields_ = [("luid", Luid), ("attributes", ctypes.c_ulong)]

        class TokenPrivileges(ctypes.Structure):
            _fields_ = [
                ("count", ctypes.c_ulong),
                ("privileges", LuidAndAttributes * 1),
            ]

        advapi32.LookupPrivilegeValueW.argtypes = [
            wt.LPCWSTR,
            wt.LPCWSTR,
            ctypes.POINTER(Luid),
        ]
        advapi32.LookupPrivilegeValueW.restype = wt.BOOL
        advapi32.AdjustTokenPrivileges.argtypes = [
            wt.HANDLE,
            wt.BOOL,
            ctypes.POINTER(TokenPrivileges),
            wt.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        advapi32.AdjustTokenPrivileges.restype = wt.BOOL

        token = wt.HANDLE()
        if not advapi32.OpenProcessToken(
            kernel32.GetCurrentProcess(), 0x0020 | 0x0008, ctypes.byref(token)
        ):
            return
        try:
            luid = Luid()
            if not advapi32.LookupPrivilegeValueW(
                None, "SeDebugPrivilege", ctypes.byref(luid)
            ):
                return
            privileges = TokenPrivileges()
            privileges.count = 1
            privileges.privileges[0].luid = luid
            privileges.privileges[0].attributes = 0x00000002
            advapi32.AdjustTokenPrivileges(
                token, False, ctypes.byref(privileges), 0, None, None
            )
        finally:
            kernel32.CloseHandle(token)
    except OSError:
        return


def inject_dll(
    process_id: int,
    dll_path: str | os.PathLike[str],
    *,
    process_handle: int | None = None,
    timeout_ms: int = 30_000,
) -> None:
    kernel32, _ = _windows_libraries()
    _enable_debug_privilege()
    resolved_dll = Path(dll_path).resolve()
    if not resolved_dll.is_file():
        raise FileNotFoundError(resolved_dll)
    dll_bytes = os.fsencode(resolved_dll) + b"\0"

    kernel32.OpenProcess.restype = wt.HANDLE
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    kernel32.CloseHandle.argtypes = [wt.HANDLE]
    kernel32.VirtualAllocEx.argtypes = [
        wt.HANDLE,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    kernel32.VirtualFreeEx.argtypes = [
        wt.HANDLE,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
    ]
    kernel32.WriteProcessMemory.argtypes = [
        wt.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetModuleHandleW.restype = wt.HMODULE
    kernel32.GetProcAddress.argtypes = [wt.HMODULE, ctypes.c_char_p]
    kernel32.GetProcAddress.restype = ctypes.c_void_p
    kernel32.CreateRemoteThread.argtypes = [
        wt.HANDLE,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.CreateRemoteThread.restype = wt.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wt.HANDLE, ctypes.c_ulong]
    kernel32.GetExitCodeThread.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_ulong)]
    owns_process_handle = process_handle is None
    process = process_handle
    if process is None:
        for _ in range(5):
            process = kernel32.OpenProcess(PROCESS_INJECTION_ACCESS, False, process_id)
            if process:
                break
            time.sleep(1)
        if not process:
            raise OSError(f"OpenProcess failed: {ctypes.get_last_error()}")

    remote_memory = None
    thread = None
    thread_completed = False
    try:
        kernel32.VirtualAllocEx.restype = ctypes.c_void_p
        remote_memory = kernel32.VirtualAllocEx(
            process, None, len(dll_bytes), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
        )
        if not remote_memory:
            raise OSError(f"VirtualAllocEx failed: {ctypes.get_last_error()}")

        written = ctypes.c_size_t()
        if not kernel32.WriteProcessMemory(
            process,
            remote_memory,
            ctypes.c_char_p(dll_bytes),
            len(dll_bytes),
            ctypes.byref(written),
        ):
            raise OSError(f"WriteProcessMemory failed: {ctypes.get_last_error()}")

        load_library = kernel32.GetProcAddress(
            kernel32.GetModuleHandleW("kernel32.dll"), b"LoadLibraryA"
        )
        if not load_library:
            raise OSError("GetProcAddress(LoadLibraryA) failed")

        thread_id = ctypes.c_ulong()
        thread = kernel32.CreateRemoteThread(
            process,
            None,
            0,
            load_library,
            remote_memory,
            0,
            ctypes.byref(thread_id),
        )
        if not thread:
            raise OSError(f"CreateRemoteThread failed: {ctypes.get_last_error()}")
        wait_result = kernel32.WaitForSingleObject(thread, timeout_ms)
        if wait_result == WAIT_TIMEOUT:
            raise TimeoutError(
                f"LoadLibraryA timed out after {timeout_ms} ms in process {process_id}"
            )
        if wait_result != WAIT_OBJECT_0:
            raise OSError(
                "WaitForSingleObject failed: "
                f"result={wait_result} error={ctypes.get_last_error()}"
            )
        thread_completed = True
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeThread(thread, ctypes.byref(exit_code))
        if not exit_code.value:
            raise RuntimeError("LoadLibraryA returned NULL")
    finally:
        if thread:
            kernel32.CloseHandle(thread)
        if remote_memory and thread_completed:
            kernel32.VirtualFreeEx(process, remote_memory, 0, MEM_RELEASE)
        if owns_process_handle:
            kernel32.CloseHandle(process)
