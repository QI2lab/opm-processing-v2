from pathlib import Path

import opm_processing.cuda as cuda


def test_preload_cuda_libraries_is_a_noop_off_linux(monkeypatch):
    def unexpected_cdll(*args, **kwargs):
        raise AssertionError("ctypes.CDLL should not be called off Linux")

    monkeypatch.setattr(cuda.sys, "platform", "win32")
    monkeypatch.setattr(cuda.ctypes, "CDLL", unexpected_cdll)
    monkeypatch.setattr(cuda, "_CUDA_PRELOAD_ATTEMPTED", False)

    cuda.preload_cuda_libraries()

    assert cuda._CUDA_PRELOAD_ATTEMPTED is False


def test_preload_cuda_libraries_loads_existing_libraries_once(tmp_path, monkeypatch):
    root = tmp_path / "nvidia"
    first = _touch_library(root, "runtime", "libfirst.so")
    second = _touch_library(root, "math", "libsecond.so")
    failed = _touch_library(root, "math", "libfailed.so")
    handles = []
    calls = []

    def fake_cdll(path, mode):
        library_path = Path(path)
        calls.append((library_path, mode))
        if library_path == failed:
            raise OSError("unresolved dependency")
        handle = object()
        handles.append(handle)
        return handle

    monkeypatch.setattr(cuda.sys, "platform", "linux")
    monkeypatch.setattr(cuda, "_nvidia_library_roots", lambda: (root,))
    monkeypatch.setattr(
        cuda,
        "_CUDA_LIBRARIES",
        (("runtime", (first.name,)), ("math", (second.name, failed.name))),
    )
    monkeypatch.setattr(cuda, "_CUDA_LIBRARY_HANDLES", [])
    monkeypatch.setattr(cuda, "_CUDA_PRELOAD_ATTEMPTED", False)
    monkeypatch.setattr(cuda.ctypes, "CDLL", fake_cdll)

    cuda.preload_cuda_libraries()
    cuda.preload_cuda_libraries()

    assert calls == [
        (first, cuda.ctypes.RTLD_GLOBAL),
        (second, cuda.ctypes.RTLD_GLOBAL),
        (failed, cuda.ctypes.RTLD_GLOBAL),
    ]
    assert cuda._CUDA_LIBRARY_HANDLES == handles


def _touch_library(root: Path, package: str, name: str) -> Path:
    library = root / package / "lib" / name
    library.parent.mkdir(parents=True, exist_ok=True)
    library.touch()
    return library
