import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def isolated_deep_ep_import(monkeypatch):
    def is_deep_ep_module(name):
        return name == "deep_ep" or name.startswith("deep_ep.")

    previous_modules = {
        name: module for name, module in sys.modules.items() if is_deep_ep_module(name)
    }
    for name in previous_modules:
        sys.modules.pop(name)

    yield monkeypatch

    for name in list(sys.modules):
        if is_deep_ep_module(name):
            sys.modules.pop(name)
    sys.modules.update(previous_modules)


class _StubTorch(ModuleType):
    def __getattr__(self, name):
        placeholder = type(name, (), {})
        setattr(self, name, placeholder)
        return placeholder


class _EventOverlap:
    def __init__(self, event, extra_tensors=None):
        self.event = event
        self.extra_tensors = extra_tensors


def _load_deep_ep_wrapper(monkeypatch, current_device):
    torch = _StubTorch("torch")
    torch.cuda = ModuleType("torch.cuda")
    torch.cuda.current_device = current_device
    torch.cuda.current_stream = lambda device=None: type(
        "Stream", (), {"cuda_stream": 123}
    )()

    torch_dist = ModuleType("torch.distributed")
    torch_dist.ProcessGroup = type("ProcessGroup", (), {})
    # Single-rank stand-ins for the cross-rank config gather in Buffer.__init__.
    torch_dist.get_world_size = lambda group=None: 1
    torch_dist.all_gather_object = lambda out, obj, group=None: out.__setitem__(0, obj)
    torch.distributed = torch_dist

    uccl = ModuleType("uccl")
    uccl_ep = ModuleType("uccl.ep")
    uccl_ep.Config = type("Config", (), {})
    uccl_ep.EventHandle = type("EventHandle", (), {})
    uccl.ep = uccl_ep

    wrapper_utils = ModuleType("deep_ep.utils")
    wrapper_utils.EventHandle = type("EventHandle", (), {})
    wrapper_utils.EventOverlap = _EventOverlap
    wrapper_utils.check_nvlink_connections = lambda *args, **kwargs: None
    wrapper_utils.initialize_uccl = lambda *args, **kwargs: None
    wrapper_utils.destroy_uccl = lambda *args, **kwargs: None
    wrapper_utils.detect_group_topology = lambda *args, **kwargs: None
    wrapper_utils._fp8_e4m3_dtype = object()

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", torch_dist)
    monkeypatch.setitem(sys.modules, "uccl", uccl)
    monkeypatch.setitem(sys.modules, "uccl.ep", uccl_ep)
    wrapper_path = Path(__file__).parents[1]
    package_path = wrapper_path / "deep_ep"
    package_spec = importlib.util.spec_from_file_location(
        "deep_ep",
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    wrapper_package = importlib.util.module_from_spec(package_spec)
    monkeypatch.setitem(sys.modules, "deep_ep", wrapper_package)
    monkeypatch.setitem(sys.modules, "deep_ep.utils", wrapper_utils)

    # deep_ep/buffer.py is a symlink to this canonical implementation. Load
    # the target directly so this CPU-only test also runs on Windows checkouts
    # where Git materializes symlinks as plain text files.
    buffer_spec = importlib.util.spec_from_file_location(
        "deep_ep.buffer", wrapper_path.parent / "bench" / "buffer.py"
    )
    buffer_module = importlib.util.module_from_spec(buffer_spec)
    monkeypatch.setitem(sys.modules, "deep_ep.buffer", buffer_module)
    buffer_spec.loader.exec_module(buffer_module)

    package_spec.loader.exec_module(wrapper_package)
    return wrapper_package, torch.cuda


class _FakeTensor:
    def __init__(self, shape, *, dtype="dtype", strides=None):
        self.shape = tuple(shape)
        self.device = "cuda:0"
        self.dtype = dtype
        self._strides = strides or tuple(reversed(range(1, len(shape) + 1)))

    def data_ptr(self):
        return 1

    def dim(self):
        return len(self.shape)

    def element_size(self):
        return 2

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def stride(self, dim):
        if dim >= len(self._strides):
            raise AssertionError("one-dimensional scale tensor queried for stride(1)")
        return self._strides[dim]

    def __getitem__(self, _):
        return self


class _FakeRuntime:
    def __init__(self, num_rdma_ranks):
        self.num_rdma_ranks = num_rdma_ranks
        self.intranode_dispatch_args = None
        self.internode_dispatch_args = None

    def get_num_rdma_ranks(self):
        return self.num_rdma_ranks

    def intranode_dispatch(self, *args):
        self.intranode_dispatch_args = args
        return None

    def intranode_prepare(self, *_):
        return 2, [2], None

    def internode_dispatch(self, *args):
        self.internode_dispatch_args = args
        return None

    def internode_prepare(self, *_):
        return 2, 2, [2], None

    def get_source_meta_bytes(self):
        return 2

    def get_num_max_nvl_peers(self):
        return 2


class _FakeGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _FakeConfig:
    num_sms = 2


def _new_buffer(deep_ep, runtime):
    buffer = object.__new__(deep_ep.Buffer)
    buffer.runtime = runtime
    buffer.group_size = 2
    buffer.group = _FakeGroup(2)
    buffer.proxies = []
    buffer.get_comm_stream = lambda: None
    return buffer


def _install_fake_empty(monkeypatch):
    torch = sys.modules["torch"]
    monkeypatch.setattr(
        torch,
        "empty",
        lambda shape, **kwargs: _FakeTensor(shape, dtype=kwargs.get("dtype")),
        raising=False,
    )


def _cached_intranode_handle():
    return (
        _FakeTensor((2, 2)),
        _FakeTensor((2, 2)),
        _FakeTensor((2, 2)),
        2,
        _FakeTensor((2,)),
        _FakeTensor((3, 2)),
        _FakeTensor((3, 2)),
    )


def _cached_internode_handle():
    return (
        _FakeTensor((3, 2)),
        _FakeTensor((2, 2)),
        _FakeTensor((2, 2)),
        _FakeTensor((2, 2)),
        _FakeTensor((2,)),
        _FakeTensor((2, 2)),
        _FakeTensor((2,)),
        2,
        2,
        _FakeTensor((2, 2)),
        _FakeTensor((3, 2)),
        _FakeTensor((2, 2)),
    )


_SCALE_CASES = [
    (None, None, (0, 0, 0)),
    ((3,), (5,), (1, 5, 0)),
    ((3, 2), (2, 1), (2, 2, 1)),
    ((3, 2), (1, 3), (2, 1, 3)),
]


@pytest.mark.parametrize(
    ("scale_shape", "scale_strides", "expected_metadata"),
    _SCALE_CASES,
)
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("num_rdma_ranks", [1, 2])
def test_dispatch_marshals_scale_metadata(
    isolated_deep_ep_import,
    cached,
    num_rdma_ranks,
    scale_shape,
    scale_strides,
    expected_metadata,
):
    monkeypatch = isolated_deep_ep_import
    deep_ep, _ = _load_deep_ep_wrapper(monkeypatch, lambda: 0)
    _install_fake_empty(monkeypatch)
    runtime = _FakeRuntime(num_rdma_ranks)
    buffer = _new_buffer(deep_ep, runtime)
    x = _FakeTensor((3, 128))
    scales = (
        None if scale_shape is None else _FakeTensor(scale_shape, strides=scale_strides)
    )

    dispatch_kwargs = {"config": _FakeConfig()}
    if num_rdma_ranks == 1:
        if cached:
            dispatch_kwargs["handle"] = _cached_intranode_handle()
        else:
            dispatch_kwargs.update(
                num_tokens_per_rank=_FakeTensor((2,)),
                is_token_in_rank=_FakeTensor((3, 2)),
                num_tokens_per_expert=_FakeTensor((2,)),
            )
    else:
        if cached:
            dispatch_kwargs["handle"] = _cached_internode_handle()
        else:
            dispatch_kwargs.update(
                num_tokens_per_rank=_FakeTensor((2,)),
                num_tokens_per_rdma_rank=_FakeTensor((2,)),
                is_token_in_rank=_FakeTensor((3, 2)),
                num_tokens_per_expert=_FakeTensor((2,)),
            )

    deep_ep.Buffer.dispatch(
        buffer, x if scales is None else (x, scales), **dispatch_kwargs
    )

    runtime_args = (
        runtime.intranode_dispatch_args
        if num_rdma_ranks == 1
        else runtime.internode_dispatch_args
    )
    assert runtime_args[5:8] == expected_metadata


def test_enable_shrink_compatibility(isolated_deep_ep_import):
    monkeypatch = isolated_deep_ep_import

    def fail_if_cuda_is_queried():
        raise AssertionError("enable_shrink=True reached CUDA initialization")

    deep_ep, cuda = _load_deep_ep_wrapper(monkeypatch, fail_if_cuda_is_queried)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    with pytest.raises(
        NotImplementedError,
        match="UCCL EP does not currently support enable_shrink=True",
    ):
        deep_ep.Buffer(group=object(), enable_shrink=True)

    class ReachedNormalInitialization(Exception):
        pass

    def mark_normal_initialization():
        raise ReachedNormalInitialization

    cuda.current_device = mark_normal_initialization
    # The stubs cannot build a config signature; ep/tests/test_config_check.py
    # covers the check itself.
    monkeypatch.setenv("UCCL_EP_CONFIG_CHECK", "0")
    with pytest.raises(ReachedNormalInitialization):
        deep_ep.Buffer(group=object(), enable_shrink=False)


def test_event_handle_is_the_stream_aware_wrapper(isolated_deep_ep_import):
    deep_ep, _ = _load_deep_ep_wrapper(isolated_deep_ep_import, lambda: 0)

    assert deep_ep.EventHandle is sys.modules["deep_ep.utils"].EventHandle


def test_config_check_ships_with_the_wrapper(isolated_deep_ep_import):
    _load_deep_ep_wrapper(isolated_deep_ep_import, lambda: 0)

    # Not a top-level config_check that happens to be importable from ep/bench.
    check = sys.modules["deep_ep.buffer"].check_consistency
    assert check.__module__ == "deep_ep.config_check"


class _LLTensor(_FakeTensor):
    _next_ptr = 0x1000

    def __init__(self, shape, *, dtype="bf16", device="cuda:0"):
        super().__init__(shape, dtype=dtype)
        self.device = device
        _LLTensor._next_ptr += 0x1000
        self._ptr = _LLTensor._next_ptr

    def data_ptr(self):
        return self._ptr


class _FakeLLRuntime:
    def __init__(self):
        self.combine_args = None

    def low_latency_combine(self, *args):
        self.combine_args = args
        return None, None


_LL_NUM_EXPERTS, _LL_MAX_TOKENS, _LL_HIDDEN = 4, 3, 8
# group size 2 -> 2 local experts, 2 * 3 receive slots per expert.
_LL_X_SHAPE = (2, 6, _LL_HIDDEN)


def _ll_setup(monkeypatch, handle_device, current_device=0):
    deep_ep, _ = _load_deep_ep_wrapper(monkeypatch, lambda: current_device)
    torch = sys.modules["torch"]
    torch.bfloat16 = "bf16"

    def empty(shape, *, dtype=None, device="cuda"):
        # Like torch: "cuda" means the current device.
        if device == "cuda":
            device = f"cuda:{current_device}"
        return _LLTensor(shape, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "empty", empty, raising=False)
    runtime = _FakeLLRuntime()
    buffer = _new_buffer(deep_ep, runtime)
    buffer._next_low_latency_combine_buffer = None
    handle = (
        _LLTensor((2, 6), dtype="int32", device=handle_device),
        _LLTensor((2, 2), dtype="int64", device=handle_device),
        _LL_MAX_TOKENS,
        _LL_HIDDEN,
        _LL_NUM_EXPERTS,
    )
    return deep_ep, buffer, runtime, handle


def _ll_combine(deep_ep, buffer, x, handle, zero_copy):
    topk = _LLTensor((3, 2), dtype="int64", device=x.device)
    weights = _LLTensor((3, 2), dtype="float32", device=x.device)
    deep_ep.Buffer.low_latency_combine(
        buffer, x, topk, weights, handle, zero_copy=zero_copy
    )


def test_ll_combine_staging_follows_handle_device(isolated_deep_ep_import):
    # The current device is cuda:0, the dispatch handle lives on cuda:1.
    deep_ep, buffer, runtime, handle = _ll_setup(
        isolated_deep_ep_import, handle_device="cuda:1"
    )
    staged = deep_ep.Buffer.get_next_low_latency_combine_buffer(buffer, handle)
    assert staged.device == "cuda:1"
    assert staged.shape == _LL_X_SHAPE

    x = _LLTensor(_LL_X_SHAPE, device="cuda:1")
    _ll_combine(deep_ep, buffer, x, handle, zero_copy=True)
    assert runtime.combine_args[0] == staged.data_ptr()


@pytest.mark.parametrize(
    "x_kwargs",
    [
        {"shape": (2, 5, _LL_HIDDEN)},
        {"dtype": "fp16"},
        {"device": "cuda:1"},
    ],
    ids=["shape", "dtype", "device"],
)
def test_ll_combine_zero_copy_mismatch_raises(isolated_deep_ep_import, x_kwargs):
    deep_ep, buffer, runtime, handle = _ll_setup(
        isolated_deep_ep_import, handle_device="cuda:0"
    )
    deep_ep.Buffer.get_next_low_latency_combine_buffer(buffer, handle)
    x = _LLTensor(x_kwargs.pop("shape", _LL_X_SHAPE), **x_kwargs)

    with pytest.raises(ValueError, match="zero_copy=True"):
        _ll_combine(deep_ep, buffer, x, handle, zero_copy=True)
    assert runtime.combine_args is None


@pytest.mark.parametrize("stage", [False, True])
def test_ll_combine_uses_x_without_zero_copy_staging(isolated_deep_ep_import, stage):
    deep_ep, buffer, runtime, handle = _ll_setup(
        isolated_deep_ep_import, handle_device="cuda:0"
    )
    if stage:
        # A staged buffer is ignored when zero_copy is off.
        deep_ep.Buffer.get_next_low_latency_combine_buffer(buffer, handle)
    x = _LLTensor(_LL_X_SHAPE)
    _ll_combine(deep_ep, buffer, x, handle, zero_copy=not stage)
    assert runtime.combine_args[0] == x.data_ptr()
