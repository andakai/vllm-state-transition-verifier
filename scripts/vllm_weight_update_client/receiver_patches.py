"""Runtime patches that make vLLM's in-place weight updates fast and correct.

``weight_update_receiver.py`` applies them in every vLLM worker before the
model is built; vLLM's source stays unmodified. Every patch first checks that
its target exists (with the expected parameters where the wrapper relies on
them) and is skipped otherwise. The patches serve every path that ends in
vLLM's layerwise reload: the packed NCCL broadcast, the CUDA IPC engine, and
``reload_weights`` from checkpoint files.

General:

* ``layerwise_count_cache``: layerwise reload dry-runs every buffered weight's
  loader on meta tensors to count the elements it loads; the count only
  depends on the loader and its arguments, so it is cached.
* ``layerwise_fast_loader``: the same buffering loader without its per-call
  fixed costs. It also learns where each transfer tensor that a loader call
  received unchanged is buffered (a *route*), so later updates buffer it
  there straight away, without the model's per-tensor ``load_weights`` path.
* ``expert_map_host_cache``: answers the global -> local expert id lookup from
  a host copy instead of one device sync per routed-expert loader call.
* ``skip_nonlocal_experts``: under expert parallelism every rank receives
  every expert; tensors of experts the rank does not hold are learned and
  never unpacked again. The unpack also replays learned routes, and learns
  the tensors ``load_weights`` copies into a tensor layerwise never reloads
  (replayed as that copy) or drops (left packed).
* ``routed_expert_copy_replay``: copies a materialized expert layer's
  buffered weights per (parameter, geometry) group instead of one Python
  loader call per expert tensor.
* ``packed_consumer_prefetch``: vLLM's packed NCCL consumer posts the receive
  of bucket k+1 only after loading bucket k, and loads alternate buckets on
  alternating streams, so a layer spanning buckets can read a buffer already
  handed to the next receive (corrupted routed experts with a fast sender).
  The patched consumer receives ``RECV_AHEAD`` buckets ahead on one stream
  and loads every bucket on one stream, from a pool of reusable buffers.
* ``auto_loader_tied_cache``: caches the tied-embedding scan every nested
  ``AutoWeightsLoader`` runs over the whole model.
* ``layerwise_compact_deferred``: a deferred attention layer keeps a compact
  copy of a buffered view, so the receive bucket behind it can be freed.
* ``trtllm_bf16_block_layout_batched``: relays out FlashInfer TRT-LLM BF16 MoE
  weights in groups of experts instead of one expert at a time.
* ``async_layer_copies``: small pageable host-to-device copies made while a
  layer is processed go through pinned memory, and copies back into pinned
  host kernel tensors do not block, so the host does not wait for the
  buckets still being received.
* ``attention_reload_keep_storage``: a re-created attention parameter or
  buffer is copied into the storage CUDA graphs captured, so the graphs read
  the new values (NaN after level-2 sleep otherwise).
"""

from __future__ import annotations

import bisect
import contextlib
import functools
import inspect
import logging
import math
import sys
import types
import weakref
from collections.abc import Callable

import torch
from torch.overrides import TorchFunctionMode

logger = logging.getLogger("vllm.receiver_patches")  # vLLM's handlers

RECV_AHEAD = 4  # packed NCCL buckets received ahead of the one loading
_SCALARS = (bool, int, float, str, type(None))
_EXPERT_LOADER = (
    "self param loaded_weight weight_name shard_id expert_id return_success"
)
# Learned per transfer name (a checkpoint tensor name), valid for the life of
# the worker:
SKIP_NAMES: set[str] = set()  # never handed to the model again
ROUTES: dict[str, tuple | None] = {}  # where a loader buffers it (None: no route)
DIRECT: dict[str, tuple | None] = {}  # the one plain copy load_weights makes
_UNROUTED: set[str] = set()  # seen once without a route
# The transfer name and size of each tensor handed out in the current bucket
# (or reload), by data pointer, and the names a loader received in it.
_TRANSFER_NAME: dict[int, tuple[str, int]] = {}
_LOADED_NOW: set[str] = set()
_IPC_UNPACK = [False]  # vLLM's IPC consumer is unpacking the sender's buffer
_RECV_POOLS: dict[str, list] = {}  # packed NCCL receive buffers per device

PATCHES: dict[str, Callable[[], str]] = {}
APPLIED: dict[str, str] = {}


def patch(name: str):
    def register(fn: Callable[[], str]) -> Callable[[], str]:
        PATCHES[name] = fn
        return fn

    return register


def _run(name: str, fn: Callable[[], str]) -> None:
    try:
        APPLIED[name] = fn()
    except Exception as exc:  # report, never break the worker
        APPLIED[name] = f"failed: {exc!r}"


def _replace(owner, attr: str, new, params: str | None = None) -> str:
    """Install ``new(original)`` as ``owner.attr`` once, if the target exists
    (with these space-separated parameter names)."""
    if isinstance(owner, type):
        original = owner.__dict__.get(attr)
    else:
        original = getattr(owner, attr, None)
    if original is None:
        return f"skipped: {attr} not found"
    tag = new.__qualname__  # several patches may wrap one target in turn
    if getattr(original, "_receiver_patched", None) == tag:
        return "applied"
    if (
        params is not None
        and list(inspect.signature(original).parameters) != params.split()
    ):
        return f"skipped: unexpected signature {inspect.signature(original)}"
    wrapped = new(original)
    wrapped._receiver_patched = tag
    setattr(owner, attr, wrapped)
    return "applied"


def _all_applied(*results: str) -> str:
    return results[0] if len(set(results)) == 1 else f"partial: {sorted(results)}"


# ---------------------------------------------------------------- learning


def _transfer_name(t) -> str | None:
    """The transfer name of the handed-out tensor ``t`` is (a view of all of)."""
    if not isinstance(t, torch.Tensor) or t.is_meta:
        return None
    entry = _TRANSFER_NAME.get(t.data_ptr())
    if entry is None or entry[1] != t.numel() * t.element_size():
        return None
    return entry[0]


def _binder(sig: inspect.Signature):
    """``sig.bind(*args, **kwargs)`` + ``apply_defaults()``, without
    ``inspect``'s per-call parameter walk for plain signatures."""
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    plain = all(p.kind in kinds for p in sig.parameters.values())
    names = list(sig.parameters)
    known = set(names)
    defaults = {
        n: p.default for n, p in sig.parameters.items() if p.default is not p.empty
    }

    def bind(args, kwargs):
        if plain and len(args) <= len(names) and kwargs.keys() <= known:
            arguments = dict(zip(names, args))
            if not kwargs.keys().isdisjoint(arguments):
                return sig.bind(*args, **kwargs)  # raises like inspect
            arguments.update(kwargs)
            for n, d in defaults.items():
                arguments.setdefault(n, d)
            if len(arguments) == len(names):
                return inspect.BoundArguments(sig, {n: arguments[n] for n in names})
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound

    return bind


def _replayable(value) -> bool:
    """A scalar, or a wrapper's ``*args`` / ``**kwargs`` of scalars."""
    if isinstance(value, dict):
        items = value.values()
    elif isinstance(value, tuple):
        items = value
    else:
        items = (value,)
    return all(isinstance(x, _SCALARS) for x in items)


def _learn_route(layer, param_name, bound, numel) -> None:
    """Remember where this loader call buffered a transfer tensor, if it got
    the tensor (or a view of all of it) and only replayable arguments besides
    it and the parameter. A name loaded twice in one bucket is never routed."""
    loaded = bound.arguments.get("loaded_weight")
    name = _transfer_name(loaded)
    if name is None or ROUTES.get(name, ()) is None:
        return
    if name in _LOADED_NOW:
        ROUTES[name] = None
        return
    _LOADED_NOW.add(name)
    param, param_arg, scalars = getattr(layer, param_name, None), None, {}
    for arg, value in bound.arguments.items():
        if value is param and param is not None:
            param_arg = arg
        elif arg != "loaded_weight":
            if not _replayable(value):
                ROUTES[name] = None
                return
            scalars[arg] = value
    # the model may pass another dtype or shape of the same bytes
    view = (loaded.dtype, tuple(loaded.shape)) if loaded.is_contiguous() else None
    if param_arg is None or view is None:
        ROUTES[name] = None
    else:
        ROUTES[name] = (
            layer,
            param_name,
            param_arg,
            bound.signature,
            scalars,
            numel,
            view,
        )


def _route(route, tensor) -> bool:
    """Buffer ``tensor`` as its learned loader call did, and process the
    layer once complete. False when the layer is not being reloaded."""
    import vllm.model_executor.model_loader.reload.layerwise as lw

    layer, param_name, param_arg, sig, scalars, numel, (dtype, shape) = route
    info = lw.LAYERWISE_INFO.get(layer)
    if info is None or info.kernel_tensors is None or not info.can_load():
        return False
    if tensor.dtype != dtype or tuple(tensor.shape) != shape:
        tensor = tensor.view(dtype).view(shape)
    args = {**scalars, param_arg: getattr(layer, param_name), "loaded_weight": tensor}
    bound = inspect.BoundArguments(sig, {n: args[n] for n in sig.parameters})
    info.loaded_weights.append((param_name, bound))
    info.load_numel += numel
    if lw.is_deferred_attention_layer(layer):
        return True  # processed at finalize
    lw.LOADING_LAYERS.add(layer)
    if info.load_numel >= info.load_numel_total:
        lw._layerwise_process(layer, info)
        lw.LOADING_LAYERS.discard(layer)
    return True


def _direct_copy(direct, tensor) -> bool:
    """Make the learned copy of ``tensor``; False if its target changed."""
    layer, attr, nbytes, dst, src = direct
    base = getattr(layer, attr, None)
    if (
        not isinstance(base, torch.Tensor)
        or base.is_meta
        or base.untyped_storage().nbytes() != nbytes
    ):
        return False
    target = torch.as_strided(base.detach(), *dst)
    target.copy_(tensor.as_strided(src[0], src[1], tensor.storage_offset() + src[2]))
    return True


def _replay(name: str, tensor: torch.Tensor, copy: bool = False) -> bool:
    """Hand ``tensor`` over by its learned route (``copy``: buffer a clone)
    or copy, if any."""
    route = ROUTES.get(name)
    if route is not None and _route(route, tensor.clone() if copy else tensor):
        return True
    direct = DIRECT.get(name)
    return direct is not None and _direct_copy(direct, tensor)


class _UnroutedRecorder(TorchFunctionMode):
    """What ``load_weights`` does with some transfer tensors: copies out of
    them, views of them, anything else (touched)."""

    def __init__(self, tensors):
        super().__init__()
        self.spans = sorted(
            (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size(), n)
            for n, t in tensors
        )
        self.starts = [s[0] for s in self.spans]
        self.copies, self.views, self.touched = {}, {}, set()

    def owner(self, x):
        if isinstance(x, torch.Tensor) and not x.is_meta:
            i = bisect.bisect_right(self.starts, x.data_ptr()) - 1
            if i >= 0 and x.data_ptr() < self.spans[i][1]:
                return self.spans[i][2]
        return None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        owners = {
            o
            for a in (*args, *kwargs.values())
            for x in (a if isinstance(a, (list, tuple)) else (a,))
            if (o := self.owner(x)) is not None
        }
        if not owners:
            return out
        name = getattr(func, "__name__", "")
        if (
            name == "copy_"
            and len(args) == 2
            and not kwargs
            and self.owner(args[0]) is None
            and (src := self.owner(args[1])) is not None
        ):
            self.copies.setdefault(src, []).append((args[0], args[1]))
            return out
        results = out if isinstance(out, (list, tuple)) else (out,)
        in_place = name.endswith("_") and not name.startswith("__")
        if in_place or any(
            isinstance(r, torch.Tensor) and self.owner(r) is None for r in results
        ):
            self.touched |= owners
        else:  # metadata, or views of it
            for r in results:
                if isinstance(r, torch.Tensor):
                    self.views.setdefault(self.owner(r), []).append(weakref.ref(r))
        return out


@contextlib.contextmanager
def _learn_unrouted(weights):
    """Around ``load_weights`` of one bucket: on a tensor's second load
    without a route, learn whether the model made one plain copy of it into a
    tensor layerwise never reloads (``DIRECT``) or dropped it (skipped)."""
    import vllm.model_executor.model_loader.reload.layerwise as lw
    from vllm.model_executor.model_loader.reload import meta

    fresh = [
        (n, t) for n, t in weights if n in _UNROUTED and n not in DIRECT and t.numel()
    ]
    if not fresh:
        yield
    else:
        refs = {n: sys.getrefcount(t) for n, t in fresh}
        with _UnroutedRecorder(fresh) as rec:
            yield
        kept = {}  # storage -> a layer tensor layerwise never reloads
        for layer in list(lw.LAYERWISE_INFO.keys()):
            for attr in getattr(meta, "SKIP_TENSORS", ()):
                t = layer._parameters.get(attr, layer._buffers.get(attr))
                if isinstance(t, torch.Tensor) and not t.is_meta:
                    kept[t.untyped_storage().data_ptr()] = (layer, attr, t)
        for n, t in fresh:
            copies = rec.copies.get(n, [])
            alive = any(v() is not None for v in rec.views.get(n, []))
            if n in ROUTES or n in rec.touched or alive or len(copies) > 1:
                DIRECT[n] = None
            elif not copies:
                if n in rec.views or sys.getrefcount(t) != refs[n]:
                    DIRECT[n] = None
                else:
                    SKIP_NAMES.add(n)  # dropped, e.g. a speculative decoding layer
            else:
                dst, src = copies[0]
                target = kept.get(dst.untyped_storage().data_ptr())
                if (
                    target is None
                    or dst.dtype != target[2].dtype
                    or src.dtype != t.dtype
                ):
                    DIRECT[n] = None
                else:
                    DIRECT[n] = (
                        target[0],
                        target[1],
                        dst.untyped_storage().nbytes(),
                        (tuple(dst.shape), dst.stride(), dst.storage_offset()),
                        (
                            tuple(src.shape),
                            src.stride(),
                            src.storage_offset() - t.storage_offset(),
                        ),
                    )
    _UNROUTED.update(n for n, _ in weights if n not in ROUTES)


# ---------------------------------------------------------------- general


def _callable_key(fn) -> tuple | None:
    """Equal for the same loader across updates (layerwise re-binds loaders)."""
    if isinstance(fn, types.MethodType):
        return ("method", id(fn.__func__), id(fn.__self__))
    if isinstance(fn, types.FunctionType):
        return ("function", id(fn))
    return None


def _args_key(bound) -> tuple | None:
    parts = []
    for name, value in bound.arguments.items():
        if isinstance(value, torch.Tensor):
            # the destination is the layer's meta tensor, kept for the
            # model's lifetime in the layerwise restore metadata
            key = id(value) if name == "param" else None
            parts.append((name, key, tuple(value.shape), value.dtype))
        elif isinstance(value, _SCALARS):
            parts.append((name, value))
        elif isinstance(value, (tuple, list)) and all(
            isinstance(x, _SCALARS) for x in value
        ):
            parts.append((name, tuple(value)))
        else:
            return None
    return tuple(parts)


@patch("layerwise_count_cache")
def _layerwise_count_cache() -> str:
    import vllm.model_executor.model_loader.reload.layerwise as lw

    cache: dict[tuple, tuple[int, object]] = {}

    def new(original):
        def get_numel_loaded(weight_loader, args):
            loader_key = _callable_key(weight_loader)
            args_key = _args_key(args) if loader_key is not None else None
            if args_key is None:
                return original(weight_loader, args)
            key = (loader_key, args_key)
            if key not in cache:
                numel, ret = original(weight_loader, args)
                if not isinstance(ret, _SCALARS):  # only plain results replay
                    return numel, ret
                cache[key] = (numel, ret)
            return cache[key]

        return get_numel_loaded

    return _replace(lw, "get_numel_loaded", new)


@patch("layerwise_fast_loader")
def _layerwise_fast_loader() -> str:
    import vllm.model_executor.model_loader.reload.layerwise as lw

    def new(original):
        @functools.wraps(original)
        def make_online_process_loader(layer, param_name):
            info = lw.get_layerwise_info(layer)
            original_loader = lw._get_original_loader(getattr(layer, param_name))
            bind = _binder(inspect.signature(original_loader))
            tensors_seen = [None]

            @functools.wraps(original_loader, assigned=("__doc__", "__annotations__"))
            def online_process_loader(*args, **kwargs):
                if not info.can_load():
                    return None  # excessive loading, as vLLM's loader
                # refresh only when the layer's tensor set changed
                seen = (len(layer._parameters), len(layer._buffers))
                if seen != tensors_seen[0]:
                    info.load_numel_total = lw.get_layer_size(layer)
                    lw._wrap_parameters_weight_loader(layer)
                    tensors_seen[0] = seen
                bound_args = bind(args, kwargs)
                info.loaded_weights.append((param_name, bound_args))
                num_loaded, ret = lw.get_numel_loaded(original_loader, bound_args)
                info.load_numel += num_loaded
                _learn_route(layer, param_name, bound_args, num_loaded)
                if lw.is_deferred_attention_layer(layer):
                    return ret
                if layer not in lw.LOADING_LAYERS and lw.has_device_tensors(bound_args):
                    lw.LOADING_LAYERS.add(layer)
                if info.load_numel >= info.load_numel_total:
                    lw._layerwise_process(layer, info)
                    lw.LOADING_LAYERS.discard(layer)
                return ret

            return online_process_loader

        return make_online_process_loader

    return _replace(lw, "make_online_process_loader", new, "layer param_name")


@patch("expert_map_host_cache")
def _expert_map_host_cache() -> str:
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        ExpertMapManager,
    )

    def new(original):
        @functools.wraps(original)
        def map_global_to_local(self, global_id):
            expert_map = self._expert_map
            if (
                expert_map is None
                or expert_map.is_meta
                or not isinstance(global_id, int)
            ):
                return original(self, global_id)
            key = (id(expert_map), expert_map._version)
            host = self.__dict__.get("_receiver_host_map")
            if host is None or host[0] != key:
                host = self.__dict__["_receiver_host_map"] = (key, expert_map.tolist())
            return host[1][global_id]

        return map_global_to_local

    return _replace(ExpertMapManager, "map_global_to_local", new)


@patch("skip_nonlocal_experts")
def _skip_nonlocal_experts() -> str:
    import vllm.distributed.weight_transfer.packed_tensor as packed
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    def new_loader(original):
        @functools.wraps(original)
        def weight_loader(
            self,
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success=False,
        ):
            quant = self.quant_config and self.quant_config.get_name()
            global_sf = (
                getattr(self.quant_method, "use_global_sf", False)
                and "input_scale" in weight_name
            )
            if (
                quant != "gpt_oss_mxfp4"
                and not global_sf
                and self._map_global_expert_id_to_local_expert_id(expert_id) == -1
            ):
                # placement is fixed without EPLB: never hand this name out again
                if not self.moe_config.moe_parallel_config.enable_eplb:
                    name = _transfer_name(loaded_weight)
                    if name is not None:
                        SKIP_NAMES.add(name)
                return False if return_success else None
            return original(
                self,
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                return_success,
            )

        return weight_loader

    kept_cache: dict[tuple, list] = {}

    def new_unpack(original):
        @functools.wraps(original)
        def unpack_tensor(packed_tensor, names, shapes, dtypes, tensor_sizes):
            # vLLM's split + view for the names not skipped; which ones and
            # where is remembered per bucket (a sender repeats its buckets)
            key = None
            if names:
                key = (
                    names[0],
                    names[-1],
                    len(names),
                    sum(tensor_sizes),
                    len(SKIP_NAMES),
                )
            kept = kept_cache.get(key)
            if kept is None:
                kept, offset = [], 0
                for i, (name, size) in enumerate(zip(names, tensor_sizes)):
                    if name not in SKIP_NAMES:
                        shape, isz = tuple(shapes[i]), dtypes[i].itemsize
                        stride = tuple(
                            math.prod(shape[d + 1 :]) for d in range(len(shape))
                        )
                        elem = offset // isz if offset % isz == 0 else None
                        kept.append((i, offset, shape, stride, elem))
                    offset += size
                kept_cache[key] = kept
            _TRANSFER_NAME.clear()
            _LOADED_NOW.clear()
            flat, out, typed = packed_tensor.view(-1), [], {}
            for i, offset, shape, stride, elem in kept:
                dtype = dtypes[i]
                if elem is None:
                    t = flat.narrow(0, offset, tensor_sizes[i]).view(dtype).view(*shape)
                else:  # one call instead of narrow + two views
                    base = typed.get(dtype)
                    if base is None:
                        usable = flat.numel() - flat.numel() % dtype.itemsize
                        base = typed[dtype] = flat[:usable].view(dtype)
                    t = base.as_strided(shape, stride, elem)
                # the IPC sender reuses its buffer: a routed tensor is a copy
                if _replay(names[i], t, copy=_IPC_UNPACK[0]):
                    continue
                if tensor_sizes[i]:
                    _TRANSFER_NAME[t.data_ptr()] = (names[i], tensor_sizes[i])
                out.append((names[i], t))
            return out

        return unpack_tensor

    def new_ipc_consumer(original):
        @functools.wraps(original)
        def packed_ipc_consumer(
            ipc_handle,
            names,
            shapes,
            dtype_names,
            tensor_sizes,
            device_index,
            importer=None,
        ):
            # vLLM's IPC consumer unpacks through unpack_tensor above and
            # copies what is left: the model's loaders get the copies
            _IPC_UNPACK[0] = True
            try:
                out = original(
                    ipc_handle,
                    names,
                    shapes,
                    dtype_names,
                    tensor_sizes,
                    device_index,
                    importer,
                )
            finally:
                _IPC_UNPACK[0] = False
            _TRANSFER_NAME.clear()
            for name, t in out:
                if t.numel():
                    _TRANSFER_NAME[t.data_ptr()] = (name, t.numel() * t.element_size())
            return out

        return packed_ipc_consumer

    unpack_params = "packed_tensor names shapes dtypes tensor_sizes"
    results = [
        _replace(RoutedExperts, "weight_loader", new_loader, _EXPERT_LOADER),
        _replace(packed, "unpack_tensor", new_unpack, unpack_params),
    ]
    try:
        import vllm.distributed.weight_transfer.ipc_engine as ipc
    except ImportError:
        ipc = None
    if ipc is not None and getattr(ipc, "packed_ipc_consumer", None) is getattr(
        packed, "packed_ipc_consumer", 0
    ):
        ipc_params = (
            "ipc_handle names shapes dtype_names tensor_sizes device_index importer"
        )
        results.append(
            _replace(packed, "packed_ipc_consumer", new_ipc_consumer, ipc_params)
        )
        ipc.packed_ipc_consumer = packed.packed_ipc_consumer
    return _all_applied(*results)


@patch("routed_expert_copy_replay")
def _routed_expert_copy_replay() -> str:
    """A routed-expert loader call runs ~45 us of Python to end in one copy
    whose geometry only depends on its arguments. The first call per
    (layer, weight, shard, expert, shapes) records its writes; once a reload
    materializes an expert layer, the buffered weights whose call made
    exactly one copy from the loaded tensor are copied per group."""
    import vllm.model_executor.model_loader.reload.layerwise as lw
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    plans: dict[tuple, tuple | None] = {}
    layouts: dict[int, tuple] = {}  # id(layer) -> (buffered weights, layout)

    def plan_key(
        layer, param, loaded_weight, weight_name, shard_id, expert_id, return_success
    ):
        return (
            id(layer),
            weight_name,
            shard_id,
            expert_id,
            return_success,
            tuple(param.shape),
            param.dtype,
            param.storage_offset(),
            tuple(loaded_weight.shape),
            loaded_weight.stride(),
            loaded_weight.dtype,
        )

    class Record(TorchFunctionMode):
        """The in-place writes into one parameter's storage."""

        def __init__(self, storage):
            super().__init__()
            self.storage, self.writes = storage, []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            name = getattr(func, "__name__", "")
            if (
                name.endswith("_")
                and not name.startswith("__")
                and args
                and isinstance(args[0], torch.Tensor)
                and args[0].untyped_storage().data_ptr() == self.storage
            ):
                self.writes.append((name, args))
            return out

    def record(original, self, param, loaded_weight, *args):
        with Record(param.untyped_storage().data_ptr()) as rec:
            ret = original(self, param, loaded_weight, *args)
        # replayable: exactly one copy into the parameter from the loaded
        # tensor's storage (calls that write nothing keep calling the loader)
        if (
            isinstance(ret, _SCALARS)
            and len(rec.writes) == 1
            and rec.writes[0][0] == "copy_"
            and len(rec.writes[0][1]) >= 2
            and isinstance(rec.writes[0][1][1], torch.Tensor)
        ):
            dst, src = rec.writes[0][1][:2]
            if (
                src.untyped_storage().data_ptr()
                == loaded_weight.untyped_storage().data_ptr()
            ):
                offset = src.storage_offset() - loaded_weight.storage_offset()
                return ret, (
                    tuple(dst.shape),
                    dst.stride(),
                    dst.storage_offset(),
                    tuple(src.shape),
                    src.stride(),
                    offset,
                    ret,
                )
        return ret, None

    def new(original):
        @functools.wraps(original)
        def weight_loader(
            self,
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success=False,
        ):
            args = (weight_name, shard_id, expert_id, return_success)
            if param.is_meta or loaded_weight.is_meta:  # layerwise counting dry run
                return original(self, param, loaded_weight, *args)
            key = plan_key(self, param, loaded_weight, *args)
            if key not in plans:
                ret, plans[key] = record(original, self, param, loaded_weight, *args)
                return ret
            return original(self, param, loaded_weight, *args)

        return weight_loader

    def layout(layer, items):
        """[(parameter, size, stride, step, first offset, item indices, source
        geometry or None)] for the replayable buffered weights, and the
        indices left to the loader."""
        groups, rest = {}, []
        for k, (name, bound) in enumerate(items):
            a = bound.arguments
            src = a["loaded_weight"]
            plan = plans.get(
                plan_key(
                    layer,
                    getattr(layer, name),
                    src,
                    a["weight_name"],
                    a["shard_id"],
                    a["expert_id"],
                    a["return_success"],
                )
            )
            if plan is None or src.is_meta:
                rest.append(k)
                continue
            dsize, dstride, doff, ssize, sstride, soff, _ = plan
            whole = (ssize, sstride, soff) == (tuple(src.shape), src.stride(), 0)
            geometry = None if whole else (ssize, sstride, soff)
            groups.setdefault((name, dsize, dstride, geometry), []).append((doff, k))
        out = []
        for (name, dsize, dstride, geometry), members in groups.items():
            members.sort()
            offs = [m[0] for m in members]
            step = offs[1] - offs[0] if len(offs) > 1 else 0
            extent = 1 + sum((n - 1) * st for n, st in zip(dsize, dstride))
            if (
                len(offs) < 2
                or step < extent
                or any(b - a != step for a, b in zip(offs, offs[1:]))
            ):
                rest += [m[1] for m in members]  # not provably disjoint and regular
                continue
            out.append(
                (name, dsize, dstride, step, offs[0], [m[1] for m in members], geometry)
            )
        return out, sorted(rest)

    def batch(layer, info):
        """Destinations at equally spaced offsets form one strided view,
        filled with one copy; the rest stays for vLLM's per-weight calls."""
        items = info.loaded_weights
        ident = [
            (
                name,
                b.arguments.get("weight_name"),
                b.arguments.get("shard_id"),
                b.arguments.get("expert_id"),
                getattr(b.arguments.get("loaded_weight"), "shape", None),
            )
            for name, b in items
        ]
        cached = layouts.get(id(layer))
        if cached is None or cached[0] != ident:
            cached = layouts[id(layer)] = (ident, layout(layer, items))
        groups, rest = cached[1]
        for name, dsize, dstride, step, off0, members, geometry in groups:
            srcs = [items[k][1].arguments["loaded_weight"] for k in members]
            if geometry is not None:
                ssize, sstride, soff = geometry
                srcs = [
                    t.as_strided(ssize, sstride, t.storage_offset() + soff)
                    for t in srcs
                ]
            dst = getattr(layer, name).data.as_strided(
                (len(members), *dsize), (step, *dstride), off0
            )
            dst.copy_(torch.stack(srcs))  # one gather kernel, one strided copy
        info.loaded_weights[:] = [items[k] for k in rest]

    def new_materialize(original):
        @functools.wraps(original)
        def materialize_layer(layer, info):
            original(layer, info)
            if info.kernel_tensors is not None and isinstance(layer, RoutedExperts):
                batch(layer, info)

        return materialize_layer

    return _all_applied(
        _replace(RoutedExperts, "weight_loader", new, _EXPERT_LOADER),
        _replace(lw, "materialize_layer", new_materialize, "layer info"),
    )


def _storage_uses(t: torch.Tensor) -> int:
    """References to ``t``'s storage, plus the one this call makes; a large
    number where the count is not available."""
    use_count = getattr(torch._C, "_storage_Use_Count", None)
    if use_count is None:
        return 1 << 30
    return use_count(t.untyped_storage()._cdata)


@patch("packed_consumer_prefetch")
def _packed_consumer_prefetch() -> str:
    import vllm.distributed.weight_transfer.nccl_engine as ne
    import vllm.distributed.weight_transfer.packed_tensor as packed

    stock = getattr(packed, "packed_nccl_broadcast_consumer", None)
    if (
        stock is None
        or getattr(ne, "packed_nccl_broadcast_consumer", None) is not stock
    ):
        return "skipped: nccl_engine does not use packed_tensor's consumer"
    nbytes_of: dict[str, int] = {}

    def new(original):
        @functools.wraps(original)
        def consumer(
            iterator,
            group,
            src,
            post_unpack_func,
            buffer_size_bytes=packed.DEFAULT_PACKED_BUFFER_SIZE_BYTES,
            num_buffers=packed.DEFAULT_PACKED_NUM_BUFFERS,
            device="cuda",
        ):
            def cuts():
                # vLLM's cuts: a bucket closes once its byte count exceeds the
                # buffer size, the crossing tensor included (cut lazily, so the
                # first receive is posted right away)
                cur, total = [], 0
                for name, (shape, dtype) in iterator:
                    size = nbytes_of.get(name)
                    if size is None:
                        size = nbytes_of[name] = math.prod(shape) * dtype.itemsize
                    cur.append((name, shape, dtype, size))
                    total += size
                    if total > buffer_size_bytes:
                        yield cur, total
                        cur, total = [], 0
                if cur:
                    yield cur, total

            comm = torch.cuda.Stream(device=device)
            compute = torch.cuda.Stream(device=device)
            compute.wait_stream(torch.cuda.current_stream())
            # Buffers are reused once no view into them is left (layerwise
            # keeps views of incomplete layers), so the allocator never has
            # to synchronize the device; the pool is kept across updates.
            pool = _RECV_POOLS.setdefault(str(device), [])
            loaded = [None]  # event after the last bucket's loads

            def take(nbytes):
                def free():
                    return [i for i, b in enumerate(pool) if _storage_uses(b) <= 2]

                # a buffer grown for a huge tensor goes back once free
                for i in reversed(free()):
                    if pool[i].numel() > 2 * buffer_size_bytes:
                        del pool[i]
                idle = free()
                fit = next((i for i in idle if pool[i].numel() >= nbytes), None)
                if fit is not None:
                    return pool[fit]
                for i in reversed(idle):  # none fits: give their memory back first
                    del pool[i]
                pool.append(torch.empty(nbytes, dtype=torch.uint8, device=device))
                return pool[-1]

            def post(bucket):
                with torch.cuda.stream(comm):
                    buf = take(bucket[1])[: bucket[1]]
                    if loaded[0] is not None:
                        comm.wait_event(loaded[0])  # its previous contents are consumed
                    group.broadcast(buf, src=src)
                    done = torch.cuda.Event()
                    done.record(comm)
                return bucket[0], buf, done

            learn = APPLIED.get("skip_nonlocal_experts") == "applied"
            buckets, pending = cuts(), []
            nxt = next(buckets, None)
            while pending or nxt is not None:
                # receive ahead while loading; a bucket grown around a huge
                # tensor is received alone, after the previous one is freed
                while nxt is not None and (
                    not pending
                    or len(pending) < RECV_AHEAD
                    and max(pending[0][1].numel(), nxt[1]) <= 2 * buffer_size_bytes
                ):
                    pending.append(post(nxt))
                    nxt = next(buckets, None)
                meta, buf, done = pending.pop(0)
                names, shapes, dtypes, sizes = (list(x) for x in zip(*meta))
                with torch.cuda.stream(compute):
                    compute.wait_event(done)
                    weights = packed.unpack_tensor(buf, names, shapes, dtypes, sizes)
                    with (
                        _learn_unrouted(weights) if learn else contextlib.nullcontext()
                    ):
                        post_unpack_func(weights)
                    del weights
                    loaded[0] = torch.cuda.Event()
                    loaded[0].record(compute)
                del buf
            comm.synchronize()
            compute.synchronize()

        return consumer

    want = "iterator group src post_unpack_func buffer_size_bytes num_buffers device"
    result = _replace(packed, "packed_nccl_broadcast_consumer", new, want)
    ne.packed_nccl_broadcast_consumer = packed.packed_nccl_broadcast_consumer
    return result


@patch("auto_loader_tied_cache")
def _auto_loader_tied_cache() -> str:
    import vllm.model_executor.models.utils as mu

    untied: weakref.WeakSet = weakref.WeakSet()

    def new(original):
        @functools.wraps(original)
        def _get_tied_embedding_params(module):
            if module in untied:
                return {}
            aliased = original(module)
            if not aliased:
                untied.add(module)
            return aliased

        return _get_tied_embedding_params

    return _replace(mu, "_get_tied_embedding_params", new)


@patch("layerwise_compact_deferred")
def _layerwise_compact_deferred() -> str:
    import vllm.model_executor.model_loader.reload.layerwise as lw

    def new(original):
        @functools.wraps(original)
        def is_deferred_attention_layer(layer):
            deferred = original(layer)
            info = lw.LAYERWISE_INFO.get(layer) if deferred else None
            if info is not None and info.loaded_weights:
                # called right after the online loader buffered this weight
                bound = info.loaded_weights[-1][1]
                for name, value in list(bound.arguments.items()):
                    if (
                        isinstance(value, torch.Tensor)
                        and not isinstance(value, torch.nn.Parameter)
                        and not value.is_meta
                        and value.untyped_storage().nbytes()
                        > value.numel() * value.element_size()
                    ):
                        bound.arguments[name] = value.clone()
            return deferred

        return is_deferred_attention_layer

    return _replace(lw, "is_deferred_attention_layer", new)


def batched_trtllm_block_layout(
    cache_permute_indices,
    w13_weight,
    w2_weight,
    is_gated_act_gemm: bool = True,
    group_bytes: int = 1 << 30,
):
    """Same contract and result as vLLM's
    ``convert_moe_weights_to_flashinfer_trtllm_block_layout``, in place, one
    ``index_select`` per group of experts (all share one row permutation)."""
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise ValueError(
            "Unquantized Moe Backend FlashInfer TRTLLM requires bfloat16 weights"
        )
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    epilogue_tile_m, block_k, num_experts = 128, 128, w13_weight.shape[0]

    def convert(weight: torch.Tensor, is_w13: bool) -> torch.Tensor:
        if not weight.is_contiguous():
            raise ValueError("expected contiguous expert weights")
        experts = weight.view(torch.uint8)
        rows, cols = experts[0].shape
        if is_w13:
            idx = _maybe_get_cached_w3_w1_permute_indices(
                cache_permute_indices,
                experts[0],
                epilogue_tile_m,
                is_gated_act_gemm=is_gated_act_gemm,
            )
            if is_gated_act_gemm:
                idx = (idx + rows // 2) % rows
        else:
            idx = get_w2_permute_indices_with_cache(
                cache_permute_indices, experts[0], epilogue_tile_m
            )
        idx = idx.to(experts.device)
        group = max(1, group_bytes // max(rows * cols, 1))
        for start in range(0, num_experts, group):
            part = experts[start : start + group]
            blocks = part.view(part.shape[0], rows, cols // block_k, block_k)
            blocks = blocks.permute(0, 2, 1, 3)
            part.view(-1).copy_(torch.index_select(blocks, 2, idx).view(-1))
        return experts.view(num_experts, cols // block_k, rows, block_k).view(
            torch.bfloat16
        )

    return convert(w13_weight, True), convert(w2_weight, False)


@patch("trtllm_bf16_block_layout_batched")
def _trtllm_bf16_block_layout_batched() -> str:
    import vllm.model_executor.layers.fused_moe.oracle.unquantized as oracle
    import vllm.model_executor.layers.quantization.utils.flashinfer_utils as fiu

    name = "convert_moe_weights_to_flashinfer_trtllm_block_layout"
    original = getattr(fiu, name)
    if getattr(oracle, name, None) is not original:
        return "skipped: oracle does not use the flashinfer_utils function"
    want = "cache_permute_indices w13_weight w2_weight is_gated_act_gemm"
    result = _replace(
        fiu,
        name,
        lambda _: functools.wraps(original)(batched_trtllm_block_layout),
        want,
    )
    setattr(oracle, name, getattr(fiu, name))
    return result


@patch("async_layer_copies")
def _async_layer_copies() -> str:
    import vllm.model_executor.model_loader.reload.layerwise as lw

    small = 64 << 20

    def is_cuda(target) -> bool:
        if isinstance(target, torch.Tensor):
            return target.is_cuda
        if isinstance(target, (torch.device, str, int)) and not isinstance(
            target, bool
        ):
            return torch.device(target).type == "cuda"
        return False

    def pageable(t) -> bool:
        return (
            isinstance(t, torch.Tensor)
            and t.device.type == "cpu"
            and not t.is_pinned()
            and t.numel() * t.element_size() <= small
        )

    class AsyncCopies(TorchFunctionMode):
        def __init__(self, pinned_kernel):
            super().__init__()
            self.pinned_kernel = pinned_kernel  # storages of pinned host kernel tensors

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.tensor and args and not isinstance(args[0], torch.Tensor):
                device = kwargs.get("device", torch.get_default_device())
                nested = isinstance(args[0], (list, tuple)) and any(
                    isinstance(x, torch.Tensor) for x in args[0]
                )
                if is_cuda(device) and not nested:
                    host = func(
                        *args, **{**kwargs, "device": "cpu", "pin_memory": True}
                    )
                    return host.to(device, non_blocking=True)
            elif (
                "non_blocking" not in kwargs
                and args
                and pageable(args[0])
                and (
                    func is torch.Tensor.to
                    and len(args) <= 3
                    and any(is_cuda(a) for a in (*args[1:], kwargs.get("device")))
                    or func is torch.Tensor.cuda
                    and len(args) <= 2
                )
            ):
                return func(
                    args[0].pin_memory(), *args[1:], **kwargs, non_blocking=True
                )
            elif (
                func is torch.Tensor.copy_
                and len(args) == 2
                and "non_blocking" not in kwargs
            ):
                dst, src = args
                if dst.is_cuda and pageable(src):
                    return func(dst, src.pin_memory(), non_blocking=True)
                if (
                    isinstance(src, torch.Tensor)
                    and src.is_cuda
                    and dst.untyped_storage().data_ptr() in self.pinned_kernel
                ):
                    return func(dst, src, non_blocking=True)
            return func(*args, **kwargs)

    def new(original):
        @functools.wraps(original)
        def _layerwise_process(layer, info):
            kernel = [
                t for group in (info.kernel_tensors or ()) for t in group.values()
            ]
            pinned = {
                t.untyped_storage().data_ptr()
                for t in kernel
                if t.device.type == "cpu" and t.is_pinned()
            }
            with AsyncCopies(pinned):
                return original(layer, info)

        return _layerwise_process

    return _replace(lw, "_layerwise_process", new, "layer info")


@patch("attention_reload_keep_storage")
def _attention_reload_keep_storage() -> str:
    import vllm.model_executor.model_loader.reload.layerwise as lw

    def new(original):
        @functools.wraps(original)
        def _finalize_attention_layer(layer, info, model_config):
            kernel = {
                n: t for group in (info.kernel_tensors or ()) for n, t in group.items()
            }
            original(layer, info, model_config)
            for name, old in kernel.items():
                table = (
                    layer._parameters if name in layer._parameters else layer._buffers
                )
                cur = table.get(name)
                if (
                    isinstance(cur, torch.Tensor)
                    and cur is not old
                    and not cur.is_meta
                    and cur.shape == old.shape
                    and cur.dtype == old.dtype
                    and cur.device == old.device
                ):
                    old.data.copy_(cur.data)
                    table[name] = old

        return _finalize_attention_layer

    params = "layer info model_config"
    return _replace(lw, "_finalize_attention_layer", new, params)


def apply() -> dict[str, str]:
    for name, fn in PATCHES.items():
        if name not in APPLIED:
            _run(name, fn)
    logger.info("receiver_patches: %s", APPLIED)
    return dict(APPLIED)
