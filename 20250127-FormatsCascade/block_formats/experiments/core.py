# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import collections
import dataclasses
import datetime
import decimal
import itertools as it
import random
import re
import string
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Iterable

import boto3
import boto3.dynamodb
import boto3.dynamodb.conditions as dbc
import boto3.dynamodb.table
import numpy as np
import torch
import tqdm
import transformers
from torch import Tensor, nn


# Sweeping

MODELS = [
    # Llama
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    # Phi
    "microsoft/phi-4",
    # Qwen
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    # Gemma
    "google/gemma-3-1b-pt",
    "google/gemma-3-4b-pt",
    "google/gemma-3-12b-pt",
]

FIELD_MODELS = dataclasses.field(default_factory=lambda: MODELS.copy())
FIELD_DEVICE = dataclasses.field(
    default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


def iter_dict_product(
    config: dict[str, Any], *axes: str, progress: bool = False
) -> Iterable[dict[str, Any]]:
    """Iterate through the product of certain axes in a config."""
    values_list = list(it.product(*(config[k] for k in axes)))
    for i, values in enumerate(values_list):
        if progress:
            print(
                f"# [{i+1}/{len(values_list)}] {dict(zip(axes, values))}",
                file=sys.stderr,
            )
        yield {**config, **dict(zip(axes, values))}


# RequantisableModel


@dataclass
class RequantisableModel:
    """Wraps transformers.PreTrainedModel, storing original parameters on CPU,
    so that they can be restored when needed.
    """

    model: transformers.PreTrainedModel
    original_params: dict[str, nn.Parameter]

    @classmethod
    def load(
        cls, name: str, device: torch.device, dtype: torch.dtype
    ) -> "RequantisableModel":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, device_map=device, torch_dtype=dtype
        )
        original_params = {
            k: v.detach().to("cpu", copy=True) for k, v in model.named_parameters()
        }
        return cls(model=model, original_params=original_params)

    @property
    def device(self) -> torch.device:
        (device,) = set(p.device for p in self.model.parameters())
        return device

    def reset(self) -> None:
        for name, p in self.model.named_parameters():
            p.data[...] = self.original_params[name].to(p.device)
            if hasattr(p, "_quantised"):
                del p._quantised

    def reset_parameter(self, name: str) -> None:
        p = dict(self.model.named_parameters())[name]
        p.data[...] = self.original_params[name].to(p.device)
        if hasattr(p, "_quantised"):
            del p._quantised


# Database

DB_REGION, DB_TABLE = ("eu-central-1", "2025-04-block-number-formats")


class AttrDict(dict):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.__dict__ = self


def generate_id() -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=10))


def _get_username() -> str | None:
    arn = boto3.client("sts").get_caller_identity()["Arn"]
    if m := re.search("user/(.+)", arn):
        return m.group(1)


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None


def _device_info() -> dict[str, Any]:
    try:
        return dict(
            device_name=torch.cuda.get_device_name(),
            device_count=torch.cuda.device_count(),
        )
    except Exception:
        return {}


def _dump_error(error: Exception) -> dict[str, Any]:
    return dict(
        type=type(error).__qualname__,
        message=str(error),
        trace=traceback.format_tb(error.__traceback__),
        **(dict(cause=_dump_error(error.__cause__)) if error.__cause__ else {}),
    )


def _to_db(value: Any, prefix: tuple[Any] = ()) -> Any:
    if isinstance(
        value, (str, int, decimal.Decimal, bool, type(None), bytes, bytearray)
    ):
        return value
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            raise TypeError(f"Bad value {'.'.join(map(str, prefix))}={value}")
        # Approximately match float32 precision
        return decimal.Decimal.from_float(value).normalize(decimal.Context(prec=8))
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return value.type
    if isinstance(value, (list, tuple)):
        return [_to_db(v, prefix + (i,)) for i, v in enumerate(value)]
    if isinstance(value, (np.ndarray, Tensor)):
        return _to_db(value.tolist(), prefix)
    if isinstance(value, set):
        return {_to_db(v, prefix + ("#",)) for v in value}
    if isinstance(value, dict):
        non_string_keys = [k for k in value if not isinstance(k, str)]
        if non_string_keys:
            raise TypeError(
                f"Cannot convert non-string keys for the database, "
                f"{'.'.join(map(str, prefix))}:{set(type(k) for k in non_string_keys)}"
            )
        return {k: _to_db(v, prefix + (k,)) for k, v in value.items()}
    raise TypeError(
        f"Unexpected type for database, {'.'.join(map(str, prefix))}:{type(value)}"
    )


def _from_db(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, list):
        return [_from_db(v) for v in value]
    if isinstance(value, set):
        return {_from_db(v) for v in value}
    if isinstance(value, dict):
        return AttrDict(**{k: _from_db(v) for k, v in value.items()})
    return value


def _db() -> boto3.dynamodb.table.TableResource:
    return boto3.resource("dynamodb", region_name=DB_REGION).Table(DB_TABLE)


class Experiment:
    """A context manager for a running experiment."""

    def __init__(self, config: dict[str, Any]):
        self._db = _db()
        config = config.copy()
        self.experiment = config.pop("experiment")
        self.run_id = generate_id()
        self._record = dict(
            experiment=self.experiment,
            run_id=self.run_id,
            config=config,
            meta=dict(
                status="running",
                time=datetime.datetime.now().isoformat(),
                user=_get_username(),
                commit=_git_head(),
                **_device_info(),
            ),
            summary={},
            error=None,
        )
        self._t0 = time.time()
        self.sync(unrecoverable=True)

    def sync(self, unrecoverable: bool = False) -> None:
        try:
            self._db.put_item(Item=_to_db(self._record))
        except Exception as error:
            if unrecoverable:
                print(
                    f'ERROR: Failed to sync experiment "{self.experiment}/{self.run_id}"'
                    f" with {error!r}",
                    file=sys.stderr,
                    flush=True,
                )
            raise

    def summary(self, **args: Any) -> None:
        old_summary = self._record["summary"]
        self._record["summary"] = {**self._record["summary"], **args}
        try:
            self.sync()
        except Exception:
            # Restore the original summary so that sync isn't broken, and we can still log the error
            self._record["summary"] = old_summary
            raise

    def __enter__(self) -> "Experiment":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_value:
            self._record.update(error=_dump_error(exc_value))
            self._record["meta"].update(status="failed")
        else:
            self._record["meta"].update(status="finished")
        self._record["meta"].update(duration=time.time() - self._t0)
        self.sync(unrecoverable=True)


def _call_paginated(
    db: boto3.dynamodb.table.TableResource, method: str, _progress: bool, **args: Any
) -> Iterable[Any]:
    start = {}
    with tqdm.tqdm(desc=method, disable=not _progress) as pbar:
        while True:
            response = getattr(db, method)(**args, **start)
            pbar.update(len(response["Items"]))
            yield from response["Items"]
            if "LastEvaluatedKey" not in response:
                break
            start = dict(ExclusiveStartKey=response["LastEvaluatedKey"])


def _run_from_db(run: dict[str, Any]) -> dict[str, Any]:
    run = _from_db(run)
    run["id"] = f"{run.experiment}/{run.run_id}"
    return run


def run(id: str) -> dict[str, Any]:
    """Fetch a specific run by ID."""
    experiment, run_id = id.split("/")
    response = _db().get_item(Key=dict(experiment=experiment, run_id=run_id))
    if "Item" not in response:
        raise KeyError(f"Run {id} not found")
    return _run_from_db(response["Item"])


def runs(experiment: str, progress: bool = False) -> list[dict[str, Any]]:
    """Fetch all runs for a given experiment."""
    items = _call_paginated(
        _db(),
        "query",
        KeyConditionExpression=dbc.Key("experiment").eq(experiment),
        _progress=progress,
    )
    return sorted((_run_from_db(x) for x in items), key=lambda x: x["meta"]["time"])


def update_run(run: dict[str, Any]) -> None:
    """Update the given run to reflect local changes (be careful!)"""
    _db().put_item(Item=_to_db({k: v for k, v in run.items() if k != "id"}))


def delete_run(id: str) -> None:
    """Remove the given run."""
    experiment, run_id = id.split("/")
    _db().delete_item(Key=dict(experiment=experiment, run_id=run_id))


def experiments(progress: bool = False) -> list[str]:
    """A list of all experiments in the database."""
    counts = collections.Counter(
        x["experiment"]
        for x in _call_paginated(
            _db(),
            "scan",
            ProjectionExpression="experiment",
            _progress=progress,
        )
    )
    return [dict(experiment=k, runs=counts[k]) for k in sorted(counts)]
