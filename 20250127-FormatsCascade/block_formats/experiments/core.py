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
import transformers
from torch import Tensor, nn

from .. import fit as F
from .. import quantisation as Q

# Sweeping

MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    "microsoft/phi-4",
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

# Maps parameter name (or a default "*") to quantisation format or fit spec
ModelFormats = dict[str, Q.TensorFormat | F.Scaled]


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
            k: v.to("cpu", copy=True) for k, v in model.state_dict().items()
        }
        return cls(model=model, original_params=original_params)

    @property
    def device(self) -> torch.device:
        (device,) = set(p.device for p in self.model.parameters())
        return device

    def reset(self) -> None:
        for name, p in self.model.state_dict().items():
            p[...] = self.original_params[name].to(p.device)

    def quantise(self, formats: ModelFormats) -> list[dict[str, Any]]:
        """Quantise parameters of the model, returning a log of the outcomes.

        formats -- dict[ParamName, Quantiser]

            Tries `formats[param_name] or formats["*"]`, if neither is found, the parameter
            is unquantised.

            Quantiser can be a Q.TensorFormat, which is used directly or an F.Scaled which
            is first fitted to each tensor being quantised

        returns -- list[ParamRecord] -- records bits, quantisation error etc.
        """
        log = []
        for name, p in self.model.state_dict().items():
            if p.ndim == 2:
                p0 = self.original_params[name].to(p.device)
                fmt_or_fit = formats.get(name, formats.get("*"))
                if fmt_or_fit:
                    if isinstance(fmt_or_fit, Q.TensorFormat):
                        fmt = fmt_or_fit
                    elif isinstance(fmt_or_fit, F.Scaled):
                        fmt = fmt_or_fit.fit(p0)
                    p[...] = fmt.quantise(p0)
                    log.append(
                        dict(
                            name=name,
                            quantised=True,
                            nelement=p.nelement(),
                            bits=fmt.count_bits_tensor(p),
                            rmse=(p - p0).float().pow(2).mean().sqrt().item(),
                            norm=p0.float().pow(2).mean().sqrt().item(),
                            fmt=dataclasses.asdict(fmt),
                            fmt_str=str(fmt),
                        )
                    )
                    continue
            log.append(
                dict(
                    name=name,
                    quantised=False,
                    nelement=p.nelement(),
                    bits=Q.TorchFormat(p.dtype).count_bits(p.shape),
                )
            )
        return log


# Database

DB_REGION, DB_TABLE = ("eu-central-1", "2025-04-block-number-formats")


class AttrDict(dict):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.__dict__ = self


def _generate_id(experiment: str) -> str:
    return (
        experiment
        + "/"
        + "".join(random.choices(string.ascii_letters + string.digits, k=10))
    )


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


def _to_db(value: Any) -> Any:
    if isinstance(
        value, (str, int, decimal.Decimal, bool, type(None), bytes, bytearray)
    ):
        return value
    if isinstance(value, float):
        # Approximately match float32 precision
        return decimal.Decimal.from_float(value).normalize(decimal.Context(prec=8))
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return value.type
    if isinstance(value, (list, tuple)):
        return [_to_db(v) for v in value]
    if isinstance(value, (np.ndarray, Tensor)):
        return _to_db(value.tolist())
    if isinstance(value, set):
        return {_to_db(v) for v in value}
    if isinstance(value, dict):
        non_string_keys = [k for k in value if not isinstance(k, str)]
        if non_string_keys:
            raise TypeError(
                f"Cannot convert non-string keys for the database: {set(type(k) for k in non_string_keys)}"
            )
        return {k: _to_db(v) for k, v in value.items()}
    raise TypeError(f"Unexpected type for database: {type(value)}")


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
        experiment = config.pop("experiment")
        self.run_id = _generate_id(experiment)
        self._record = dict(
            experiment=experiment,
            run_id=self.run_id,
            config=_to_db(config),
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
        self.sync()

    def sync(self) -> None:
        self._db.put_item(Item=_to_db(self._record))

    def summary(self, **args: Any) -> None:
        self._record["summary"].update(args)
        self.sync()

    def __enter__(self) -> "Experiment":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_value:
            self._record.update(
                error=dict(
                    type=exc_type.__qualname__,
                    message=str(exc_value),
                    trace=traceback.format_tb(tb),
                ),
            )
            self._record["meta"].update(status="failed")
        else:
            self._record["meta"].update(status="finished")
        self._record["meta"].update(duration=time.time() - self._t0)
        self.sync()


def _call_paginated(
    db: boto3.dynamodb.table.TableResource, method: str, **args: Any
) -> Iterable[Any]:
    start = {}
    while True:
        response = getattr(db, method)(**args, **start)
        yield from response["Items"]
        if "LastEvaluatedKey" not in response:
            break
        start = dict(ExclusiveStartKey=response["LastEvaluatedKey"])


def run(run_id: str) -> dict[str, Any]:
    """Fetch a specific run by ID."""
    response = _db().get_item(Key=dict(experiment=run_id.split("/")[0], run_id=run_id))
    if "Item" not in response:
        raise KeyError(f"Run {run_id} not found")
    return response["Item"]


def runs(experiment: str) -> list[dict[str, Any]]:
    """Fetch all runs for a given experiment."""
    items = _call_paginated(
        _db(), "query", KeyConditionExpression=dbc.Key("experiment").eq(experiment)
    )
    return sorted((_from_db(x) for x in items), key=lambda x: x["meta"]["time"])


def experiments() -> list[str]:
    """A list of all experiments in the database."""
    counts = collections.Counter(
        x["experiment"]
        for x in _call_paginated(_db(), "scan", ProjectionExpression="experiment")
    )
    return [dict(experiment=k, runs=counts[k]) for k in sorted(counts)]


def delete_run(run_id: str) -> None:
    """Remove the given run."""
    experiment = run_id.split("/")[0]
    _db().delete_item(Key=dict(experiment=experiment, run_id=run_id))
