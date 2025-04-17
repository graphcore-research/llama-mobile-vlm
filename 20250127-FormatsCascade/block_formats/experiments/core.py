# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

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
from types import TracebackType
from typing import Any, Iterable

import boto3
import boto3.dynamodb
import boto3.dynamodb.conditions as dbc
import boto3.dynamodb.table
import numpy as np
import torch
from torch import Tensor

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


def runs(experiment: str) -> list[dict[str, Any]]:
    """Fetch all runs for a given experiment."""
    response = _db().query(KeyConditionExpression=dbc.Key("experiment").eq(experiment))
    return _from_db(response["Items"])


def experiments() -> list[str]:
    """A list of all experiments in the database."""
    unique = set([])
    start = {}
    while True:
        response = _db().scan(ProjectionExpression="experiment", **start)
        unique.update(x["experiment"] for x in response["Items"])
        if "LastEvaluatedKey" not in response:
            return sorted(unique)
        start = dict(ExclusiveStartKey=response["LastEvaluatedKey"])


def delete_run(run_id: str) -> None:
    """Remove the given run."""
    experiment = run_id.split("/")[0]
    _db().delete_item(Key=dict(experiment=experiment, run_id=run_id))
