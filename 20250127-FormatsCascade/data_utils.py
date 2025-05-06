"""Utilities for data manipulation."""

from typing import Any

import numpy as np
import pandas as pd


def drop(d: dict[str, Any], *cols: str) -> dict[str, Any]:
    return {k: d[k] for k in d if k not in cols}


def best(df: pd.DataFrame, col: str, group: list[str] | None) -> pd.DataFrame:
    if group is None:
        return df.loc[df[col].idxmin()]
    return (
        df.groupby(group)
        .apply(lambda g: g.loc[g[col].idxmin()], include_groups=False)
        .reset_index()
    )


def select(df: pd.DataFrame, select: dict[str, Any], drop: bool = True) -> pd.DataFrame:
    d = df
    for c, value in select.items():
        if isinstance(value, list):
            d = d[d[c].isin(value)]
        else:
            d = d[d[c] == value]
            if drop:
                d = d.drop(columns=c)
    return d


def optimal(s0: pd.Series, s1: pd.Series) -> pd.Series:
    assert np.array_equal(s0.index, s1.index)
    a = np.array(s0)
    b = np.array(s1)
    optimal = ~((a[None, :] < a[:, None]) & (b[None, :] <= b[:, None])).any(1) & ~(
        (a[None, :] <= a[:, None]) & (b[None, :] < b[:, None])
    ).any(1)
    return pd.Series(optimal, index=s0.index)
