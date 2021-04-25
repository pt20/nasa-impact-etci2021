#!python

from PIL import Image
import pandas as pd
import argparse
from tqdm import tqdm
from multiprocessing import Pool, RLock
from pathlib import Path
from typing import Tuple


def proc(idx: Tuple[str, str, str], df: pd.DataFrame):
    (place, datetime, t) = idx
    max_x, max_y, _ = df.max()

    img = Image.new(mode="L", size=(max_y * 256, max_x * 256))

    for _, (x, y, path) in tqdm(df.iterrows(), total=df.shape[0]):
        tmp = Image.open(Path(".").joinpath(path))
        img.paste(tmp, (256 * y, 256 * x))

    img.save(Path(path).parents[-3].joinpath(f"{place}_{datetime}_{t}.png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""Combine dataset imates """)
    parser.add_argument(
        "dataset", type=str, choices=["train", "val", "test"], help="Dataset type"
    )
    parser.add_argument(
        "--place",
        type=str,
        choices=["northal", "nebraska", "bangladesh"],
        help="Specific place to gent",
    )
    parser.add_argument("--datetime", type=pd.to_datetime, help="Specific datetime")
    args = parser.parse_args()

    # Read generated CSV with csv_gen.sh
    csv = pd.read_csv(
        f"data/{args.dataset}.csv",
        parse_dates=["Datetime"],
        index_col=["Place", "Datetime", "Type"],
        dtype={"x": "Int64", "y": "Int64"},
    )

    # NOT WORKING, I DON'T KNOW WHY
    # tqdm.set_lock(RLock())
    # with Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
    # p.map(proc, csv.groupby(level=["Place", "Datetime", "Type"]))

    for idx, df in csv.groupby(level=["Place", "Datetime", "Type"]):
        proc(idx, df)
