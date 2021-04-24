#!python

from PIL import Image
import pandas as pd
import argparse
from tqdm import tqdm, trange
from multiprocessing import Pool, RLock


def proc(itr):
    (place, datetime, t), (max_x, max_y) = itr
    max_x += 1
    max_y += 1

    img = Image.new(mode="L", size=(max_y * 256, max_x * 256))
    date = datetime.strftime("%Y%m%dt%H%M%S")

    for num in trange(max_x * max_y):
        x = num // max_y
        y = num % max_y

        tmp = Image.open(
            f"data/train/{place}_{date}/tiles/{t}/{place}_{date}_x-{x}_y-{y}_{t}.png"
        )
        img.paste(tmp, (256 * y, 256 * x))
    img.save(f"data/train/{place}_{date}_{t}.png")


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

    tcsv = pd.read_csv(f"{args.dataset}.csv", parse_dates=["Datetime"])

    tqdm.set_lock(RLock())  # for managing output contention
    p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    p.map(proc, tcsv.groupby(by=["Place", "Datetime", "type"]).max().iterrows())
