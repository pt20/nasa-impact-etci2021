from pathlib import Path

import pandas as pd

DATA_DIR = "../data/train/"


def harmonize_and_export(data_dir=DATA_DIR):
    """
    necessary for sanity at later stage - takes time
    """
    columns = ["vv", "vh", "flood_label", "water_body_label"]

    images_vv = sorted(
        [str(i) for i in list(Path(data_dir).rglob(f"**/{columns[0]}/*.png"))]
    )
    images_vh = sorted(
        [str(i) for i in list(Path(data_dir).rglob(f"**/{columns[1]}/*.png"))]
    )
    images_fld = sorted(
        [str(i) for i in list(Path(data_dir).rglob(f"**/{columns[2]}/*.png"))]
    )
    images_wtr = sorted(
        [str(i) for i in list(Path(data_dir).rglob(f"**/{columns[3]}/*.png"))]
    )

    all_files = []

    for f in images_vv:
        png_name = f.split("/")[-1]
        fname = png_name[:-7]

        if (
            any(fname in s for s in images_vh)
            and any(fname in s for s in images_fld)
            and any(fname in s for s in images_wtr)
        ):
            vh_idx = [i for i, x in enumerate(images_vh) if fname in x][0]
            fld_idx = [i for i, x in enumerate(images_fld) if fname in x][0]
            wtr_idx = [i for i, x in enumerate(images_wtr) if fname in x][0]

            all_files.append(
                (f, images_vh[vh_idx], images_fld[fld_idx], images_wtr[wtr_idx])
            )

        else:
            print(f"{f}: not found in all folders")

    df = pd.DataFrame(all_files, columns=columns)

    # write csv
    df.to_csv("../all_files_harmonised.csv")
