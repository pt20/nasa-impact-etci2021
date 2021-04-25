import os
from pathlib import Path

import pandas as pd
import tensorflow as tf


# NOTE: deprecated function - keeping it for now - just in case
def collect_image_paths_old(base_path, img_type):
    files = []
    for i in list(Path(base_path).rglob(f"**/{img_type}/*.png")):
        files.append(str(i.name))

    return [os.path.join(base_path, i) for i in sorted(files)]


def collect_image_paths(csv, img_type):
    df = pd.read_csv(csv, index_col=0)
    assert img_type in list(df.columns)

    return df[img_type].tolist()


def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.cast(img, tf.float32)
    img /= 255.0
    img = tf.identity(img, name=path.decode().split("/")[-1])
    return img


def load_flood_mask(path):
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.float32)
    mask /= 255.0
    mask = tf.identity(mask, name=path.decode().split("/")[-1])
    return mask


def preprocess(img_path, mask_path):
    def f(img_path, mask_path):
        vh = load_image(img_path[0])
        vv = load_image(img_path[1])
        img = tf.concat([vh, vv], axis=-1)
        mask = load_flood_mask(mask_path)
        return img, mask

    image, mask = tf.numpy_function(f, [img_path, mask_path], [tf.float32, tf.float32])
    return image, mask


def tf_dataset(images, masks, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size=batch)
    dataset = dataset.prefetch(2)
    return dataset
