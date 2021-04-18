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
    img = tf.image.decode_png(img)
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    img = tf.identity(img, name=path.decode().split("/")[-1])

    return img


def load_flood_mask(path):
    # https://www.spacefish.biz/2020/11/rgb-segmentation-masks-to-classes-in-tensorflow/
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask)

    colors = [(0, 0, 0), (255, 255, 255)]  # black - non-flood,  # white - flood

    one_hot_map = []
    for color in colors:
        class_map = tf.reduce_all(tf.equal(mask, color), axis=-1)
        one_hot_map.append(class_map)

    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.float32)

    mask = tf.argmax(one_hot_map, axis=-1)
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.cast(mask, tf.float32, name=path.decode().split("/")[-1])

    return mask


def preprocess(img_path, mask_path):
    def f(img_path, mask_path):
        x = load_image(img_path)
        y = load_flood_mask(mask_path)

        return x, y

    image, mask = tf.numpy_function(f, [img_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([256, 256, 3])
    mask.set_shape([256, 256, 1])

    return image, mask


def tf_dataset(images, masks, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size=batch)
    dataset = dataset.prefetch(2)

    return dataset
