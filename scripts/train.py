import tensorflow as tf

# physical_devices = tf.config.list_physical_devices("GPU")
# try:
# for pd in physical_devices:
# tf.config.experimental.set_memory_growth(pd, True)
# except:
# # Invalid device or cannot modify virtual devices once initialized.
# pass

import os

import datetime

import mlflow

# import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
import pandas as pd

from architecture import build_unet, build_vgg19_unet
from double_unet import build_double_unet
from metrics import *
from pipeline import collect_image_paths, tf_dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TODO: bring them to yaml (or json file) for auto logging
input_files_csv = "data/all_files_harmonized.csv"
batch_size_per_replica = 5
input_shape = (256, 256, 3)
epochs = 10
lr = 1e-4
model_path = "data/results/"
timestamp_str = str(datetime.datetime.now()).replace(" ", "_")
ckpt_path = os.path.join(model_path, timestamp_str, "model.h5")
csv_path = f"{timestamp_str}-data.csv"


inputs_df = pd.read_csv(input_files_csv, index_col=0)
# images_vv = collect_image_paths(input_files_csv, img_type="vv")
# masks_fld = collect_image_paths(input_files_csv, img_type="flood_label")

# train test split
train_x, test_x = train_test_split(
    inputs_df[["vh", "vv"]].to_numpy(), test_size=0.2, random_state=112
)
train_y, test_y = train_test_split(
    inputs_df["flood_label"].to_numpy(), test_size=0.2, random_state=112
)

# Set strategy
# tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

callbacks = [
    ModelCheckpoint(ckpt_path, monitor="val_loss", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, verbose=1),
    CSVLogger(csv_path),
    EarlyStopping(monitor="val_loss", patience=10),
    TensorBoard(),
]


with strategy.scope():
    model = build_double_unet(input_shape)
    metrics = [dice_coef, iou, tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]

model.compile(loss=dice_loss, optimizer=tf.keras.optimizers.Adam(lr), metrics=metrics)
model.summary()

# my_mlflow_uri = os.environ.get("MY_MLFLOW_URI")
# mlflow.set_tracking_uri(my_mlflow_uri)
mlflow.tensorflow.autolog()
mlflow.set_experiment("sample-experiment")

with mlflow.start_run():
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("input_shape", input_shape)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("train_length", len(train_dataset))
    mlflow.log_param("test_length", len(test_dataset))
    mlflow.log_param("model", model.name)

    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    mlflow.log_artifact(csv_path)
    mlflow.log_artifact(ckpt_path)
