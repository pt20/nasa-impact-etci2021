import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import datetime

import mlflow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from architecture import build_unet, build_vgg19_unet
from metrics import *
from pipeline import collect_image_paths, tf_dataset

# TODO: bring them to yaml (or json file) for auto logging
input_files_csv = "all_files_harmonized.csv"
batch_size = 8
input_shape = (256, 256, 3)
epochs = 10
lr = 1e-4
model_path = "results/"
timestamp_str = str(datetime.datetime.now()).replace(" ", "_")
ckpt_path = os.path.join(model_path, timestamp_str, "model.h5")
csv_path = f"{timestamp_str}-data.csv"

my_mlflow_uri = os.environ.get("MY_MLFLOW_URI")
mlflow.set_tracking_uri(my_mlflow_uri)

images_vv = collect_image_paths(input_files_csv, img_type="vv")
masks_fld = collect_image_paths(input_files_csv, img_type="flood_label")

# train test split
train_x, test_x = train_test_split(images_vv, test_size=0.2, random_state=112)
train_y, test_y = train_test_split(masks_fld, test_size=0.2, random_state=112)

train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

callbacks = [
    ModelCheckpoint(ckpt_path, monitor="val_loss", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, verbose=1),
    CSVLogger(csv_path),
    EarlyStopping(monitor="val_loss", patience=10),
    TensorBoard(),
]

metrics = [dice_coef, iou, tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]

model = build_vgg19_unet(input_shape)
model.compile(loss=dice_loss, optimizer=tf.keras.optimizers.Adam(lr), metrics=metrics)

train_steps = len(train_x) // batch_size
if len(train_x) % batch_size != 0:
    train_steps += 1

test_steps = len(test_x) // batch_size
if len(test_x) % batch_size != 0:
    test_steps += 1

mlflow.tensorflow.autolog()
mlflow.set_experiment("sample-experiment")

with mlflow.start_run():
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("input_shape", input_shape)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("train_steps", train_steps)
    mlflow.log_param("test_steps", test_steps)
    mlflow.log_param("model", model.name)

    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=test_steps,
        callbacks=callbacks,
    )

    mlflow.log_artifact(csv_path)
    mlflow.log_artifact(ckpt_path)
