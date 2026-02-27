# =========================
# Multi-Task: Segmentation + Classification (Cat/Dog)
# Dataset: Oxford-IIIT Pets (TFDS)
# =========================

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# If you used this in your notebook already:
from tensorflow_examples.models.pix2pix import pix2pix

# -------------------------
# Config
# -------------------------
IMG_SIZE = 128
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EPOCHS = 20

# Choose whether to use per-pixel sample-weights for segmentation (recommended)
USE_SAMPLE_WEIGHTS = True

# Where to save the trained model
SAVE_PATH = "pet_multi_task.keras"

# -------------------------
# Load Dataset
# -------------------------
dataset, info = tfds.load("oxford_iiit_pet", with_info=True)

print("TFDS version:", info.version)
print("Splits:", info.splits)
print("Species names:", info.features["species"].names)  # usually ['cat', 'dog']

TRAIN_LENGTH = info.splits["train"].num_examples
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# -------------------------
# Preprocess
# -------------------------
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.int32)
    input_mask -= 1  # original mask labels are 1..3 -> make them 0..2
    return input_image, input_mask

def load_image_multitask(datapoint):
    # image
    image = tf.image.resize(datapoint["image"], (IMG_SIZE, IMG_SIZE))
    # mask (keep nearest to avoid mixing labels)
    mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (IMG_SIZE, IMG_SIZE),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image, mask = normalize(image, mask)

    # classification label: species (0=cat, 1=dog typically)
    species = tf.cast(datapoint["species"], tf.int32)

    return image, {
        "segmentation": mask,          # (H,W,1) with values 0..2
        "classification": species      # scalar int (0/1)
    }

# -------------------------
# Augmentation (flip image + mask, keep species unchanged)
# -------------------------
class AugmentMulti(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_masks  = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        img = self.augment_inputs(inputs)
        msk = self.augment_masks(labels["segmentation"])
        sp  = labels["classification"]
        return img, {"segmentation": msk, "classification": sp}

# -------------------------
# Optional: per-pixel sample weights for segmentation
# (helps class imbalance: background vs pet vs border)
# -------------------------
def add_sample_weights_multitask(image, y):
    # class weights for segmentation classes [pet, background, border] AFTER mask -= 1
    # In the original TF tutorial they used [2,2,1] then normalized.
    class_weights = tf.constant([2.0, 2.0, 1.0], dtype=tf.float32)
    class_weights = class_weights / tf.reduce_sum(class_weights)

    seg = tf.cast(y["segmentation"], tf.int32)
    seg_w = tf.gather(class_weights, indices=seg)  # same shape as mask (H,W,1)

    # classification sample weight (all ones)
    cls_w = tf.ones_like(y["classification"], dtype=tf.float32)

    sample_w = {"segmentation": seg_w, "classification": cls_w}
    return image, y, sample_w

# -------------------------
# Build tf.data pipelines
# -------------------------
train_ds = dataset["train"].map(load_image_multitask, num_parallel_calls=tf.data.AUTOTUNE)
test_ds  = dataset["test"].map(load_image_multitask,  num_parallel_calls=tf.data.AUTOTUNE)

train_batches = (
    train_ds
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(AugmentMulti())
    .prefetch(tf.data.AUTOTUNE)
)

test_batches = test_ds.batch(BATCH_SIZE)

if USE_SAMPLE_WEIGHTS:
    train_batches_for_fit = train_batches.map(add_sample_weights_multitask)
else:
    train_batches_for_fit = train_batches

# -------------------------
# Model: U-Net backbone (MobileNetV2) + Segmentation head + Classification head
# -------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=[IMG_SIZE, IMG_SIZE, 3],
    include_top=False
)

layer_names = [
    "block_1_expand_relu",   # 64x64
    "block_3_expand_relu",   # 32x32
    "block_6_expand_relu",   # 16x16
    "block_13_expand_relu",  # 8x8
    "block_16_project",      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

OUTPUT_CLASSES = 3   # segmentation: 0..2 after mask -= 1
CLS_CLASSES = 2      # classification: cat/dog

def multitask_unet_model(seg_output_channels=OUTPUT_CLASSES, cls_classes=CLS_CLASSES):
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])

    # Encoder
    skips = down_stack(inputs)
    x = skips[-1]              # deepest feature map (4x4)
    skip_connections = reversed(skips[:-1])

    # --- Classification head (from encoder)
    cls = tf.keras.layers.GlobalAveragePooling2D()(x)
    cls = tf.keras.layers.Dropout(0.2)(cls)
    cls = tf.keras.layers.Dense(128, activation="relu")(cls)
    cls_logits = tf.keras.layers.Dense(cls_classes, name="classification")(cls)  # logits

    # --- Segmentation decoder (U-Net style)
    for up, skip in zip(up_stack, skip_connections):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    seg_logits = tf.keras.layers.Conv2DTranspose(
        filters=seg_output_channels,
        kernel_size=3,
        strides=2,
        padding="same",
        name="segmentation"
    )(x)  # logits (128x128x3)

    return tf.keras.Model(inputs=inputs, outputs={"segmentation": seg_logits, "classification": cls_logits})

model = multitask_unet_model()
model.summary()

# -------------------------
# Compile
# -------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss={
        "segmentation": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "classification": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    },
    metrics={
        "segmentation": ["accuracy"],
        "classification": ["accuracy"],
    },
    # segmentation is main task; classification helps
    loss_weights={"segmentation": 1.0, "classification": 0.3},
)

# -------------------------
# Train
# -------------------------
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits["test"].num_examples // BATCH_SIZE // VAL_SUBSPLITS

history = model.fit(
    train_batches_for_fit,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=test_batches,
    validation_steps=VALIDATION_STEPS,
)

# -------------------------
# Save
# -------------------------
model.save(SAVE_PATH)
print("Saved model to:", os.path.abspath(SAVE_PATH))

# -------------------------
# Inference utilities
# -------------------------
SPECIES_NAMES = info.features["species"].names  # ['cat','dog'] typically

def predict_on_image(model, image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_arr = tf.keras.utils.img_to_array(img) / 255.0
    x = tf.expand_dims(img_arr, axis=0)

    pred = model.predict(x, verbose=0)

    # segmentation mask
    seg_logits = pred["segmentation"][0]                  # (H,W,3)
    seg_mask = tf.argmax(seg_logits, axis=-1).numpy()     # (H,W)

    # classification
    cls_logits = pred["classification"][0]                # (2,)
    cls_id = int(tf.argmax(cls_logits).numpy())
    cls_name = SPECIES_NAMES[cls_id] if cls_id < len(SPECIES_NAMES) else str(cls_id)

    return img_arr.astype(np.float32), seg_mask, cls_id, cls_name

def show_prediction(image_arr, seg_mask, cls_name):
    plt.figure(figsize=(12,5))

    plt.subplot(1,3,1)
    plt.imshow(image_arr)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(seg_mask)
    plt.title("Predicted Mask (0..2)")
    plt.axis("off")

    # Overlay (simple)
    plt.subplot(1,3,3)
    plt.imshow(image_arr)
    plt.imshow(seg_mask, alpha=0.5)
    plt.title(f"Overlay | Class: {cls_name}")
    plt.axis("off")

    plt.show()

# -------------------------
# Example test on a custom image
# -------------------------
# Put your own path here:
# img_arr, mask, cls_id, cls_name = predict_on_image(model, "my_pet.jpg")
# print("Predicted:", cls_name, "(id:", cls_id, ")")
# show_prediction(img_arr, mask, cls_name)
