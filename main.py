# =========================
# GUI App: Multi-Task Segmentation + Classification (Cat/Dog)
# Train (terminal) OR Test (choose model + image) with buttons
# =========================

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog, messagebox

# --- Optional dependency: tensorflow_examples (pix2pix upsample)
# Fallback is provided if not installed
try:
    from tensorflow_examples.models.pix2pix import pix2pix
    def upsample(filters, size):
        return pix2pix.upsample(filters, size)
except ImportError:
    def upsample(filters, size):
        initializer = tf.random_normal_initializer(0., 0.02)
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(
                filters, size, strides=2, padding='same',
                kernel_initializer=initializer, use_bias=False
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

# -------------------------
# GPU config (if available)
# -------------------------
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU detected:", gpus)
        except Exception as e:
            print("⚠️ GPU detected but couldn't set memory growth:", e)
    else:
        print("⚠️ No GPU found. Using CPU.")

# -------------------------
# Helpers: preprocessing
# -------------------------
def normalize(image, mask):
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.int32)
    mask -= 1  # original mask labels are 1..3 -> make them 0..2
    return image, mask


def load_image_multitask(datapoint, img_size):
    image = tf.image.resize(datapoint["image"], (img_size, img_size))
    mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (img_size, img_size),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image, mask = normalize(image, mask)

    # species: 0=cat, 1=dog
    species = tf.cast(datapoint["species"], tf.int32)

    return image, {
        "segmentation": mask,         # (H,W,1) values 0..2
        "classification": species     # scalar 0/1
    }


class AugmentMulti(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_masks = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        img = self.augment_inputs(inputs)
        msk = self.augment_masks(labels["segmentation"])
        sp = labels["classification"]
        return img, {"segmentation": msk, "classification": sp}


def add_sample_weights_multitask(image, y):
    # weights for segmentation classes AFTER mask -= 1
    # common setting: [pet, background, border]
    class_weights = tf.constant([2.0, 2.0, 1.0], dtype=tf.float32)
    class_weights = class_weights / tf.reduce_sum(class_weights)

    seg = tf.cast(y["segmentation"], tf.int32)
    seg_w = tf.gather(class_weights, indices=seg)  # (H,W,1)

    cls_w = tf.ones_like(y["classification"], dtype=tf.float32)
    sample_w = {"segmentation": seg_w, "classification": cls_w}

    return image, y, sample_w


# -------------------------
# Model builder
# -------------------------
def build_multitask_unet(img_size=128, output_classes=3, cls_classes=2, encoder_trainable=False):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[img_size, img_size, 3],
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
    down_stack.trainable = bool(encoder_trainable)

    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),   # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=[img_size, img_size, 3])

    # Encoder
    skips = down_stack(inputs)
    x = skips[-1]
    skip_connections = reversed(skips[:-1])

    # ---- Classification head
    cls = tf.keras.layers.GlobalAveragePooling2D()(x)
    cls = tf.keras.layers.Dropout(0.2)(cls)
    cls = tf.keras.layers.Dense(128, activation="relu")(cls)
    cls_logits = tf.keras.layers.Dense(cls_classes, name="classification")(cls)  # logits

    # ---- Segmentation decoder
    for up, skip in zip(up_stack, skip_connections):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    seg_logits = tf.keras.layers.Conv2DTranspose(
        filters=output_classes,
        kernel_size=3,
        strides=2,
        padding="same",
        name="segmentation"
    )(x)  # logits (img_size x img_size x output_classes)

    model = tf.keras.Model(inputs=inputs, outputs={"segmentation": seg_logits, "classification": cls_logits})
    return model


# -------------------------
# Train + Save (terminal)
# -------------------------
def train_and_save(
    save_path="pet_multi_task.keras",
    img_size=128,
    batch_size=64,
    buffer_size=1000,
    epochs=20,
    use_sample_weights=True,
    loss_weights=(1.0, 0.3),
    data_dir=None
):
    configure_gpu()

    dataset, info = tfds.load("oxford_iiit_pet", with_info=True, data_dir=data_dir)
    print("TFDS version:", info.version)
    print("Species names:", info.features["species"].names)  # ['cat','dog']

    train_len = info.splits["train"].num_examples
    steps_per_epoch = train_len // batch_size

    train_ds = dataset["train"].map(lambda x: load_image_multitask(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = dataset["test"].map(lambda x: load_image_multitask(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)

    train_batches = (
        train_ds
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .repeat()
        .map(AugmentMulti())
        .prefetch(tf.data.AUTOTUNE)
    )

    test_batches = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if use_sample_weights:
        train_for_fit = train_batches.map(add_sample_weights_multitask)
    else:
        train_for_fit = train_batches

    model = build_multitask_unet(img_size=img_size, output_classes=3, cls_classes=2, encoder_trainable=False)
    model.summary()

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
        loss_weights={"segmentation": float(loss_weights[0]), "classification": float(loss_weights[1])},
    )

    val_subsplits = 5
    validation_steps = info.splits["test"].num_examples // batch_size // val_subsplits

    history = model.fit(
        train_for_fit,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_batches,
        validation_steps=validation_steps,
    )

    model.save(save_path)
    print("✅ Saved model to:", os.path.abspath(save_path))
    return save_path


# -------------------------
# Load + Test (GUI selection)
# -------------------------
def load_and_test(model_path, image_path, img_size=128, species_names=("cat", "dog")):
    model = tf.keras.models.load_model(model_path, compile=False)

    img = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    img_arr = tf.keras.utils.img_to_array(img) / 255.0
    x = tf.expand_dims(img_arr, axis=0)

    pred = model.predict(x, verbose=0)

    seg_logits = pred["segmentation"][0]               # (H,W,3)
    seg_mask = tf.argmax(seg_logits, axis=-1).numpy()  # (H,W) values 0..2

    cls_logits = pred["classification"][0]             # (2,)
    cls_id = int(tf.argmax(cls_logits).numpy())
    cls_name = species_names[cls_id] if cls_id < len(species_names) else str(cls_id)

    # Show results in a new window (Matplotlib)
    pet_binary = (seg_mask == 0).astype(np.float32)  # class 0 = pet after mask -= 1

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_arr)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(seg_mask)
    plt.title("Predicted Mask (0..2)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img_arr)
    plt.imshow(pet_binary, alpha=0.5)
    plt.title(f"Overlay | Class: {cls_name}")
    plt.axis("off")

    plt.show()

    return cls_name


# -------------------------
# GUI
# -------------------------
DEFAULT_MODEL_NAME = "pet_multi_task.keras"
DEFAULT_IMG_SIZE = 128

def on_train_clicked(root):
    # Close GUI and start training in terminal
    root.destroy()

    # You can change defaults here if you want
    save_path = DEFAULT_MODEL_NAME
    try:
        out = train_and_save(
            save_path=save_path,
            img_size=DEFAULT_IMG_SIZE,
            batch_size=64,
            epochs=20,
            use_sample_weights=True
        )
        print("✅ Training finished. Model saved:", out)
    except Exception as e:
        print("❌ Training failed:", e)

    # End program
    sys.exit(0)


def on_test_clicked(root):
    # Choose model file
    model_path = filedialog.askopenfilename(
        title="Select a .keras model file",
        filetypes=[("Keras model", "*.keras"), ("All files", "*.*")]
    )
    if not model_path:
        return

    # Choose image
    image_path = filedialog.askopenfilename(
        title="Select an image to test",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
    )
    if not image_path:
        return

    # Hide GUI (optional)
    root.withdraw()

    try:
        cls_name = load_and_test(
            model_path=model_path,
            image_path=image_path,
            img_size=DEFAULT_IMG_SIZE,
            species_names=("cat", "dog")
        )
        messagebox.showinfo("Done", f"Prediction: {cls_name}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

    # End program after closing result window
    root.destroy()
    sys.exit(0)


def main():
    root = tk.Tk()
    root.title("Pet Multi-Task (Segmentation + Classification)")

    root.geometry("420x200")
    root.resizable(False, False)

    lbl = tk.Label(
        root,
        text="Choose an option:\nTrain (terminal) OR Test (pick model + image)",
        font=("Segoe UI", 11),
        pady=20
    )
    lbl.pack()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    train_btn = tk.Button(
        btn_frame,
        text="Train",
        width=15,
        height=2,
        command=lambda: on_train_clicked(root)
    )
    train_btn.grid(row=0, column=0, padx=10)

    test_btn = tk.Button(
        btn_frame,
        text="Test",
        width=15,
        height=2,
        command=lambda: on_test_clicked(root)
    )
    test_btn.grid(row=0, column=1, padx=10)

    hint = tk.Label(
        root,
        text=f"Default save name: {DEFAULT_MODEL_NAME}\nImage size: {DEFAULT_IMG_SIZE}x{DEFAULT_IMG_SIZE}",
        font=("Segoe UI", 9),
        fg="gray"
    )
    hint.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
