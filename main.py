from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import pickle
from google.colab import drive
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

drive.mount('/content/drive')


if __name__ == "__main__":
    X_list, y_list = [], []

    # Load pickle files
    data_path = "/content/drive/MyDrive/tih/"
    for i in range(3):
        filename = data_path + f"data{i}.pickle"
        with open(filename, "rb") as f:
            data = pickle.load(f)
            X = data['x_train']
            y = data['y_train']

            # Fixing the channel size, converting grayscale to RGB
            if X.shape[1] == 1:
                X = np.repeat(X, 3, axis=1)

            # Converting from (channels, height, width) to (height, width, channels)
            X = X.transpose(0, 2, 3, 1)

            X_list.append(X)
            y_list.append(y)

    # Dataset generator
    def generator():
        for X, y in zip(X_list, y_list):
            for i in range(len(X)):
                # normalize here
                yield X[i].astype(np.float32) / 255.0, int(y[i])

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.int64),
        output_shapes=((32, 32, 3), ())
    )

    # Shuffle and split
    total_size = sum(len(y) for y in y_list)
    train_size = int(0.8 * total_size)

    train_dataset = dataset.take(train_size).batch(
        32).repeat().prefetch(tf.data.AUTOTUNE)
    val_dataset = dataset.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = train_size // 32

    unique_labels = np.unique(np.concatenate(y_list))

    # Index → class mapping
    index_to_class = {i: int(cls) for i, cls in enumerate(unique_labels)}

    # Save mapping
    with open("/content/drive/MyDrive/tih/class_mapping.pkl", "wb") as f:
        pickle.dump(index_to_class, f)

    print("✅ Class mapping saved:", index_to_class)

    # Model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(set(np.concatenate(y_list))), activation='softmax')
    ])

    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callback to save after every epoch
    checkpoint_path = "/content/drive/MyDrive/checkpoints/model_epoch_{epoch:02d}.keras"
    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq="epoch",
        save_best_only=False,   # every epoch saved
        save_weights_only=False,
        verbose=1
    )

    # Training
    history = model.fit(train_dataset,
                        epochs=10,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_dataset)

    # Evaluation
    loss, acc = model.evaluate(val_dataset)
    print(f"Validation accuracy: {acc:.4f}")

    # Save model
    model.save('/content/drive/MyDrive/tih/final_trained_model.keras')
