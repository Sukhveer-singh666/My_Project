"""Standalone training script - reproduces Train_plant_disease.ipynb."""
import json
import tensorflow as tf

DATA_DIR = "New Plant Diseases Dataset(Augmented)"

# Data
training_set = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
)
validation_set = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/valid",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
)

# Prefetch for speed
AUTOTUNE = tf.data.AUTOTUNE
training_set = training_set.prefetch(AUTOTUNE)
validation_set = validation_set.prefetch(AUTOTUNE)

# Model (identical to notebook)
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=[128, 128, 3]))
cnn.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(2, strides=2))
cnn.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(2, strides=2))
cnn.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(2, strides=2))
cnn.add(tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(256, 3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(2, strides=2))
cnn.add(tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.25))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(1500, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.4))
cnn.add(tf.keras.layers.Dense(38, activation='softmax'))

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
cnn.summary()

# Train
history = cnn.fit(
    x=training_set,
    validation_data=validation_set,
    epochs=10,
)

# Evaluate
train_loss, train_acc = cnn.evaluate(training_set)
val_loss, val_acc = cnn.evaluate(validation_set)
print(f"Training accuracy: {train_acc}")
print(f"Validation accuracy: {val_acc}")

# Save
cnn.save('trained_plant_disease_model.keras')
with open('training_hist.json', 'w') as f:
    json.dump(history.history, f)
print("Model saved to trained_plant_disease_model.keras")
