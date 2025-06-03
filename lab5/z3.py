import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
import shutil
import random
from pathlib import Path
import shutil

shutil.rmtree("data", ignore_errors=True)

def prepare_data(src_dir, train_dir, val_dir, split=0.8):
    Path(train_dir, "cats").mkdir(parents=True, exist_ok=True)
    Path(train_dir, "dogs").mkdir(parents=True, exist_ok=True)
    Path(val_dir, "cats").mkdir(parents=True, exist_ok=True)
    Path(val_dir, "dogs").mkdir(parents=True, exist_ok=True)

    for label in ["cat", "dog"]:
        images = list(Path(src_dir).glob(f"{label}.*.jpg"))
        random.shuffle(images)
        split_idx = int(len(images) * split)
        for img in images[:split_idx]:
            shutil.copy(img, Path(train_dir, label + "s", img.name))
        for img in images[split_idx:]:
            shutil.copy(img, Path(val_dir, label + "s", img.name))

prepare_data(
    src_dir="dogs-cats-mini",
    train_dir="data/train",
    val_dir="data/val",
    split=0.8
)

train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "data/train", target_size=(150, 150), batch_size=32, class_mode='binary'
)
val_data = val_gen.flow_from_directory(
    "data/val", target_size=(150, 150), batch_size=32, class_mode='binary', shuffle='false'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, validation_data=val_data, epochs=10)

model.save('cat_dog_model.keras')

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Dokładność')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Strata')
plt.legend()
plt.show()

val_data.reset()
preds = (model.predict(val_data) > 0.5).astype("int32")
true_labels = val_data.classes

print(confusion_matrix(true_labels, preds))
print(classification_report(true_labels, preds, target_names=["cat", "dog"]))

import matplotlib.pyplot as plt

filenames = val_data.filenames
errors = np.where(preds.reshape(-1) != true_labels)[0]

for i in errors[:5]:
    img_path = os.path.join("data/val", filenames[i])
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    pred_label = 'dog' if preds[i][0] == 1 else 'cat'
    true_label = 'dog' if true_labels[i] == 1 else 'cat'
    plt.title(f'Prawda: {true_label}, Pred: {pred_label}')
    plt.show()