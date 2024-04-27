
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

training_data = tf.keras.utils.image_dataset_from_directory(
    "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    verbose=True,
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    verbose=True,
)

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Dropout,Input, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Input(shape=(128,128, 3)))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(
    units=2000,
    activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(
    units=38,
    activation='softmax'
))

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

training_history = model.fit(x=training_data, validation_data=validation_data, epochs=10)

train_loss, train_acc = model.evaluate(training_data)

print(train_loss, train_acc)

val_loss, val_acc = model.evaluate(validation_data)

print(val_loss, val_acc)

model.save('trained_detector.keras')

epochs = [i for i in range (1,11)]
epochs

plt.plot(epochs, training_history.history['accuracy'], label="Trainig Acc")
plt.plot(epochs, training_history.history['loss'], label="Trainig Loss")
plt.plot(epochs, training_history.history['val_accuracy'], label="Validation Acc")
plt.plot(epochs, training_history.history['val_loss'], label="Validation Loss")
plt.xlabel("No of epochs")
plt.ylabel("Training History")
plt.legend()
plt.show()

import json
with open("Training_history.json", "w") as f:
    json.dump(training_history.history, f)

test_data = tf.keras.utils.image_dataset_from_directory(
    "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    verbose=True,
)

y_pred = model.predict(test_data)
print(y_pred)

predicted_index_location = tf.argmax(y_pred, axis=1)
print(predicted_index_location)

true_values = tf.concat([y for x,y in test_data], axis=0)
expected_values = tf.argmax(true_values, axis=1)
print(expected_values)

class_names = test_data.class_names
print(class_names)

from sklearn.metrics import classification_report, confusion_matrix

classifier_report = classification_report(expected_values, predicted_index_location, target_names=class_names, output_dict=False)
print(classifier_report)

cm = confusion_matrix(expected_values, predicted_index_location)
print(cm)

plt.figure(figsize=(40,40))
sns.heatmap(cm, annot=True)
plt.xlabel("Actual Category")
plt.ylabel("Predicted Category")

