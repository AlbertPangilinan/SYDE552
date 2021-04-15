# Name: Ray Albert Pangilinan
# Student ID: 20661046
# SYDE 552 - Winter 2021
# Final Project

import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_epochs = 10
num_runs = 10
training_split = 0.8
genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def get_mfcc(song):
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)
    mfcc /= np.amax(np.absolute(mfcc))
    return np.ndarray.flatten(mfcc)[:25000]


def onehot_encoding(labels):
    uniq_ids, row_ids = np.unique(labels, return_inverse=True)
    row_ids = row_ids.astype(np.int32, copy=False)
    return tf.keras.utils.to_categorical(row_ids, len(uniq_ids))


def import_data(path, genres):
    features, labels = [], []

    for genre in genres:
        audio_files = glob.glob(path + genre + "/*.wav")
        print("Genre: " + genre)
        for i in range(len(audio_files)):
            song = audio_files[i]
            print("Importing song " + str(i + 1))
            mfcc = get_mfcc(song)
            features.append(mfcc)
            labels.append(genre)

    return np.stack(features), onehot_encoding(labels)


all_accuracies_gtzan, all_accuracies_self = [], []

# Import and prepare GTZAN dataset (training and test data)
features_gtzan, labels = import_data("GTZAN/Data/genres_original/", genres)

data_gtzan = np.column_stack((features_gtzan, labels))
np.random.shuffle(data_gtzan)
split_i = int(len(data_gtzan) * training_split)
train, test_gtzan = data_gtzan[:split_i, :], data_gtzan[split_i:, :]

train_input = train[:, :-10]
train_labels = train[:, -10:]

test_gtzan_input = test_gtzan[:, :-10]
test_gtzan_labels = test_gtzan[:, -10:]

# Import and prepare self-compiled dataset (test data only)
features_self, labels_self = import_data("self_compiled/", genres)

data_self = np.column_stack((features_self, labels_self))
np.random.shuffle(data_self)

test_self_input = data_self[:, :-10]
test_self_labels = data_self[:, -10:]

# Define models
models = [
    tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                128, input_dim=np.shape(train_input)[1], activation="relu"
            ),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    ),
    tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                128, input_dim=np.shape(train_input)[1], activation="relu"
            ),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    ),
    tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                128, input_dim=np.shape(train_input)[1], activation="relu"
            ),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    ),
    tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                128, input_dim=np.shape(train_input)[1], activation="sigmoid"
            ),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    ),
    tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                128, input_dim=np.shape(train_input)[1], activation="sigmoid"
            ),
            tf.keras.layers.Dense(64, activation="sigmoid"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    ),
    tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                128, input_dim=np.shape(train_input)[1], activation="sigmoid"
            ),
            tf.keras.layers.Dense(64, activation="sigmoid"),
            tf.keras.layers.Dense(32, activation="sigmoid"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    ),
]

for model in models:
    model_accuracy_gtzan, model_accuracy_self = [], []
    for i in range(num_runs):
        print("Run " + str(i + 1))

        # Compile model and print summary
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        print(model.summary())

        # Train and test model on GTZAN dataset for 10 epochs
        model.fit(
            train_input,
            train_labels,
            epochs=num_epochs,
            validation_data=(test_gtzan_input, test_gtzan_labels),
        )
        model_accuracy_gtzan.append(model.history.history["val_accuracy"])

        # Test model on self-compiled dataset
        loss, acc = model.evaluate(test_self_input, test_self_labels)
        model_accuracy_self.append(acc)

    all_accuracies_gtzan.append(model_accuracy_gtzan)
    all_accuracies_self.append(model_accuracy_self)

# Create epoch steps for x-axes
epoch_x_axes = list(range(1, num_epochs + 1))

# Plot charts for GTZAN dataset

# Plot test accuracy for all runs for each model
for i in range(len(models)):
    accuracy = all_accuracies_gtzan[i]

    # Plot test accuracy
    for j in range(num_runs):
        plt.plot(epoch_x_axes, accuracy[j], label="Run " + str(j + 1))

    plt.title("Test Accuracy for Model " + str(i + 1) + " - (GTZAN Dataset, 10 Runs)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# Plot avg. test accuracy for each model
avg_accuracies_gtzan = list(
    map(lambda accuracy: np.mean(accuracy), all_accuracies_gtzan)
)
plt.scatter(list(range(1, len(models) + 1)), avg_accuracies_gtzan)
plt.title("Average Test Accuracies for All Models (GTZAN Dataset)")
plt.xlabel("Model Number")
plt.ylabel("Avg. Accuracy")
plt.show()

# Plot charts for self-compiled dataset

print(len(all_accuracies_self))

# Plot test accuracy for all runs for each model
for i in range(len(models)):
    accuracy = all_accuracies_self[i]
    plt.plot(epoch_x_axes, accuracy, label="Model " + str(i + 1))

plt.title("Test Accuracy for All Models - (Self-Compiled Dataset, 10 Runs)")
plt.xlabel("Run")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot avg. test accuracy for each model
avg_accuracies_self = list(map(lambda accuracy: np.mean(accuracy), all_accuracies_self))
plt.scatter(list(range(1, len(models) + 1)), avg_accuracies_self)
plt.title("Average Test Accuracies for All Models (Self-Compiled Dataset)")
plt.xlabel("Model Number")
plt.ylabel("Avg. Accuracy")
plt.show()
