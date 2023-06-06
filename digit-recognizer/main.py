# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# %%
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import os

from mylib import plot_model_history

FILES_PATH = "./files/input"

print(os.environ.get("HW_ENV"))
for dirname, _, filenames in os.walk(FILES_PATH):
    for filename in filenames:
        local = True
        # print(os.path.join(dirname, filename))

if not local:
    FILES_PATH = "/kaggle/input/digit-recognizer"

train_set = pd.read_csv(f"{FILES_PATH}/train.csv")
test_set = pd.read_csv(f"{FILES_PATH}/test.csv")
# test_set.describe()

train_x, train_y = train_set.iloc[:, 1:], train_set.iloc[:, :1]

train_x = train_x.to_numpy()
train_y = train_y.to_numpy()
test_x = test_set.to_numpy()

train_x = train_x.astype("float32") / 255.0
test_x = test_x.astype("float32") / 255.0

label_binarizer = LabelBinarizer()
train_y = label_binarizer.fit_transform(train_y)

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, train_size=0.8)

train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
valid_x = valid_x.reshape((valid_x.shape[0], 28, 28, 1))
# print(train_x.shape)

sequential_model = Sequential()
sequential_model.add(layers.Input(shape=(28, 28, 1)))

sequential_model.add(
    layers.Conv2D(kernel_size=(3, 3), padding="same", strides=(2, 2), filters=16))
sequential_model.add(layers.ReLU())
sequential_model.add(layers.BatchNormalization())
sequential_model.add(
    layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")
)
sequential_model.add(layers.Dropout(rate=0.5))

sequential_model.add(layers.Flatten())
# sequential_model.add(layers.Dense(256, activation="sigmoid"))
sequential_model.add(layers.Dense(128, activation="sigmoid"))
sequential_model.add(layers.Dense(10, activation="softmax"))
sequential_model.summary()

sequential_model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
history = sequential_model.fit(
    train_x,
    train_y,
    validation_data=(valid_x, valid_y),
    epochs=10,
    batch_size=256,
    verbose=0,
)

# plot_model_history(history, 'loss', [0., 2.0])
plot_model_history(history, 'accuracy')

# %%
_, accuracy = sequential_model.evaluate(valid_x, valid_y, verbose=0)
print(f"Accuracy:{accuracy}")

r = sequential_model.predict(test_x)
print(r.shape)
# print(r)
size = 28000
answer = [np.argmax(row) for row in r]
print(answer[:100])

answer = {"ImageId": list(range(1, size + 1)), "Label": answer}

answer = pd.DataFrame.from_dict(answer)
# print(answer)

answer.to_csv("answer.csv", index=False)

# %%
