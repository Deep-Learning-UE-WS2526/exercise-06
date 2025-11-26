import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop("Name", axis=1)
df = df.drop("PassengerId", axis=1)

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
def make_numeric(df):
  df["Sex"] = pd.factorize(df["Sex"])[0]
  df["Cabin"] = pd.factorize(df["Cabin"])[0]
  df["Ticket"] = pd.factorize(df["Ticket"])[0]
  df["Embarked"] = pd.factorize(df["Embarked"])[0]
  return df
df = make_numeric(df)

# 3. Remove all rows that contain missing values
df = df.dropna()

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
y = df["Survived"]
x = df.drop("Survived", axis=1)

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.1)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0, solver="liblinear")
classifier.fit(x_train, y_train)

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
y_pred = classifier.predict(x_test)

from sklearn.metrics import precision_score, recall_score, f1_score

print("precision: "+ str(precision_score(y_test, y_pred)))
print("recall: "+ str(recall_score(y_test, y_pred)))
print("f1: "+ str(f1_score(y_test, y_pred)))


# 5. Set up a network with a single hidden layer 


input_dim = x_train.shape[1]   

model1 = keras.Sequential()
model1.add(layers.Input(shape=(input_dim,)))
model1.add(layers.Dense(10, activation="softmax"))
model1.add(layers.Dense(2, activation="softmax"))

model1.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)

model1.fit(x_train, y_train, epochs=20, batch_size=16, verbose=0)

probs1 = model1.predict(x_test)
y_pred1 = np.argmax(probs1, axis=1)

print("Model 1 precision:", precision_score(y_test, y_pred1))
print("Model 1 recall   :", recall_score(y_test, y_pred1))
print("Model 1 f1       :", f1_score(y_test, y_pred1))


# 6. Create a network with 2 hidden layers 


model2 = keras.Sequential()
model2.add(layers.Input(shape=(input_dim,)))
model2.add(layers.Dense(20, activation="sigmoid"))
model2.add(layers.Dense(10, activation="relu"))
model2.add(layers.Dense(2, activation="softmax"))

model2.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)

model2.fit(x_train, y_train, epochs=20, batch_size=16, verbose=0)

probs2 = model2.predict(x_test)
y_pred2 = np.argmax(probs2, axis=1)

print("Model 2 precision:", precision_score(y_test, y_pred2))
print("Model 2 recall   :", recall_score(y_test, y_pred2))
print("Model 2 f1       :", f1_score(y_test, y_pred2))

