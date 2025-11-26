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

""" print("precision: "+ str(precision_score(y_test, y_pred)))
print("recall: "+ str(recall_score(y_test, y_pred)))
print("f1: "+ str(f1_score(y_test, y_pred))) """

#-------------------------------------------------------------------------------

import numpy as np 
import keras as kd 
from keras import layers 
from keras.metrics import F1Score
from keras.models import Sequential
from keras.layers import Dense
 

#Ich benutze das Sequential aus Keras
model = kd.Sequential() 
model.add(layers.Input(shape=(9,))) #9 Features, wieso ist das wichtig? 
model.add(layers.Dense(10, activation="softmax")) 
model.add(layers.Dense(2, activation="softmax"))



model.compile(
    loss="sparse_categorical_crossentropy",  # oder categorical_crossentropy
    optimizer="sgd",
    metrics=["accuracy"]
)

#Wie hängt die Compile Mehtod mit der aktivation zusammen? Optimizer ist ja nur ein String, was sind loss und metrics?

model.fit(x_train, y_train, epochs=10, batch_size=5)

# Was sind epochs, was sind batch sizes? 

y_pred_keras_1 = model.predict(x_test, batch_size=None, verbose="auto", steps=None, callbacks=None)
y_pred_keras_1_final = np.argmax(y_pred_keras_1, axis=1)      # (n_samples,)



print("f1: "+ str(f1_score(y_test, y_pred_keras_1_final)))



#-------------------------------------------------------------------------------

model_two = kd.Sequential() 
model_two.add(layers.Input(shape=(9,)))
model_two.add(layers.Dense(20, activation="sigmoid")) 
model_two.add(layers.Dense(10, activation="relu")) 
model_two.add(layers.Dense(2, activation="softmax"))



model_two.compile(
    loss="sparse_categorical_crossentropy",  # oder categorical_crossentropy
    optimizer="sgd",
    metrics=["accuracy"]
)
model_two.fit(x_train, y_train, epochs=10, batch_size=5)

y_pred_keras_2 = model_two.predict(x_test, batch_size=None, verbose="auto", steps=None, callbacks=None)
y_pred_keras_2_final = np.argmax(y_pred_keras_2, axis=1)      # (n_samples,)


print("f1: "+ str(f1_score(y_test, y_pred_keras_2_final)))






