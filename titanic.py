import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

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

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.1)

import keras

# 1. Set up a network with a single hidden layer of size 10. The hidden and the output layer use the `softmax` activation function. Test and evaluate.

model1 = keras.Sequential()
model1.add(keras.layers.InputLayer(shape=(x_train.shape[1],)))
model1.add(keras.layers.Dense(10, activation='softmax'))
model1.add(keras.layers.Dense(2, activation='softmax'))

model1.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model1.fit(x_train, y_train, epochs=50, batch_size=10)

results = model1.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Calculate F1-score manually


y_pred = model1.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred_classes)
print(f"F1-Score: {f1:.4f}")


# 2. Create a network with 2 hidden layers of sizes 20 and 10. The first layer uses a `sigmoid` activation, the second one `relu` (output layer should use `softmax` again).

model2 = keras.Sequential()
model2.add(keras.layers.InputLayer(shape=(x_train.shape[1],)))
model2.add(keras.layers.Dense(20, activation='sigmoid'))
model2.add(keras.layers.Dense(10, activation='relu'))
model2.add(keras.layers.Dense(2, activation='softmax'))

model2.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model2.fit(x_train, y_train, epochs=50, batch_size=20)

results2 = model2.evaluate(x_test, y_test)
print(f"Test Loss: {results2[0]}, Test Accuracy: {results2[1]}")

# Calculate F1-score manually
y_pred2 = model2.predict(x_test)
y_pred_classes2 = np.argmax(y_pred2, axis=1)
f1_2 = f1_score(y_test, y_pred_classes2)
print(f"F1-Score: {f1_2:.4f}")

# 3. Experiment with different settings. Can you increase the f-score above 56%? If not, do you have an idea why not?

model3 = keras.Sequential()
model3.add(keras.layers.InputLayer(shape=(x_train.shape[1],)))
model3.add(keras.layers.Dense(20, activation='sigmoid'))
model3.add(keras.layers.Dense(20, activation='relu'))
model3.add(keras.layers.Dense(10, activation='sigmoid'))
model3.add(keras.layers.Dense(5, activation='relu'))
model3.add(keras.layers.Dense(2, activation='softmax'))

#Hier lag das problem. meanSquare ist gut für regressionsprobleme, das problem ist aber ein klassifikationsproblem also ist SCC besser.
model3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model3.fit(x_train, y_train, epochs=60, batch_size=20)

results3 = model3.evaluate(x_test, y_test)
print(f"Test Loss: {results3[0]}, Test Accuracy: {results3[1]}")

# Calculate F1-score manually
y_pred3 = model3.predict(x_test)
y_pred_classes3 = np.argmax(y_pred3, axis=1)
f1_3 = f1_score(y_test, y_pred_classes3)
print(f"F1-Score: {f1_3:.4f}")






# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.



# # 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
# from sklearn.linear_model import LogisticRegression

# classifier = LogisticRegression(random_state=0, solver="liblinear")
# classifier.fit(x_train, y_train)

# # 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
# y_pred = classifier.predict(x_test)

# from sklearn.metrics import precision_score, recall_score, f1_score

# print("precision: "+ str(precision_score(y_test, y_pred)))
# print("recall: "+ str(recall_score(y_test, y_pred)))
# print("f1: "+ str(f1_score(y_test, y_pred)))

