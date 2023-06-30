import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# Preproccess data
diabetes = pd.read_csv(r"...\diabetes.csv")

X = diabetes.drop("Outcome", axis=1)
y = diabetes["Outcome"]

print(X.head())
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

tf.random.set_seed(42)

# Define model architecture.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=100)

# Model tests
Y_pred = model.predict(X_test)

Y_pred_classes = [1 if prob > 0.5 else 0 for prob in Y_pred]

confusion_mtx = confusion_matrix(y_test, Y_pred_classes) 

# Compute classification report
classificationReport = classification_report(y_test, Y_pred_classes)

print("Confusion Matrix:")
print(confusion_mtx)

print("\nClassification Report:")
print(classificationReport)

model.save('diabetes_prediction_model.h5')
