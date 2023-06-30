import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

model = tf.keras.models.load_model("diabetes_prediction_model.h5")

if __name__ == '__main__':
    print("Diabetes prediction model.")
    print("Pregnancies:")
    p = int(input(""))
    print("Glucose:")
    g = int(input(""))
    print("Blood Pressure:")
    bp = int(input(""))
    print("Skin thickness:")
    s = int(input(""))
    print("Insulin:")
    i = int(input(""))
    print("BMI:")
    bmi = float(input(""))
    print("Diabetes Percentage:")
    dp = float(input(""))
    print("Age:")
    a = int(input(""))

    result = model.predict([[p, g, bp, s, i, bmi, dp, a]])
    
    if result > 0.5:
        result = "Positive for diabetes."
    else:
        result = "Negative for diabetes"

    print("Result: ", result)
