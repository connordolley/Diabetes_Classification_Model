import os
import random
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def gather_data():
    print("Please provide the following information:")
    pregnncy = int(input("Pregnancies: "))
    glucose = int(input("Glucose: "))
    blood_pressure = int(input("Blood Pressure: "))
    skin_thickness = int(input("Skin thickness: "))
    insulin = int(input("Insulin: "))
    bmi = float(input("BMI: "))
    diabetes_percentage = float(input("Diabetes Percentage: "))
    age = int(input("Age: "))

    patient_data = [pregnncy, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_percentage, age]
    
    return patient_data

def analyze_data(model, patient_data):
    result = model.predict([patient_data])
    
    if result > 0.5:
        result = "Positive for diabetes."
    else:
        result = "Negative for diabetes."
    
    return result

def main():
    print("Diabetes prediction model.")
    model = tf.keras.models.load_model("diabetes_prediction_model.h5")
    patient_data = gather_data()
    result = analyze_data(model, patient_data)
    print("Result: ", result)

if __name__ == '__main__':
    main()
