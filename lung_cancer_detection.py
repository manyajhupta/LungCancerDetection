import numpy as np
import pandas as pd
import pydicom as dicom
import os
import matplotlib.pyplot as plt
import cv2
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses
from sklearn.metrics import confusion_matrix

from tkinter import *
from tkinter import messagebox, ttk
import tkinter as tk
from PIL import Image, ImageTk


class LungCancerDetection:
    def __init__(self, root):
        self.root = root
        # Window size
        self.root.geometry("1006x500+0+0")
        self.root.resizable(False, False)
        self.root.title("Lung Cancer Detection")

        img4 = Image.open(r"Images\Lung-Cancer-Detection.jpg")
        img4 = img4.resize((1006, 500), Image.LANCZOS)
        self.photoimg4 = ImageTk.PhotoImage(img4)

        bg_img = Label(self.root, image=self.photoimg4)
        bg_img.place(x=0, y=50, width=1006, height=500)

        # Title Label
        title_lbl = Label(
            text="Lung Cancer Detection",
            font=("Bradley Hand ITC", 30, "bold"),
            bg="black",
            fg="white",
        )
        title_lbl.place(x=0, y=0, width=1006, height=50)

        # Button 1
        self.b1 = Button(
            text="Import Data",
            cursor="hand2",
            command=self.import_data,
            font=("Times New Roman", 15, "bold"),
            bg="white",
            fg="black",
        )
        self.b1.place(x=80, y=130, width=180, height=30)

        # Button 2
        self.b2 = Button(
            text="Pre-Process Data",
            cursor="hand2",
            command=self.preprocess_data,
            font=("Times New Roman", 15, "bold"),
            bg="white",
            fg="black",
        )
        self.b2.place(x=80, y=180, width=180, height=30)
        self.b2["state"] = "disabled"
        self.b2.config(cursor="arrow")

        # Button 3
        self.b3 = Button(
            text="Train Data",
            cursor="hand2",
            command=self.train_data,
            font=("Times New Roman", 15, "bold"),
            bg="white",
            fg="black",
        )
        self.b3.place(x=80, y=230, width=180, height=30)
        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")

    def import_data(self):
        """Import data from DICOM scans and labels CSV"""
        # Data directory
        self.data_directory = "sample_images/"
        self.lung_patients = os.listdir(self.data_directory)

        # Read labels csv
        self.labels = pd.read_csv("stage1_labels.csv", index_col=0)

        # Setting x*y size to 10
        self.size = 10

        # Setting z-dimension (number of slices to 5)
        self.num_slices = 5

        messagebox.showinfo("Import Data", "Data Imported Successfully!")

        self.b1["state"] = "disabled"
        self.b1.config(cursor="arrow")
        self.b2["state"] = "normal"
        self.b2.config(cursor="hand2")

    def preprocess_data(self):
        """Preprocess DICOM data into numpy arrays"""

        def chunks(l, n):
            count = 0
            for i in range(0, len(l), n):
                if count < self.num_slices:
                    yield l[i : i + n]
                    count = count + 1

        def mean(l):
            return sum(l) / len(l)

        def data_processing(patient, labels_df, size=10, noslices=5):
            label = labels_df.loc[patient, "cancer"]
            path = self.data_directory + patient
            slices = [
                dicom.read_file(path + "/" + s) for s in os.listdir(path)
            ]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

            new_slices = []
            slices = [
                cv2.resize(np.array(each_slice.pixel_array), (size, size))
                for each_slice in slices
            ]

            chunk_sizes = math.floor(len(slices) / noslices)
            for slice_chunk in chunks(slices, chunk_sizes):
                slice_chunk = list(map(mean, zip(*slice_chunk)))
                new_slices.append(slice_chunk)

            if label == 1:  # Cancer Patient
                label = np.array([0, 1])
            elif label == 0:  # Non Cancerous Patient
                label = np.array([1, 0])
            return np.array(new_slices), label

        image_data = []
        # Check if Data Labels is available in CSV or not
        for num, patient in enumerate(self.lung_patients):
            if num % 50 == 0:
                print("Saved -", num)
            try:
                img_data, label = data_processing(
                    patient, self.labels, size=self.size, noslices=self.num_slices
                )
                image_data.append([img_data, label, patient])
            except KeyError as e:
                print("Data is unlabeled")

        # Save results
        np.save(
            "imageDataNew-{}-{}-{}.npy".format(
                self.size, self.size, self.num_slices
            ),
            image_data,
        )

        messagebox.showinfo(
            "Pre-Process Data", "Data Pre-Processing Done Successfully!"
        )

        self.b2["state"] = "disabled"
        self.b2.config(cursor="arrow")
        self.b3["state"] = "normal"
        self.b3.config(cursor="hand2")

    def train_data(self):
        """Train the 3D CNN model"""

        image_data = np.load("imageDataNew-10-10-5.npy", allow_pickle=True)
        training_data = image_data[0:45]
        validation_data = image_data[45:50]

        training_data_label = Label(
            text="Total Training Data: " + str(len(training_data)),
            font=("Times New Roman", 13, "bold"),
            bg="black",
            fg="white",
        )
        training_data_label.place(x=750, y=150, width=200, height=18)

        validation_data_label = Label(
            text="Total Validation Data: " + str(len(validation_data)),
            font=("Times New Roman", 13, "bold"),
            bg="black",
            fg="white",
        )
        validation_data_label.place(x=750, y=190, width=200, height=18)

        size = 10
        num_slices = 5

        # Build model using Keras
        def build_cnn_model():
            model = models.Sequential()
            
            # Input layer
            model.add(layers.InputLayer(input_shape=(size, size, num_slices, 1)))
            
            # Conv block 1
            model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
            model.add(layers.MaxPooling3D((2, 2, 2), padding='same'))
            
            # Conv block 2
            model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
            model.add(layers.MaxPooling3D((2, 2, 2), padding='same'))
            
            # Conv block 3
            model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
            model.add(layers.MaxPooling3D((2, 2, 2), padding='same'))
            
            # Conv block 4
            model.add(layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same'))
            model.add(layers.MaxPooling3D((2, 2, 2), padding='same'))
            
            # Conv block 5
            model.add(layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same'))
            model.add(layers.MaxPooling3D((2, 2, 2), padding='same'))
            
            # Flatten and dense layers
            model.add(layers.Flatten())
            model.add(layers.Dense(256, activation='relu'))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(2, activation='softmax'))
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=1e-3),
                loss=losses.CategoricalCrossentropy(),
                metrics=['accuracy']
            )
            
            return model

        # Prepare training data
        X_train = np.array([i[0] for i in training_data]).reshape(-1, size, size, num_slices, 1)
        y_train = np.array([i[1] for i in training_data])
        
        X_val = np.array([i[0] for i in validation_data]).reshape(-1, size, size, num_slices, 1)
        y_val = np.array([i[1] for i in validation_data])

        # Build and train model
        model = build_cnn_model()
        
        epochs = 100
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )

        # Evaluate model
        final_accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
        
        final_accuracy_label = Label(
            text="Final Accuracy: " + str(round(final_accuracy, 4)),
            font=("Times New Roman", 13, "bold"),
            bg="black",
            fg="white",
        )
        final_accuracy_label.place(x=750, y=230, width=200, height=18)

        # Get predictions
        predictions = model.predict(X_val)
        predicted_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(y_val, axis=1)

        patients = []
        actual = []
        predicted = []

        for i in range(len(validation_data)):
            patients.append(validation_data[i][2])

        for i in predicted_classes:
            if i == 1:
                predicted.append("Cancer")
            else:
                predicted.append("No Cancer")

        for i in actual_classes:
            if i == 1:
                actual.append("Cancer")
            else:
                actual.append("No Cancer")

        for i in range(len(patients)):
            print("----------------------------------------------------")
            print("Patient: ", patients[i])
            print("Actual: ", actual[i])
            print("Predicted: ", predicted[i])
            print("----------------------------------------------------")

        # Confusion matrix
        y_actual = pd.Series(actual_classes, name="Actual")
        y_predicted = pd.Series(predicted_classes, name="Predicted")

        df_confusion = pd.crosstab(y_actual, y_predicted).reindex(
            columns=[0, 1], index=[0, 1], fill_value=0
        )
        print("Confusion Matrix:\n")
        print(df_confusion)

        prediction_label = Label(
            text=">>>>    P R E D I C T I O N    <<<<",
            font=("Times New Roman", 14, "bold"),
            bg="#778899",
            fg="black",
        )
        prediction_label.place(x=0, y=458, width=1006, height=20)

        result1 = []

        for i in range(len(validation_data)):
            result1.append(patients[i])
            if y_actual[i] == 1:
                result1.append("Cancer")
            else:
                result1.append("No Cancer")

            if y_predicted[i] == 1:
                result1.append("Cancer")
            else:
                result1.append("No Cancer")

        total_rows = int(len(patients))
        total_columns = int(len(result1) / len(patients))

        heading = ["Patient: ", "Actual: ", "Predicted: "]

        self.root.geometry("1006x" + str(500 + (len(patients) * 20) - 20) + "+0+0")
        self.root.resizable(False, False)

        for i in range(total_rows):
            for j in range(total_columns):
                self.e = Entry(
                    root, width=42, fg="black", font=("Times New Roman", 12, "bold")
                )
                self.e.grid(row=i, column=j)
                self.e.place(x=(j * 335), y=(478 + i * 20))
                self.e.insert(END, heading[j] + result1[j + i * 3])
                self.e["state"] = "disabled"
                self.e.config(cursor="arrow")

        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")

        messagebox.showinfo("Train Data", "Model Trained Successfully!")

        # Plot confusion matrix
        def plot_confusion_matrix(df_confusion, title="Confusion matrix", cmap=plt.cm.gray_r):
            plt.matshow(df_confusion, cmap=cmap)
            plt.colorbar()
            tick_marks = np.arange(len(df_confusion.columns))
            plt.title(title)
            plt.xticks(tick_marks, df_confusion.columns, rotation=45)
            plt.yticks(tick_marks, df_confusion.index)
            plt.ylabel(df_confusion.index.name)
            plt.xlabel(df_confusion.columns.name)
            plt.show()

        plot_confusion_matrix(df_confusion)


# For GUI
if __name__ == "__main__":
    root = Tk()
    obj = LungCancerDetection(root)
    root.mainloop()
