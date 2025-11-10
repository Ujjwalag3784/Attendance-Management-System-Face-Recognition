import csv
import os, cv2
import numpy as np
import pandas as pd
import datetime
import time
from PIL import ImageTk, Image


# Train Image
def _send_feedback(message, text, text_to_speech):
    if message and hasattr(message, "configure"):
        try:
            message.configure(text=text)
        except Exception:
            pass
    if text_to_speech:
        try:
            text_to_speech(text)
        except Exception:
            pass


def TrainImage(
    haarcasecade_path,
    trainimage_path,
    trainimagelabel_path,
    message=None,
    text_to_speech=None,
):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(haarcasecade_path)
    faces, Id = getImagesAndLables(trainimage_path)
    if not faces or not Id:
        res = "No training images found. Please capture faces first."
        _send_feedback(message, res, text_to_speech)
        return
    recognizer.train(faces, np.array(Id))
    recognizer.save(trainimagelabel_path)
    res = "Image Trained successfully"  # +",".join(str(f) for f in Id)
    _send_feedback(message, res, text_to_speech)


def getImagesAndLables(path):
    imagePath = []
    # Walk through the training folder so we support both nested directories and loose files
    for root, _, files in os.walk(path):
        for file_name in files:
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                imagePath.append(os.path.join(root, file_name))

    faces = []
    Ids = []
    for imagePath in imagePath:
        pilImage = Image.open(imagePath).convert("L")
        imageNp = np.array(pilImage, "uint8")
        file_name = os.path.split(imagePath)[-1]
        try:
            Id = int(file_name.split("_")[0])
        except (IndexError, ValueError):
            # Skip files that do not follow the expected naming convention
            continue
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids
