import csv
import os, cv2
import numpy as np
import pandas as pd
import datetime
import time
from pathlib import Path



# take Image of user
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


def TakeImage(
    l1,
    l2,
    haarcasecade_path,
    trainimage_path,
    studentdetail_path,
    message=None,
    err_screen=None,
    text_to_speech=None,
    show_window=True,
    target_samples=50,
    timeout=25,
):
    if (l1 == "") and (l2==""):
        t='Please Enter the your Enrollment Number and Name.'
        _send_feedback(message, t, text_to_speech)
    elif l1=='':
        t='Please Enter the your Enrollment Number.'
        _send_feedback(message, t, text_to_speech)
    elif l2 == "":
        t='Please Enter the your Name.'
        _send_feedback(message, t, text_to_speech)
    else:
        try:
            enrollment_id = l1.strip()
            student_name = l2.strip()
            safe_name = student_name.replace(" ", "_")
            train_dir = Path(trainimage_path)
            train_dir.mkdir(parents=True, exist_ok=True)
            student_dir = train_dir / f"{enrollment_id}_{safe_name}"
            student_dir.mkdir(parents=True, exist_ok=True)

            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier(haarcasecade_path)
            if detector.empty():
                t = "Cannot load haarcascade file."
                _send_feedback(message, t, text_to_speech)
                return
            if not cam.isOpened():
                t = "Cannot access the camera."
                _send_feedback(message, t, text_to_speech)
                return

            sampleNum = 0
            start_time = time.time()
            while True:
                ret, img = cam.read()
                if not ret:
                    if time.time() - start_time > 2:
                        break
                    else:
                        continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    if show_window:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sampleNum += 1
                    filename = f"{enrollment_id}_{safe_name}_{sampleNum}.jpg"
                    cv2.imwrite(
                        str(student_dir / filename),
                        gray[y : y + h, x : x + w],
                    )

                if show_window:
                    cv2.imshow("Frame", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    time.sleep(0.01)

                if sampleNum >= target_samples:
                    break
                if not show_window and (time.time() - start_time) > timeout:
                    _send_feedback(
                        message, "Timeout reached. Please try again.", text_to_speech
                    )
                    break

            cam.release()
            if show_window:
                cv2.destroyAllWindows()
            
            if sampleNum == 0:
                t = "No face detected. Please try again."
                _send_feedback(message, t, text_to_speech)
                if student_dir.exists() and not any(student_dir.iterdir()):
                    student_dir.rmdir()
                return

            row = [enrollment_id, student_name]
            student_csv_path = Path(studentdetail_path)
            student_csv_path.parent.mkdir(parents=True, exist_ok=True)
            student_csv_path.touch(exist_ok=True)
            with student_csv_path.open("a+", newline="") as csvFile:
                csvFile.seek(0)
                reader = csv.reader(csvFile)
                existing = {rows[0] for rows in reader if rows}
                if enrollment_id not in existing:
                    csvFile.seek(0, os.SEEK_END)
                    writer = csv.writer(csvFile, delimiter=",")
                    writer.writerow(row)
            res = (
                "Images Saved for ER No:"
                + enrollment_id
                + " Name:"
                + student_name
            )
            _send_feedback(message, res, text_to_speech)
        except Exception as err:
            _send_feedback(
                message, "Failed to capture images, please try again.", text_to_speech
            )
            print(f"Error during image capture: {err}")
