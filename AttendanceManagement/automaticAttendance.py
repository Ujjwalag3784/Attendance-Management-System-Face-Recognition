import os, cv2
import numpy as np
import pandas as pd
import datetime
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
haarcasecade_path = str(BASE_DIR / "HaarCascade/haarcascade-facerecogmodel.xml")
trainimagelabel_path = BASE_DIR / "TrainingImageLabel" / "Trainner.yml"
trainimagelabel_path.parent.mkdir(parents=True, exist_ok=True)
trainimage_path = BASE_DIR / "TrainingImage"
trainimage_path.mkdir(parents=True, exist_ok=True)
studentdetail_path = BASE_DIR / "StudentDetails" / "studentdetails.csv"
studentdetail_path.parent.mkdir(parents=True, exist_ok=True)
attendance_path = BASE_DIR / "Attendance"
attendance_path.mkdir(parents=True, exist_ok=True)





def _announce(text_to_speech, message: str):
    if text_to_speech:
        try:
            text_to_speech(message)
        except Exception:
            pass


def capture_attendance(subject_name, duration=20, text_to_speech=None, show_window=True):
    """
    Run the attendance capture loop and return structured metadata that UIs can consume.
    """
    subject_name = subject_name.strip()
    result = {
        "success": False,
        "message": "",
        "file_path": None,
        "records": [],
        "columns": [],
    }

    def _set_message(text):
        result["message"] = text
        _announce(text_to_speech, text)

    if not subject_name:
        _set_message("Please enter the subject name!!!")
        return result

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            recognizer.read(str(trainimagelabel_path))
        except Exception:
            _set_message("Model not found,please train model")
            return result

        facecasCade = cv2.CascadeClassifier(haarcasecade_path)
        if facecasCade.empty():
            _set_message("Cannot load haarcascade file.")
            return result

        if not studentdetail_path.exists():
            _set_message("Student details missing, please register students first.")
            return result

        df = pd.read_csv(studentdetail_path)
        if df.empty:
            _set_message("Student details missing, please register students first.")
            return result
        df["Enrollment"] = df["Enrollment"].astype(str)
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            _set_message("Cannot access camera.")
            return result

        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ["Enrollment", "Name"]
        attendance = pd.DataFrame(columns=col_names)
        end_time = time.time() + duration

        try:
            while True:
                ret, im = cam.read()
                if not ret:
                    break
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = facecasCade.detectMultiScale(gray, 1.2, 5)
                for (x, y, w, h) in faces:
                    Id, conf = recognizer.predict(gray[y : y + h, x : x + w])
                    if conf < 70:
                        recognized_id = str(Id)
                        matches = df.loc[df["Enrollment"] == recognized_id, "Name"].values
                        name_value = str(matches[0]) if matches.size else "Unknown"
                        label_text = f"{recognized_id}-{name_value}"
                        attendance.loc[len(attendance)] = [recognized_id, name_value]
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 4)
                        cv2.putText(
                            im, label_text, (x + h, y), font, 1, (255, 255, 0), 4
                        )
                    else:
                        label_text = "Unknown"
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 7)
                        cv2.putText(im, label_text, (x + h, y), font, 1, (0, 25, 255), 4)

                if show_window:
                    cv2.imshow("Filling Attendance...", im)
                    key = cv2.waitKey(30) & 0xFF
                    if key == 27:
                        break
                else:
                    time.sleep(0.01)

                if time.time() > end_time:
                    break
        finally:
            cam.release()
            if show_window:
                cv2.destroyAllWindows()

        if attendance.empty:
            _set_message("No known faces found for attendance.")
            return result

        attendance = attendance.drop_duplicates(["Enrollment"], keep="first")
        timestamp = datetime.datetime.now()
        date = timestamp.strftime("%Y-%m-%d")
        timeStamp = timestamp.strftime("%H:%M:%S")
        attendance[date] = 1
        Hour, Minute, Second = timeStamp.split(":")
        subject_dir = attendance_path / subject_name
        subject_dir.mkdir(parents=True, exist_ok=True)
        file_path = subject_dir / f"{subject_name}_{date}_{Hour}-{Minute}-{Second}.csv"
        attendance.to_csv(file_path, index=False)

        message = f"Attendance Filled Successfully for {subject_name}"
        _set_message(message)
        result["success"] = True
        result["file_path"] = str(file_path)
        result["records"] = attendance.to_dict(orient="records")
        result["columns"] = attendance.columns.tolist()
        return result
    except Exception:
        if "cam" in locals():
            try:
                cam.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        _set_message("No Face found for attendance")
        return result






