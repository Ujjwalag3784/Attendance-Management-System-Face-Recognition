import os
from pathlib import Path
from typing import Callable, Optional

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

import automaticAttendance
import show_attendance
import takeImage
import trainImage


BASE_DIR = Path(__file__).resolve().parent
HAARCASCADE_PATH = str(BASE_DIR / "HaarCascade/haarcascade-facerecogmodel.xml")
TRAIN_IMAGE_PATH = BASE_DIR / "TrainingImage"
TRAIN_IMAGE_PATH.mkdir(parents=True, exist_ok=True)
TRAIN_LABEL_PATH = BASE_DIR / "TrainingImageLabel" / "Trainner.yml"
TRAIN_LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
STUDENT_DETAIL_PATH = BASE_DIR / "StudentDetails" / "studentdetails.csv"
STUDENT_DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
if not STUDENT_DETAIL_PATH.exists():
    STUDENT_DETAIL_PATH.touch()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "class-vision-secret")

ENABLE_TTS = os.getenv("ENABLE_TTS", "0") == "1"
_SPEECH_FN: Optional[Callable[[str], None]] = None


def _speech_callback() -> Optional[Callable[[str], None]]:
    global _SPEECH_FN
    if not ENABLE_TTS:
        return None
    if _SPEECH_FN is not None:
        return _SPEECH_FN
    try:
        import pyttsx3
    except Exception:
        return None

    engine = pyttsx3.init()

    def speak(text: str) -> None:
        engine.say(text)
        engine.runAndWait()

    _SPEECH_FN = speak
    return _SPEECH_FN


class MessageCollector:
    """Mimic the configure(text=...) API used by the legacy Tkinter widgets."""

    def __init__(self) -> None:
        self.text = ""

    def configure(self, **kwargs) -> None:
        text = kwargs.get("text")
        if text is not None:
            self.text = text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        enrollment = request.form.get("enrollment", "").strip()
        full_name = request.form.get("full_name", "").strip()
        collector = MessageCollector()
        takeImage.TakeImage(
            enrollment,
            full_name,
            HAARCASCADE_PATH,
            str(TRAIN_IMAGE_PATH),
            str(STUDENT_DETAIL_PATH),
            collector,
            None,
            _speech_callback(),
            show_window=False,
            target_samples=60,
            timeout=30,
        )
        message = collector.text or "Capture finished."
        flash(message)
        return redirect(url_for("register"))
    return render_template("register.html")


@app.post("/train")
def train_model():
    collector = MessageCollector()
    trainImage.TrainImage(
        HAARCASCADE_PATH,
        str(TRAIN_IMAGE_PATH),
        str(TRAIN_LABEL_PATH),
        collector,
        _speech_callback(),
    )
    flash(collector.text or "Training complete.")
    return redirect(url_for("index"))


@app.route("/attendance", methods=["GET", "POST"])
def attendance_view():
    subject = ""
    duration = 5
    result = None
    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        duration = request.form.get("duration", type=int) or 5
        result = automaticAttendance.capture_attendance(
            subject,
            duration=duration,
            text_to_speech=_speech_callback(),
            show_window=False,
        )
        if result["message"]:
            flash(result["message"])
    return render_template(
        "attendance.html",
        subject=subject,
        duration=duration,
        result=result,
    )


@app.route("/summary", methods=["GET", "POST"])
def summary_view():
    subject = ""
    result = None
    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        result = show_attendance.build_attendance_summary(
            subject,
            text_to_speech=_speech_callback(),
        )
        if result["message"]:
            flash(result["message"])
    return render_template("summary.html", subject=subject, result=result)


if __name__ == "__main__":
    app.run(debug=True)
