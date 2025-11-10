import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
attendance_path = BASE_DIR / "Attendance"
attendance_path.mkdir(parents=True, exist_ok=True)


def _announce(text_to_speech, message: str):
    """Announce a message using text-to-speech if available"""
    if text_to_speech:
        try:
            text_to_speech(message)
        except Exception:
            pass


def build_attendance_summary(subject_name, text_to_speech=None):
    subject_name = subject_name.strip()
    result = {
        "success": False,
        "message": "",
        "summary_path": None,
        "records": [],
        "columns": [],
    }

    def _set_message(text):
        result["message"] = text
        _announce(text_to_speech, text)

    if subject_name == "":
        _set_message("Please enter the subject name.")
        return result

    subject_dir = attendance_path / subject_name
    if not subject_dir.exists():
        _set_message("No attendance records found for that subject yet.")
        return result

    csv_files = sorted(subject_dir.glob(f"{subject_name}_*.csv"))
    if not csv_files:
        _set_message("No attendance CSV files were generated for this subject.")
        return result

    frames = [pd.read_csv(file) for file in csv_files]
    newdf = frames[0]
    for frame in frames[1:]:
        newdf = pd.merge(newdf, frame, on=["Enrollment", "Name"], how="outer")
    newdf.fillna(0, inplace=True)
    date_columns = [col for col in newdf.columns if col not in ("Enrollment", "Name")]
    if date_columns:
        numeric = newdf[date_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        percentages = (numeric.mean(axis=1) * 100).round().astype(int)
        newdf["Attendance"] = percentages.astype(str) + "%"
    else:
        newdf["Attendance"] = "0%"

    summary_path = subject_dir / "attendance.csv"
    newdf.to_csv(summary_path, index=False)

    _set_message(f"Attendance summary ready for {subject_name}.")
    result["success"] = True
    result["summary_path"] = str(summary_path)
    result["records"] = newdf.to_dict(orient="records")
    result["columns"] = newdf.columns.tolist()
    return result