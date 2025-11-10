# 🧠 Attendance Management System using Face Recognition

### 📘 Overview  
An AI-powered web application that automates attendance marking using **real-time facial recognition**.  
Built with **Flask**, **OpenCV**, and **dlib**, it captures faces via webcam, identifies registered users,  
and logs attendance (date & time) into an **SQLite database** with a clean web interface.

---

### ⚙️ Tech Stack  
- **Python 3.8+**  
- **Flask** (Backend Framework)  
- **OpenCV** (Real-time Computer Vision)  
- **dlib & face_recognition** (128-D Facial Encoding)  
- **SQLite3** (Database)  
- **HTML, CSS, JavaScript** (Frontend – Flask Templates)

---

### 🚀 Features  
✅ Real-time face detection & recognition  
✅ Automated attendance logging (date & time)  
✅ SQLite database integration  
✅ CSV export of attendance logs  
✅ Web interface to view, add, and manage users  
✅ Optimized facial matching with vectorized NumPy operations for faster results  
✅ Modular scripts for training, capturing, and managing images  

---

### 📂 Project Structure
AttendanceManagement/
│
├── app.py # Main Flask server file
├── takeImage.py # Capture images for registration
├── trainImage.py # Generate encodings and train face data
├── automaticAttendance.py # Runs recognition & logs attendance
├── show_attendance.py # Displays attendance summary
├── static/ # CSS, JS, and stored images
│ ├── images/ # User face images (ignored in Git)
│ └── models/ # Haar cascade model files
├── templates/ # HTML templates
│ ├── index.html
│ ├── attendance.html
│ ├── register.html
│ ├── summary.html
│ └── base.html
├── Attendance/ # CSV attendance records
│ └── Computer Vision/
├── HaarCascade/ # Haar cascade XML models
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── .gitignore # Ignore venv, DB, image files, etc.

yaml
Copy code

---

### 🧑‍💻 Setup & Installation

1️⃣ **Clone the repository**
```bash
git clone https://github.com/Ujjwalag3784/Attendance-Management-System-Face-Recognition.git
cd Attendance-Management-System-Face-Recognition
2️⃣ Create a virtual environment (recommended)

bash
Copy code
py -3.8 -m venv venv
venv\Scripts\activate
3️⃣ Install dependencies

bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
4️⃣ Run the app

bash
Copy code
python app.py
5️⃣ Open in your browser →
👉 http://127.0.0.1:5000/

⚡ How It Works
Face Detection:
Uses HOG (Histogram of Oriented Gradients) model from dlib to locate faces.

Face Encoding:
Converts each detected face into a 128-dimensional vector (unique facial signature).

Matching & Logging:
Compares real-time encodings with known users using Euclidean distance.
If a match is found → attendance is automatically logged with timestamp in SQLite.

Database Integration:
Attendance is stored and can be viewed/exported via the Flask web UI.

🧩 Optimization
Downscales frames for faster recognition without quality loss

Uses NumPy vectorization for 10× faster embedding comparison

HOG model chosen for real-time CPU inference (GPU not required)

🧾 Example Use Case
This system can be used in:

Educational institutions for student attendance automation

Corporate environments for employee check-in systems

Secure access control via facial authentication

📦 Requirements
Python 3.8 (recommended for dlib compatibility)

Webcam access for face capture

Pretrained model: haarcascade-facerecogmodel.xml (included)

💡 Future Enhancements
Integrate email or SMS alerts for absent users

Deploy Flask app on cloud (Heroku / Render)

Add Admin login and dashboard analytics

Replace HOG with CNN-based face detector for higher accuracy (if GPU available)

👨‍💻 Author
Ujjwal Agrawal
📍 VIT Vellore
💼 GitHub: Ujjwalag3784
📧 Email: (ujjwal.agrawal2022@vitstudent.ac.in)
