import cv2
import csv
from pathlib import Path

# Project folder path
PROJECT_DIR = Path(__file__).resolve().parent

# Face detection XML file
CASCADE_PATH = PROJECT_DIR / "haarcascade_frontalface_default.xml"

# Dataset folder
DATASET_DIR = PROJECT_DIR / "dataset"

# Number of face images to capture
TOTAL_IMAGES = 50


def main():
    print("=== Face Recognition Attendance System ===")

    # Get student details
    student_id = input("Enter Student ID: ").strip()
    student_name = input("Enter Student Name: ").strip()

    if not student_id or not student_name:
        print("Error: Student ID and name are required.")
        return

    # Prevent invalid folder names
    if any(char in student_id for char in '<>:"/\\|?*'):
        print("Error: Please enter a valid Student ID.")
        return

    # Load the face detection XML
    face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))

    if face_detector.empty():
        print("ERROR: XML file not loaded!")
        print("Check that haarcascade_frontalface_default.xml")
        print("is present in your project folder.")
        return

    # Create a separate folder for this student
    student_folder = DATASET_DIR / student_id
    student_folder.mkdir(parents=True, exist_ok=True)

    # Save student ID and name in students.csv
    students_file = PROJECT_DIR / "students.csv"
    students = {}

    if students_file.exists():
        with open(students_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2 and row[0] != "Student_ID":
                    students[row[0]] = row[1]

    students[student_id] = student_name

    with open(students_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Student_ID", "Student_Name"])
        for sid, name in students.items():
            writer.writerow([sid, name])

    # Open webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Camera not found!")
        return

    count = 0
    print(f"Capturing images for {student_name}...")
    print("Look at the camera and move your face slightly.")
    print("Press Q to stop.")

    try:
        while count < TOTAL_IMAGES:
            success, frame = camera.read()

            if not success:
                print("ERROR: Cannot read from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                # Draw a green rectangle around the face
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Crop and save the face
                face = gray[y:y + h, x:x + w]

                if face.size > 0:
                    count += 1
                    image_path = student_folder / f"{count}.jpg"
                    cv2.imwrite(str(image_path), face)

                # Save one image per frame
                break

            cv2.putText(
                frame,
                f"Images: {count}/{TOTAL_IMAGES}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Face Capture - Press Q to stop", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\nCapture finished: {count} images saved.")
    print(f"Student: {student_name} (ID: {student_id})")
    print(f"Images folder: {student_folder}")

    if count == TOTAL_IMAGES:
        print("SUCCESS: All 50 face images captured!")
    elif count == 0:
        print("No face images captured. Check lighting and camera.")
    else:
        print("Some images were captured. You can run the script again.")


if __name__ == "__main__":
    main()