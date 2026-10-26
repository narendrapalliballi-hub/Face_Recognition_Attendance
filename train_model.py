import cv2
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "trainer"
MODEL_PATH = MODEL_DIR / "face_model.yml"


def main():
    # Check LBPH support
    if not hasattr(cv2, "face"):
        print("ERROR: OpenCV face module is missing.")
        print("Run: python -m pip install opencv-contrib-python")
        return

    if not DATASET_DIR.exists():
        print("ERROR: Dataset folder not found.")
        return

    faces = []
    labels = []
    student_ids = []

    # Read each student's images
    for student_folder in DATASET_DIR.iterdir():
        if not student_folder.is_dir():
            continue

        try:
            student_id = int(student_folder.name)
        except ValueError:
            print(f"Skipping invalid student ID folder: {student_folder.name}")
            continue

        for image_path in student_folder.glob("*.jpg"):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None or image.size == 0:
                print(f"Skipping unreadable image: {image_path.name}")
                continue

            # Resize images to a consistent size
            image = cv2.resize(image, (200, 200))
            faces.append(image)
            labels.append(student_id)
            student_ids.append(student_id)

    if not faces:
        print("ERROR: No face images found inside dataset/<student_id>/.")
        return

    # Train LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels, dtype=np.int32))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(MODEL_PATH))

    print("\nTRAINING SUCCESSFUL!")
    print(f"Images used: {len(faces)}")
    print(f"Student IDs trained: {sorted(set(student_ids))}")
    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    main()