
import cv2
import csv
import os
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# REAL-TIME PEOPLE COUNTING SYSTEM
# ==========================================

MODEL_PATH = "yolo26n.pt"
REPORT_PATH = "people_count_report.csv"
MAX_PEOPLE = 10

# Load YOLO model
print("Loading YOLO AI model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit(1)

# Counting statistics
entry_count = 0
exit_count = 0
people_inside = 0

# Tracking data
previous_positions = {}
last_counted_direction = {}

print("People Counting System started.")
print("Press Q in the webcam window to stop.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to read webcam frame.")
            break

        height, width = frame.shape[:2]
        line_y = height // 2

        # Detect and track people
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False,
            conf=0.4
        )

        # Draw virtual counting line
        cv2.line(
            frame,
            (0, line_y),
            (width, line_y),
            (255, 0, 0),
            2
        )

        for result in results:
            if result.boxes is None:
                continue

            if result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, person_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)

                # Center of the person's bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Draw detection box and tracking ID
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    f"ID: {person_id}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Check whether this person crossed the line
                if person_id in previous_positions:
                    old_y = previous_positions[person_id]

                    if old_y < line_y <= center_y:
                        # Top to bottom = entry
                        if last_counted_direction.get(person_id) != "entry":
                            entry_count += 1
                            people_inside += 1
                            last_counted_direction[person_id] = "entry"

                    elif old_y >= line_y > center_y:
                        # Bottom to top = exit
                        if last_counted_direction.get(person_id) != "exit":
                            exit_count += 1
                            people_inside = max(0, people_inside - 1)
                            last_counted_direction[person_id] = "exit"

                previous_positions[person_id] = center_y

        # Dashboard background
        cv2.rectangle(
            frame,
            (10, 10),
            (330, 155),
            (30, 30, 30),
            -1
        )

        # Display statistics
        cv2.putText(
            frame,
            f"Entries: {entry_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Exits: {exit_count}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"People Inside: {people_inside}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Maximum: {MAX_PEOPLE}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        # Crowd alert
        if people_inside >= MAX_PEOPLE:
            cv2.putText(
                frame,
                "CROWD LIMIT REACHED!",
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Window title
        cv2.imshow("Real-Time People Counting System", frame)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

    # Save session summary to CSV
    file_exists = os.path.exists(REPORT_PATH)
    file_is_empty = (
        not file_exists or os.path.getsize(REPORT_PATH) == 0
    )

    try:
        with open(REPORT_PATH, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if file_is_empty:
                writer.writerow([
                    "Date and Time",
                    "Total Entries",
                    "Total Exits",
                    "People Inside"
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                entry_count,
                exit_count,
                people_inside
            ])

        print("\nSession report saved successfully!")
        print(f"Report location: {os.path.abspath(REPORT_PATH)}")
        print(f"Total entries: {entry_count}")
        print(f"Total exits: {exit_count}")
        print(f"Estimated people inside: {people_inside}")

    except OSError as error:
        print(f"Could not save the CSV report: {error}")

    print("People Counting System stopped.")
