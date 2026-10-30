import queue
import threading
import time
import tkinter as tk
from datetime import date, datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

from database import Database
from reports import Reports

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo26n.pt"

CAMERA_INDEX = 0
CONFIDENCE = 0.15
IMAGE_SIZE = 640
LINE_POSITION = 0.50


class PeopleCountingDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time People Counting System")
        self.root.geometry("1250x800")
        self.root.minsize(1000, 650)
        self.root.configure(bg="#071426")

        self.db = Database()
        self.reports = Reports(self.db)

        self.entries = 0
        self.exits = 0
        self.people_inside = 0
        self.crowd_limit = max(
            1, int(self.db.get_setting("crowd_limit", "10"))
        )

        self.model = None
        self.model_error = None
        self.running = False
        self.stop_event = threading.Event()
        self.events = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.photo = None

        self.limit_var = tk.StringVar(value=str(self.crowd_limit))
        self.date_var = tk.StringVar(value=date.today().isoformat())
        self.search_var = tk.StringVar()
        self.filter_var = tk.StringVar(value="ALL")
        self.status_var = tk.StringVar(value="Loading YOLO model...")

        self.build_ui()
        self.refresh_all()
        self.update_clock()

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(40, self.process_events)

        threading.Thread(target=self.load_model, daemon=True).start()

    # ---------------- USER INTERFACE ----------------

    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#091A2E",
            fieldbackground="#091A2E",
            foreground="white",
            rowheight=25
        )
        style.configure(
            "Treeview.Heading",
            background="#173556",
            foreground="white"
        )

        header = tk.Frame(self.root, bg="#071426")
        header.pack(fill="x", padx=15, pady=10)

        tk.Label(
            header,
            text="PEOPLE COUNTING SYSTEM",
            font=("Segoe UI", 20, "bold"),
            fg="white",
            bg="#071426"
        ).pack(side="left")

        self.clock = tk.Label(
            header, fg="#A9BEDB", bg="#071426"
        )
        self.clock.pack(side="right")

        cards = tk.Frame(self.root, bg="#071426")
        cards.pack(fill="x", padx=10)
        self.kpis = {}

        for title, color in [
            ("ENTRIES", "#25D695"),
            ("EXITS", "#FFB020"),
            ("PEOPLE INSIDE", "#4DB7FF"),
            ("CROWD LIMIT", "#B28AFF")
        ]:
            card = tk.Frame(cards, bg="#102846", padx=12, pady=8)
            card.pack(side="left", expand=True, fill="x", padx=5)

            tk.Label(
                card, text=title, bg="#102846", fg="#A9BEDB"
            ).pack(anchor="w")

            value = tk.Label(
                card, text="0",
                font=("Segoe UI", 22, "bold"),
                fg=color, bg="#102846"
            )
            value.pack(anchor="w")

            subtitle = tk.Label(
                card, text="", fg="#A9BEDB", bg="#102846"
            )
            subtitle.pack(anchor="w")
            self.kpis[title] = (value, subtitle)

        main = tk.Frame(self.root, bg="#071426")
        main.pack(fill="both", expand=True, padx=10, pady=10)
        main.columnconfigure(0, weight=6)
        main.columnconfigure(1, weight=5)

        camera_box = self.section(main, "LIVE CAMERA")
        camera_box.grid(row=0, column=0, sticky="nsew", padx=5)

        self.video = tk.Label(
            camera_box, text="Loading YOLO model...",
            bg="#06101E", fg="white",
            font=("Segoe UI", 12)
        )
        self.video.pack(fill="both", expand=True, padx=8, pady=8)

        right = tk.Frame(main, bg="#071426")
        right.grid(row=0, column=1, sticky="nsew", padx=5)

        controls = self.section(right, "CAMERA AND CONTROLS")
        controls.pack(fill="x", pady=5)

        row = tk.Frame(controls, bg="#0C2039")
        row.pack(fill="x", padx=8, pady=5)

        self.start_btn = self.button(
            row, "START CAMERA", self.start_camera, "#07965A"
        )
        self.start_btn.pack(side="left", expand=True, fill="x", padx=3)

        self.stop_btn = self.button(
            row, "STOP CAMERA", self.stop_camera, "#D92D45"
        )
        self.stop_btn.configure(state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=3)

        row = tk.Frame(controls, bg="#0C2039")
        row.pack(fill="x", padx=8, pady=5)

        tk.Label(
            row, text="Crowd limit:", bg="#0C2039", fg="white"
        ).pack(side="left")

        tk.Entry(
            row, textvariable=self.limit_var, width=7
        ).pack(side="left", padx=6)

        self.button(
            row, "SET LIMIT", self.set_limit, "#1769D2"
        ).pack(side="left")

        row = tk.Frame(controls, bg="#0C2039")
        row.pack(fill="x", padx=8, pady=5)

        self.button(
            row, "EXPORT ALL CSV", self.export_all
        ).pack(side="left", expand=True, fill="x", padx=2)

        self.button(
            row, "DAILY REPORT", self.export_daily
        ).pack(side="left", expand=True, fill="x", padx=2)

        self.button(
            row, "RESET COUNTS", self.reset_counts, "#394E6C"
        ).pack(side="left", expand=True, fill="x", padx=2)

        tk.Label(
            controls, textvariable=self.status_var,
            bg="#0C2039", fg="#39D98A",
            wraplength=400, justify="left"
        ).pack(fill="x", padx=10, pady=10)

        lower = tk.Frame(self.root, bg="#071426")
        lower.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        lower.columnconfigure(0, weight=4)
        lower.columnconfigure(1, weight=6)

        chart_box = self.section(lower, "HOURLY TRAFFIC")
        chart_box.grid(row=0, column=0, sticky="nsew", padx=5)

        self.chart_area = tk.Frame(chart_box, bg="#0C2039")
        self.chart_area.pack(fill="both", expand=True, padx=5, pady=5)

        history_box = self.section(lower, "ENTRY / EXIT HISTORY")
        history_box.grid(row=0, column=1, sticky="nsew", padx=5)

        filters = tk.Frame(history_box, bg="#0C2039")
        filters.pack(fill="x", padx=6, pady=5)

        tk.Entry(
            filters, textvariable=self.search_var
        ).pack(side="left", fill="x", expand=True)

        self.search_var.trace_add(
            "write", lambda *_: self.refresh_history()
        )

        combo = ttk.Combobox(
            filters,
            textvariable=self.filter_var,
            values=("ALL", "ENTRY", "EXIT", "RESET"),
            state="readonly", width=8
        )
        combo.pack(side="left", padx=4)
        combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self.refresh_history()
        )

        self.button(
            filters, "REFRESH", self.refresh_history
        ).pack(side="left")

        columns = (
            "timestamp", "event", "id", "entries", "exits", "inside"
        )
        self.tree = ttk.Treeview(
            history_box, columns=columns,
            show="headings", height=6
        )

        for col, heading, width in [
            ("timestamp", "Timestamp", 140),
            ("event", "Event", 65),
            ("id", "Track ID", 60),
            ("entries", "Entries", 55),
            ("exits", "Exits", 50),
            ("inside", "Inside", 55)
        ]:
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor="center")

        self.tree.tag_configure("ENTRY", foreground="#25D695")
        self.tree.tag_configure("EXIT", foreground="#FFB020")
        self.tree.tag_configure("RESET", foreground="#B28AFF")
        self.tree.pack(fill="both", expand=True, padx=6, pady=5)

        footer = tk.Frame(self.root, bg="#0A1C32")
        footer.pack(fill="x", padx=10, pady=5)

        tk.Label(
            footer, text="Date (YYYY-MM-DD):",
            bg="#0A1C32", fg="white"
        ).pack(side="left")

        tk.Entry(
            footer, textvariable=self.date_var, width=12
        ).pack(side="left", padx=5)

        self.button(
            footer, "VIEW DAILY SUMMARY", self.view_daily
        ).pack(side="left")

        self.summary_label = tk.Label(
            footer, text="", bg="#0A1C32", fg="white"
        )
        self.summary_label.pack(side="left", padx=10)

    def section(self, parent, title):
        box = tk.Frame(
            parent, bg="#0C2039",
            highlightbackground="#1E4168", highlightthickness=1
        )
        tk.Label(
            box, text=title, bg="#0C2039", fg="white",
            font=("Segoe UI", 11, "bold")
        ).pack(fill="x", padx=10, pady=8)
        return box

    def button(self, parent, text, command, color="#173B62"):
        return tk.Button(
            parent, text=text, command=command,
            bg=color, fg="white",
            activebackground=color,
            activeforeground="white",
            relief="flat", cursor="hand2",
            padx=8, pady=6
        )

    def update_clock(self):
        try:
            self.clock.configure(
                text=datetime.now().strftime("%d %b %Y  %H:%M:%S")
            )
            self.root.after(1000, self.update_clock)
        except tk.TclError:
            pass

    # ---------------- LOAD YOLO MODEL ----------------

    def load_model(self):
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model file not found:\n{MODEL_PATH}"
                )
            model = YOLO(str(MODEL_PATH))
            self.events.put(("model_ready", model))
        except Exception as exc:
            self.events.put(("model_error", str(exc)))

    # ---------------- CAMERA ----------------

    def start_camera(self):
        if self.running:
            return

        if self.model is None:
            messagebox.showinfo(
                "Please wait",
                self.model_error or "YOLO model is still loading."
            )
            return

        self.stop_event.clear()
        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        threading.Thread(
            target=self.camera_worker, daemon=True
        ).start()

    def stop_camera(self):
        self.stop_event.set()
        self.status_var.set("Stopping camera...")

    def camera_worker(self):
        camera = None
        previous_y = {}
        last_counted = {}

        try:
            camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(CAMERA_INDEX)

            if not camera.isOpened():
                self.events.put(
                    ("camera_error", "Could not open camera.")
                )
                return

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.events.put(("camera_status", "Camera running"))

            while not self.stop_event.is_set():
                ok, frame = camera.read()

                if not ok or frame is None:
                    self.events.put(
                        ("camera_error", "Could not read camera frame.")
                    )
                    break

                height, width = frame.shape[:2]
                line_y = int(height * LINE_POSITION)

                # Predict detects people; track supplies tracking IDs.
                result = self.model.track(
                    source=frame,
                    persist=True,
                    classes=[0],
                    conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    tracker="bytetrack.yaml",
                    verbose=False
                )[0]

                # Show the counting line.
                cv2.line(
                    frame, (0, line_y), (width, line_y),
                    (0, 220, 255), 2
                )
                cv2.putText(
                    frame, "COUNTING LINE",
                    (10, max(22, line_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 255), 2
                )

                detected_count = 0

                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    detected_count = len(boxes)

                    ids = None
                    if result.boxes.id is not None:
                        ids = result.boxes.id.int().cpu().tolist()

                    # Draw all detections, even when tracking IDs are absent.
                    for index, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        confidence = float(
                            result.boxes.conf[index].item()
                        )

                        label = f"Person {confidence:.2f}"
                        color = (30, 230, 100)

                        if ids is not None:
                            track_id = ids[index]
                            label = f"ID {track_id} {confidence:.2f}"

                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            old_y = previous_y.get(track_id)
                            now = time.monotonic()

                            can_count = (
                                now - last_counted.get(track_id, 0) >= 1.0
                            )

                            if old_y is not None and can_count:
                                if old_y < line_y <= cy:
                                    self.events.put(
                                        ("crossing", "ENTRY", track_id)
                                    )
                                    last_counted[track_id] = now
                                elif old_y >= line_y > cy:
                                    self.events.put(
                                        ("crossing", "EXIT", track_id)
                                    )
                                    last_counted[track_id] = now

                            previous_y[track_id] = cy

                        cv2.rectangle(
                            frame, (x1, y1), (x2, y2), color, 2
                        )
                        cv2.putText(
                            frame, label,
                            (x1, max(20, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 2
                        )

                cv2.putText(
                    frame, f"People detected: {detected_count}",
                    (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2
                )

                try:
                    if self.frames.full():
                        self.frames.get_nowait()
                    self.frames.put_nowait(frame)
                except queue.Full:
                    pass

        except Exception as exc:
            self.events.put(("camera_error", str(exc)))

        finally:
            if camera is not None:
                camera.release()
            self.events.put(("camera_stopped", None))

    # ---------------- PROCESS CAMERA EVENTS ----------------

    def process_events(self):
        try:
            while True:
                kind, *data = self.events.get_nowait()

                if kind == "model_ready":
                    self.model = data[0]
                    self.status_var.set(
                        "Model ready. Press START CAMERA."
                    )
                    self.video.configure(
                        text="Model ready\nPress START CAMERA"
                    )

                elif kind == "model_error":
                    self.model_error = data[0]
                    self.status_var.set("Model loading failed")
                    self.video.configure(text=self.model_error)

                elif kind == "camera_status":
                    self.status_var.set(data[0])

                elif kind == "crossing":
                    direction, track_id = data

                    if direction == "ENTRY":
                        self.entries += 1
                        self.people_inside += 1
                    else:
                        self.exits += 1
                        self.people_inside = max(
                            0, self.people_inside - 1
                        )

                    self.db.add_event(
                        direction, track_id,
                        self.entries, self.exits,
                        self.people_inside
                    )

                    self.update_kpis()
                    self.refresh_history()
                    self.refresh_chart()
                    self.update_summary()

                    if self.people_inside >= self.crowd_limit:
                        self.status_var.set(
                            "WARNING: CROWD LIMIT REACHED"
                        )
                    else:
                        self.status_var.set("Camera running")

                elif kind == "camera_error":
                    self.status_var.set("Camera error")
                    self.stop_event.set()
                    messagebox.showerror("Camera Error", data[0])

                elif kind == "camera_stopped":
                    self.running = False
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.status_var.set("Camera stopped")

        except queue.Empty:
            pass

        try:
            frame = self.frames.get_nowait()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((
                max(320, self.video.winfo_width() - 10),
                max(220, self.video.winfo_height() - 10)
            ))
            self.photo = ImageTk.PhotoImage(image)
            self.video.configure(image=self.photo, text="")
        except queue.Empty:
            pass

        try:
            self.root.after(40, self.process_events)
        except tk.TclError:
            pass

    # ---------------- COUNTERS ----------------

    def update_kpis(self):
        summary = self.db.daily_summary(date.today().isoformat())

        values = {
            "ENTRIES": (self.entries, f'{summary["entries"]} today'),
            "EXITS": (self.exits, f'{summary["exits"]} today'),
            "PEOPLE INSIDE": (
                self.people_inside, "Current occupancy"
            ),
            "CROWD LIMIT": (self.crowd_limit, "Configured limit")
        }

        for title, (value, subtitle) in values.items():
            self.kpis[title][0].configure(text=str(value))
            self.kpis[title][1].configure(text=subtitle)

    # ---------------- HISTORY AND REPORTS ----------------

    def refresh_history(self):
        if not hasattr(self, "tree"):
            return

        rows = self.db.get_events(
            event_type=self.filter_var.get(),
            search=self.search_var.get().strip(),
            limit=1000
        )

        self.tree.delete(*self.tree.get_children())

        for row in rows:
            event = row["event_type"]
            self.tree.insert(
                "", "end",
                values=(
                    row["timestamp"], event, row["tracking_id"],
                    row["entries"], row["exits"], row["people_inside"]
                ),
                tags=(event,)
            )

    def selected_date(self):
        try:
            return datetime.strptime(
                self.date_var.get().strip(), "%Y-%m-%d"
            ).date().isoformat()
        except ValueError:
            messagebox.showwarning(
                "Invalid date", "Enter date as YYYY-MM-DD."
            )
            return None

    def update_summary(self):
        try:
            selected = datetime.strptime(
                self.date_var.get().strip(), "%Y-%m-%d"
            ).date().isoformat()
        except ValueError:
            return

        summary = self.db.daily_summary(selected)
        self.summary_label.configure(
            text=f'{selected} | In: {summary["entries"]} | '
                 f'Out: {summary["exits"]} | Peak: {summary["peak"]}'
        )

    def refresh_chart(self):
        try:
            selected = datetime.strptime(
                self.date_var.get().strip(), "%Y-%m-%d"
            ).date().isoformat()
        except ValueError:
            selected = date.today().isoformat()

        self.reports.draw_hourly_chart(self.chart_area, selected)

    def view_daily(self):
        selected = self.selected_date()
        if selected:
            self.reports.draw_hourly_chart(self.chart_area, selected)
            self.update_summary()

    def export_daily(self):
        selected = self.selected_date()
        if selected:
            self.reports.export_daily(self.root, selected)

    def export_all(self):
        self.reports.export_all(self.root)

    # ---------------- CROWD LIMIT ----------------

    def set_limit(self):
        try:
            limit = int(self.limit_var.get())
            if limit < 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning(
                "Invalid limit",
                "Enter a whole number greater than zero."
            )
            return

        self.crowd_limit = limit
        self.db.set_setting("crowd_limit", limit)
        self.update_kpis()

        if self.people_inside >= limit:
            self.status_var.set("WARNING: CROWD LIMIT REACHED")
        else:
            self.status_var.set(f"Crowd limit set to {limit}")

    # ---------------- RESET ----------------

    def reset_counts(self):
        if not messagebox.askyesno(
            "Reset counters",
            "Reset current counters to zero?\n\n"
            "Saved history will not be deleted."
        ):
            return

        self.entries = 0
        self.exits = 0
        self.people_inside = 0

        self.db.add_event(
            "RESET", "-", 0, 0, 0,
            "Counters reset; historical records retained"
        )

        self.update_kpis()
        self.refresh_history()
        self.refresh_chart()
        self.update_summary()
        self.status_var.set("Counters reset; history retained")

    def refresh_all(self):
        self.update_kpis()
        self.refresh_history()
        self.refresh_chart()
        self.update_summary()

    def close(self):
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PeopleCountingDashboard(root)
    root.mainloop()