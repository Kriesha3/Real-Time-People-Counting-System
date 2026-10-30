
import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


HEADERS = [
    "Timestamp", "Event", "Tracking ID",
    "Entries", "Exits", "People Inside", "Note"
]


class Reports:
    def __init__(self, database):
        self.db = database

    def export_all(self, parent):
        rows = self.db.get_events()
        self._export_rows(
            parent, rows, "people_count_report.csv",
            "All Event History"
        )

    def export_daily(self, parent, selected_date):
        summary = self.db.daily_summary(selected_date)
        rows = self.db.get_events(
            start_date=selected_date,
            end_date=selected_date
        )

        path = filedialog.asksaveasfilename(
            parent=parent,
            title="Export Daily Report",
            defaultextension=".csv",
            initialfile=f"people_count_daily_{selected_date}.csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(["People Counting System - Daily Report"])
                writer.writerow(["Date", selected_date])
                writer.writerow(["Total Entries", summary["entries"]])
                writer.writerow(["Total Exits", summary["exits"]])
                writer.writerow([
                    "Recorded Peak Occupancy", summary["peak"]
                ])
                writer.writerow([])
                writer.writerow(HEADERS)
                writer.writerows(self._row_values(row) for row in rows)

            messagebox.showinfo(
                "Export Complete", f"Daily report saved:\n{path}",
                parent=parent
            )
        except OSError as exc:
            messagebox.showerror(
                "Export Failed", str(exc), parent=parent
            )

    def _export_rows(self, parent, rows, filename, title):
        path = filedialog.asksaveasfilename(
            parent=parent,
            title=title,
            defaultextension=".csv",
            initialfile=filename,
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(HEADERS)
                writer.writerows(self._row_values(row) for row in rows)

            messagebox.showinfo(
                "Export Complete",
                f"Exported {len(rows)} records:\n{path}",
                parent=parent
            )
        except OSError as exc:
            messagebox.showerror(
                "Export Failed", str(exc), parent=parent
            )

    @staticmethod
    def _row_values(row):
        return [
            row["timestamp"], row["event_type"], row["tracking_id"],
            row["entries"], row["exits"], row["people_inside"],
            row["note"]
        ]

    def draw_hourly_chart(self, parent, selected_date):
        for child in parent.winfo_children():
            child.destroy()

        entries, exits = self.db.hourly_summary(selected_date)

        figure = Figure(
            figsize=(5.2, 2.6), dpi=90, facecolor="#0C2039"
        )
        axis = figure.add_subplot(111)
        axis.set_facecolor("#0C2039")

        hours = list(range(24))
        axis.plot(
            hours, entries, marker="o", linewidth=2,
            color="#25D695", label="Entries"
        )
        axis.plot(
            hours, exits, marker="o", linewidth=2,
            color="#FFB020", label="Exits"
        )
        axis.set_xticks(range(0, 24, 3))
        axis.set_xlabel("Hour of day", color="#A9BEDB")
        axis.set_ylabel("People", color="#A9BEDB")
        axis.tick_params(colors="#A9BEDB", labelsize=8)
        axis.grid(True, alpha=0.2)

        for spine in axis.spines.values():
            spine.set_color("#29496C")

        axis.legend(
            facecolor="#102846", edgecolor="#29496C",
            labelcolor="#DCE8FA", fontsize=8
        )
        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
