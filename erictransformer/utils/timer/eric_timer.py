import csv
import os
import time
from contextlib import contextmanager


class EricTimer:
    def __init__(self, out_dir: str, enabled: bool = True):
        self.enabled = enabled  # if false self.report() does nothing.
        self._stats = {}
        self._active = {}
        self._out_dir = out_dir
        self._category_dir = os.path.join(out_dir, "categories")
        self._ensure_dirs()
        self.start_time = time.perf_counter()

    @contextmanager
    def section(self, category: str, label: str):
        self.start(category, label)
        try:
            yield
        finally:
            self.stop(category, label)

    def start(self, category: str, label: str):
        if self._active:
            raise ValueError(f"Eric Timer is already running! with {self._active}")
        if label == "misc_time":
            raise ValueError("misc_time is an invalid label")

        if category not in self._active:
            self._active[category] = {}

        if label in self._active[category]:
            raise ValueError(f"{category!r}/{label!r} is active. Call stop() first.")

        self._active[category][label] = time.perf_counter()

    def stop(self, category: str, label: str):
        if category not in self._active or label not in self._active[category]:
            raise RuntimeError(
                f"stop() called without calling start for {category!r}/{label!r} first."
            )

        t0 = self._active[category][label]
        del self._active[category][label]
        if not self._active[category]:
            del self._active[category]

        dt = time.perf_counter() - t0

        if category not in self._stats:
            self._stats[category] = {}

        if label not in self._stats[category]:
            self._stats[category][label] = {"total": 0.0, "count": 0}

        self._stats[category][label]["total"] += dt
        self._stats[category][label]["count"] += 1

        return dt

    def _ensure_dirs(self):
        if self.enabled:
            if self._out_dir:
                os.makedirs(self._out_dir, exist_ok=True)
            if self._category_dir:
                os.makedirs(self._category_dir, exist_ok=True)

    def _category_path(self, category: str):
        fname = f"{category}.csv"
        return os.path.join(self._category_dir, fname)

    def _write_category_json(self, category: str):
        if category in self._stats:
            labels = self._stats[category]
        else:
            labels = {}

        rows = []
        for label, s in labels.items():
            total = float(s["total"])
            count = int(s["count"])
            mean = (total / count) if count else 0.0
            rows.append(
                {
                    "label": label,
                    "total": round(total, 4),
                    "mean": round(mean, 4),
                    "count": count,
                }
            )
        rows.sort(key=lambda r: r["total"], reverse=True)

        with open(
            self._category_path(category), "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["label", "total", "mean", "count"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def _write_categories_summary_json(self):

        if self.enabled:
            path = os.path.join(self._out_dir, "summary.csv")
            rows = []
            for category, labels in self._stats.items():
                total = 0.0
                for s in labels.values():
                    total += float(s["total"])
                rows.append({"category": category, "total": round(total, 4)})
            rows.sort(key=lambda r: r["total"], reverse=True)

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["category", "total"])
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)

    def report(self):
        if self.enabled:
            accounted_time = 0.0
            for labels in self._stats.values():
                for s in labels.values():
                    accounted_time += float(s["total"])

            misc_time = time.perf_counter() - self.start_time - accounted_time
            if misc_time < 0:
                misc_time = 0.0

            misc_cat = "misc"
            if misc_cat not in self._stats:
                self._stats[misc_cat] = {}
            self._stats[misc_cat]["misc_time"] = {"total": misc_time, "count": 1}

            for category in list(self._stats.keys()):
                self._write_category_json(category)

            self._write_categories_summary_json()

    def reset(self):
        self._stats.clear()
        self._active.clear()
        self.start_time = time.perf_counter()
