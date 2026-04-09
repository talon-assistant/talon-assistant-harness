"""Job Inbox — browse tracker applications, manage status, run pipelines.

Provides a non-modal window for reviewing every job currently in the
job_tracker database: filter by status/source/score, sort by any column,
manage status transitions via a dropdown, kick off the tailored-resume
and cover-letter pipelines per row, and drop in a raw URL for a job the
scraper missed or that came from outside the configured search URLs.

Pipelines run on a background QThread so the UI never blocks; results
come back via Qt signals and refresh the row. Fit analysis is kicked off
the same way _run_fit_analysis does it internally (daemon thread in the
talent), so the inbox stays responsive.
"""
from __future__ import annotations

import logging
import os
import webbrowser
from datetime import date
from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)


# Workers that were still running when the dialog was closed get moved
# here so Python doesn't garbage-collect them mid-scrape. They finish on
# their own and self-drop via _detached_worker_done.
_DETACHED_WORKERS: list = []


def _detached_worker_done(worker) -> None:
    """Drop a detached worker from the keepalive list when it finishes."""
    try:
        _DETACHED_WORKERS.remove(worker)
    except ValueError:
        pass
    try:
        worker.deleteLater()
    except Exception:
        pass


_STATUS_CHOICES = [
    "new", "applied", "interview", "offer", "rejected", "withdrawn", "archived",
]

_COLUMNS = [
    ("ID", 50),
    ("Fit", 55),
    ("Company", 180),
    ("Position", 280),
    ("Location", 140),
    ("Source", 90),
    ("Found", 90),
    ("Status", 120),
    ("Actions", 360),
]


# ── Background pipeline worker ──────────────────────────────────────────────

class _PipelineWorker(QThread):
    """Runs a job_search talent action on a background thread.

    Supported actions:
        "prepare_materials"   -> tailored resume DOCX/PDF
        "cover_letter"        -> cover letter txt/DOCX/PDF
        "prepare_everything"  -> both
        "add_from_url"        -> scrape a raw URL and add to tracker
        "run_search"          -> scrape every configured search URL + score
    """

    done = pyqtSignal(str, dict)   # (action, result_dict)
    failed = pyqtSignal(str, str)  # (action, error_message)

    def __init__(self, action: str, job_search, *, app_id: int | None = None,
                 url: str | None = None) -> None:
        super().__init__()
        self._action = action
        self._job_search = job_search
        self._app_id = app_id
        self._url = url

    def run(self) -> None:  # noqa: D401 — QThread entry point
        try:
            if self._action == "add_from_url":
                result = self._job_search.scrape_and_add_from_url(self._url or "")
                self.done.emit(self._action, result)
                return

            if self._action == "run_search":
                result = self._job_search._handle_search(context={})
                self.done.emit(self._action, result or {})
                return

            # All talent-level commands expect a text command string
            cmd_map = {
                "prepare_materials": f"prepare materials for #{self._app_id}",
                "cover_letter": f"write a cover letter for #{self._app_id}",
                "prepare_everything": f"prepare everything for #{self._app_id}",
            }
            command = cmd_map.get(self._action)
            if not command:
                self.failed.emit(self._action, f"Unknown action: {self._action}")
                return

            result = self._job_search.execute(command, context={})
            self.done.emit(self._action, result or {})
        except Exception as e:
            log.exception(f"[JobInbox] Pipeline worker failed: {e}")
            self.failed.emit(self._action, str(e))


# ── Main dialog ─────────────────────────────────────────────────────────────

class JobInboxDialog(QDialog):
    """Browse and manage tracker jobs; kick off resume / cover-letter pipelines."""

    def __init__(self, assistant, bridge, parent=None) -> None:
        super().__init__(parent)
        self._assistant = assistant
        self._bridge = bridge
        self._workers: list[_PipelineWorker] = []
        self._rows_cache: list[dict] = []

        self.setWindowTitle("Job Inbox")
        self.setObjectName("job_inbox_dialog")
        self.setMinimumSize(1200, 680)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._setup_ui()
        self.refresh()

    # ── UI build ─────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Drop-in URL bar ─────────────────────────────────────────
        drop_box = QHBoxLayout()
        drop_label = QLabel("Drop in job URL:")
        drop_label.setFont(QFont("", 10, QFont.Weight.Bold))
        self._url_input = QLineEdit()
        self._url_input.setPlaceholderText(
            "Paste a job posting URL (LinkedIn / Dice / Built In / ATS / etc.) and hit Add"
        )
        self._url_input.returnPressed.connect(self._on_add_url)
        self._add_url_btn = QPushButton("Add + Score")
        self._add_url_btn.clicked.connect(self._on_add_url)

        self._run_search_btn = QPushButton("Run Search + Score")
        self._run_search_btn.setToolTip(
            "Scrape every configured job search URL, add new listings, "
            "and run fit scoring in the background."
        )
        self._run_search_btn.clicked.connect(self._on_run_search)

        drop_box.addWidget(drop_label)
        drop_box.addWidget(self._url_input, stretch=1)
        drop_box.addWidget(self._add_url_btn)
        drop_box.addWidget(self._run_search_btn)
        layout.addLayout(drop_box)

        # ── Filter / search bar ──────────────────────────────────────
        filter_box = QHBoxLayout()

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search company or title...")
        self._search_input.textChanged.connect(self._apply_filters)

        self._status_filter = QComboBox()
        self._status_filter.addItem("All statuses", "")
        for s in _STATUS_CHOICES:
            self._status_filter.addItem(s.title(), s)
        self._status_filter.currentIndexChanged.connect(self._apply_filters)

        self._source_filter = QComboBox()
        self._source_filter.addItem("All sources", "")
        for src in ("LinkedIn", "Dice", "Built In", "Manual"):
            self._source_filter.addItem(src, src)
        self._source_filter.currentIndexChanged.connect(self._apply_filters)

        self._min_fit = QComboBox()
        self._min_fit.addItem("Any fit", 0)
        for score in (50, 60, 70, 80, 90):
            self._min_fit.addItem(f"{score}+", score)
        self._min_fit.currentIndexChanged.connect(self._apply_filters)

        self._show_archived = QComboBox()
        self._show_archived.addItem("Active only", False)
        self._show_archived.addItem("Include archived", True)
        self._show_archived.currentIndexChanged.connect(self.refresh)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)

        filter_box.addWidget(self._search_input, stretch=2)
        filter_box.addWidget(self._status_filter)
        filter_box.addWidget(self._source_filter)
        filter_box.addWidget(self._min_fit)
        filter_box.addWidget(self._show_archived)
        filter_box.addWidget(refresh_btn)
        layout.addLayout(filter_box)

        # ── Table ────────────────────────────────────────────────────
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in _COLUMNS])
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.itemSelectionChanged.connect(self._on_row_selected)

        header = self._table.horizontalHeader()
        for i, (_, width) in enumerate(_COLUMNS):
            self._table.setColumnWidth(i, width)
        header.setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch  # Position column
        )
        layout.addWidget(self._table, stretch=1)

        # ── Detail pane ──────────────────────────────────────────────
        detail_label = QLabel("Fit analysis / notes:")
        detail_label.setFont(QFont("", 9, QFont.Weight.Bold))
        layout.addWidget(detail_label)

        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setMaximumHeight(140)
        self._detail.setPlaceholderText(
            "Select a row to see its fit analysis, recommendation, and JD excerpt."
        )
        layout.addWidget(self._detail)

        # ── Status bar ───────────────────────────────────────────────
        status_box = QHBoxLayout()
        self._status_label = QLabel("Ready.")
        self._busy_bar = QProgressBar()
        self._busy_bar.setRange(0, 0)  # indeterminate
        self._busy_bar.setVisible(False)
        self._busy_bar.setMaximumWidth(220)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        status_box.addWidget(self._status_label, stretch=1)
        status_box.addWidget(self._busy_bar)
        status_box.addWidget(close_btn)
        layout.addLayout(status_box)

    # ── Data loading / filtering ─────────────────────────────────────────

    def _db(self):
        from talents.job_tracker import _DB, _data_dir
        return _DB(os.path.join(_data_dir(), "job_tracker.db"))

    def _job_search_talent(self):
        """Return the live JobSearchTalent instance from the assistant."""
        talents = getattr(self._assistant, "talents", None) or []
        for t in talents:
            if getattr(t, "name", "") == "job_search":
                return t
        # Fallback: construct a throwaway instance (scraping still works,
        # config won't be loaded but per-row pipelines don't need it).
        from talents.job_search import JobSearchTalent
        return JobSearchTalent()

    def refresh(self) -> None:
        include_archived = bool(self._show_archived.currentData())
        try:
            rows = self._db().list_all(include_archived=include_archived)
        except Exception as e:
            log.error(f"[JobInbox] DB read failed: {e}")
            QMessageBox.critical(
                self, "Job Inbox",
                f"Could not read job tracker database:\n\n{e}"
            )
            return
        self._rows_cache = rows
        self._apply_filters()

    def _apply_filters(self) -> None:
        search = (self._search_input.text() or "").lower().strip()
        status = self._status_filter.currentData() or ""
        source = self._source_filter.currentData() or ""
        min_fit = int(self._min_fit.currentData() or 0)

        filtered = []
        for row in self._rows_cache:
            if status:
                row_status = (row.get("status") or "").lower()
                if status == "archived":
                    if not row.get("archived"):
                        continue
                elif row_status != status:
                    continue
            if source and (row.get("source") or "") != source:
                continue
            if min_fit and int(row.get("fit_score") or 0) < min_fit:
                continue
            if search:
                hay = " ".join([
                    str(row.get("company") or ""),
                    str(row.get("position") or ""),
                    str(row.get("location") or ""),
                ]).lower()
                if search not in hay:
                    continue
            filtered.append(row)

        self._populate_table(filtered)
        self._status_label.setText(
            f"{len(filtered)} of {len(self._rows_cache)} jobs"
        )

    def _populate_table(self, rows: list[dict]) -> None:
        # Sorting off during repopulate
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)

        for row in rows:
            r = self._table.rowCount()
            self._table.insertRow(r)

            # ID
            id_item = QTableWidgetItem()
            id_item.setData(Qt.ItemDataRole.DisplayRole, int(row["id"]))
            id_item.setData(Qt.ItemDataRole.UserRole, int(row["id"]))
            self._table.setItem(r, 0, id_item)

            # Fit
            fit = int(row.get("fit_score") or 0)
            fit_item = QTableWidgetItem()
            fit_item.setData(Qt.ItemDataRole.DisplayRole, fit)
            if fit >= 80:
                fit_item.setForeground(QColor("#2e7d32"))  # green
            elif fit >= 60:
                fit_item.setForeground(QColor("#ef6c00"))  # orange
            elif fit > 0:
                fit_item.setForeground(QColor("#c62828"))  # red
            else:
                fit_item.setForeground(QColor("#888"))
            self._table.setItem(r, 1, fit_item)

            # Company / Position / Location / Source / Date
            self._table.setItem(r, 2, QTableWidgetItem(row.get("company") or ""))
            self._table.setItem(r, 3, QTableWidgetItem(row.get("position") or ""))
            self._table.setItem(r, 4, QTableWidgetItem(row.get("location") or ""))
            self._table.setItem(r, 5, QTableWidgetItem(row.get("source") or ""))
            self._table.setItem(r, 6, QTableWidgetItem(row.get("date_found") or ""))

            # Status dropdown
            status_combo = QComboBox()
            for s in _STATUS_CHOICES:
                status_combo.addItem(s.title(), s)
            current = (row.get("status") or "new").lower()
            if row.get("archived"):
                current = "archived"
            idx = max(0, status_combo.findData(current))
            status_combo.setCurrentIndex(idx)
            status_combo.currentIndexChanged.connect(
                lambda _i, app_id=int(row["id"]), combo=status_combo:
                    self._on_status_changed(app_id, combo.currentData())
            )
            self._table.setCellWidget(r, 7, status_combo)

            # Actions cell
            actions = QWidget()
            ab = QHBoxLayout(actions)
            ab.setContentsMargins(2, 2, 2, 2)
            ab.setSpacing(3)

            btn_all = QPushButton("All")
            btn_all.setToolTip("Prepare resume + cover letter")
            btn_all.clicked.connect(
                lambda _c, app_id=int(row["id"]):
                    self._on_run_pipeline("prepare_everything", app_id)
            )
            btn_res = QPushButton("Resume")
            btn_res.setToolTip("Tailored resume only")
            btn_res.clicked.connect(
                lambda _c, app_id=int(row["id"]):
                    self._on_run_pipeline("prepare_materials", app_id)
            )
            btn_cl = QPushButton("Letter")
            btn_cl.setToolTip("Cover letter only")
            btn_cl.clicked.connect(
                lambda _c, app_id=int(row["id"]):
                    self._on_run_pipeline("cover_letter", app_id)
            )
            btn_open = QPushButton("Open")
            btn_open.setToolTip("Open the job URL in your default browser")
            btn_open.clicked.connect(
                lambda _c, url=row.get("job_url", ""):
                    self._open_url(url)
            )
            btn_del = QPushButton("Delete")
            btn_del.setToolTip("Permanently delete this row")
            btn_del.clicked.connect(
                lambda _c, app_id=int(row["id"]):
                    self._on_delete(app_id)
            )

            for b in (btn_all, btn_res, btn_cl, btn_open, btn_del):
                b.setFixedHeight(28)
                b.setMinimumWidth(62)
                ab.addWidget(b)

            self._table.setCellWidget(r, 8, actions)
            # Make sure the row is tall enough for the action buttons
            self._table.setRowHeight(r, 36)

        self._table.setSortingEnabled(True)

    # ── Row actions ──────────────────────────────────────────────────────

    def _on_row_selected(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        id_item = self._table.item(row, 0)
        if not id_item:
            return
        app_id = int(id_item.data(Qt.ItemDataRole.UserRole))
        try:
            app = self._db().get_application(app_id)
        except Exception:
            app = None
        if not app:
            # Might be archived — pull from full list
            app = next(
                (r for r in self._rows_cache if int(r["id"]) == app_id), None
            )
        if not app:
            return
        parts = []
        notes = app.get("notes") or ""
        if notes:
            parts.append(notes)
        jd = (app.get("job_description") or "").strip()
        if jd:
            excerpt = jd[:1500] + ("\n[... more ...]" if len(jd) > 1500 else "")
            parts.append("\n--- JD excerpt ---\n" + excerpt)
        if app.get("job_url"):
            parts.append("\n" + app["job_url"])
        self._detail.setPlainText("\n".join(parts) if parts else
                                  "(No fit analysis or JD stored yet.)")

    def _on_status_changed(self, app_id: int, new_status: str) -> None:
        try:
            db = self._db()
            if new_status == "archived":
                db.archive(app_id)
            else:
                fields: dict[str, Any] = {"status": new_status}
                if new_status == "applied":
                    fields["date_applied"] = date.today().isoformat()
                fields["date_updated"] = date.today().isoformat()
                db.update_application(app_id, **fields)
            self._status_label.setText(f"#{app_id} → {new_status}")
        except Exception as e:
            log.error(f"[JobInbox] Status update failed: {e}")
            QMessageBox.warning(self, "Job Inbox", f"Could not update status: {e}")

    def _on_delete(self, app_id: int) -> None:
        resp = QMessageBox.question(
            self, "Delete application",
            f"Permanently delete application #{app_id}?\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        try:
            self._db().hard_delete(app_id)
            self._status_label.setText(f"#{app_id} deleted.")
            self.refresh()
        except Exception as e:
            log.error(f"[JobInbox] Delete failed: {e}")
            QMessageBox.warning(self, "Job Inbox", f"Could not delete: {e}")

    def _open_url(self, url: str) -> None:
        if not url:
            QMessageBox.information(self, "Job Inbox", "No URL on this row.")
            return
        try:
            webbrowser.open(url)
        except Exception as e:
            QMessageBox.warning(self, "Job Inbox", f"Could not open URL: {e}")

    # ── Pipelines ────────────────────────────────────────────────────────

    def _on_run_pipeline(self, action: str, app_id: int) -> None:
        job_search = self._job_search_talent()
        worker = _PipelineWorker(action, job_search, app_id=app_id)
        worker.done.connect(lambda a, r: self._on_pipeline_done(a, r, app_id))
        worker.failed.connect(self._on_pipeline_failed)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        self._set_busy(True, f"#{app_id}: running {action.replace('_', ' ')}...")
        worker.start()

    def _on_pipeline_done(self, action: str, result: dict, app_id: int) -> None:
        self._set_busy(False)
        success = bool(result.get("success"))
        response = result.get("response") or ""
        title = f"#{app_id} — {action.replace('_', ' ')}"
        if success:
            QMessageBox.information(self, title, response or "Done.")
        else:
            QMessageBox.warning(self, title, response or "Pipeline failed.")
        self.refresh()

    def _on_pipeline_failed(self, action: str, error: str) -> None:
        self._set_busy(False)
        QMessageBox.critical(
            self, f"{action.replace('_', ' ')} failed",
            error
        )

    def _drop_worker(self, worker: _PipelineWorker) -> None:
        try:
            self._workers.remove(worker)
        except ValueError:
            pass
        worker.deleteLater()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy_bar.setVisible(busy)
        if message:
            self._status_label.setText(message)
        elif not busy:
            self._status_label.setText("Ready.")

    # ── Drop-in URL ──────────────────────────────────────────────────────

    def _on_add_url(self) -> None:
        url = (self._url_input.text() or "").strip()
        if not url:
            return
        if not url.startswith("http"):
            QMessageBox.information(
                self, "Job Inbox",
                "Paste a full URL starting with http:// or https://"
            )
            return

        self._url_input.setEnabled(False)
        self._add_url_btn.setEnabled(False)
        self._set_busy(True, "Scraping URL and scoring...")

        job_search = self._job_search_talent()
        worker = _PipelineWorker("add_from_url", job_search, url=url)
        worker.done.connect(self._on_url_added)
        worker.failed.connect(self._on_url_failed)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        worker.start()

    def _on_url_added(self, _action: str, result: dict) -> None:
        self._url_input.setEnabled(True)
        self._add_url_btn.setEnabled(True)
        self._set_busy(False)

        if not result.get("success"):
            QMessageBox.warning(
                self, "Add from URL",
                result.get("error") or "Could not add that URL."
            )
            return

        self._url_input.clear()
        app_id = result.get("id")
        duplicate = result.get("duplicate")
        company = result.get("company", "")
        position = result.get("position", "")
        if duplicate:
            msg = (
                f"That URL is already tracked as #{app_id}: "
                f"{company} — {position}."
            )
        else:
            msg = (
                f"Added #{app_id}: {company} — {position}.\n\n"
                "Fit scoring is running in the background. "
                "Hit Refresh in a minute to see the score."
            )
        QMessageBox.information(self, "Add from URL", msg)
        self.refresh()

    def _on_url_failed(self, _action: str, error: str) -> None:
        self._url_input.setEnabled(True)
        self._add_url_btn.setEnabled(True)
        self._set_busy(False)
        QMessageBox.critical(self, "Add from URL", error)

    # ── Run full search ──────────────────────────────────────────────────

    def _on_run_search(self) -> None:
        """Kick off the configured job search across every saved URL."""
        job_search = self._job_search_talent()
        if job_search is None:
            QMessageBox.warning(
                self, "Run Search",
                "Job search talent is not loaded."
            )
            return

        urls = (job_search._search_config or {}).get("urls", [])
        if not urls:
            QMessageBox.information(
                self, "Run Search",
                "No search URLs configured. In the main chat, say "
                "'add a search URL <link>' to add one."
            )
            return

        self._run_search_btn.setEnabled(False)
        self._add_url_btn.setEnabled(False)
        self._url_input.setEnabled(False)
        self._set_busy(
            True,
            f"Searching {len(urls)} URL(s) — this can take a minute..."
        )

        worker = _PipelineWorker("run_search", job_search)
        worker.done.connect(self._on_search_done)
        worker.failed.connect(self._on_search_failed)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        worker.start()

    def _on_search_done(self, _action: str, result: dict) -> None:
        self._run_search_btn.setEnabled(True)
        self._add_url_btn.setEnabled(True)
        self._url_input.setEnabled(True)
        self._set_busy(False)

        response = result.get("response") or "Search complete."
        if result.get("success"):
            QMessageBox.information(self, "Run Search", response)
        else:
            QMessageBox.warning(self, "Run Search", response)
        self.refresh()

    def _on_search_failed(self, _action: str, error: str) -> None:
        self._run_search_btn.setEnabled(True)
        self._add_url_btn.setEnabled(True)
        self._url_input.setEnabled(True)
        self._set_busy(False)
        QMessageBox.critical(self, "Run Search", error)

    # ── Cleanup ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        # Hide immediately so the window never ghosts on screen while
        # background workers are still finishing.
        self.hide()

        # Workers wrap a blocking Selenium loop, so QThread.quit() can't
        # interrupt them. Disconnect their signals (so they don't fire on
        # a dead dialog) and stash them in a module-level list so Python
        # doesn't GC the QThread object while it's still running.
        for w in list(self._workers):
            try:
                if w.isRunning():
                    try:
                        w.done.disconnect()
                    except Exception:
                        pass
                    try:
                        w.failed.disconnect()
                    except Exception:
                        pass
                    try:
                        w.finished.disconnect()
                    except Exception:
                        pass
                    w.finished.connect(
                        lambda worker=w: _detached_worker_done(worker)
                    )
                    _DETACHED_WORKERS.append(w)
                else:
                    w.deleteLater()
            except Exception:
                pass
        self._workers.clear()
        super().closeEvent(event)
