"""Talent Marketplace dialog — browse, search, install, and remove talents.

Layout:
    +----------------------------------------------+
    | Talent Marketplace                           |
    | [Search...              ] [Category v] [Refresh] |
    +----------------------------------------------+
    | talent_card  | talent_card  | talent_card    |
    | talent_card  | talent_card  | ...            |
    +----------------------------------------------+
    | Status bar message                           |
    +----------------------------------------------+
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QLineEdit, QComboBox, QScrollArea,
                             QWidget, QFrame, QGridLayout, QMessageBox,
                             QPlainTextEdit, QDialogButtonBox)
from PyQt6.QtCore import pyqtSignal, Qt, QThread, pyqtSlot
from PyQt6.QtGui import QFont


class CatalogWorker(QThread):
    """Fetch the marketplace catalog in a background thread."""
    catalog_ready = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, marketplace_client, force_refresh=False):
        super().__init__()
        self.client = marketplace_client
        self.force_refresh = force_refresh

    def run(self):
        try:
            catalog = self.client.get_catalog(force_refresh=self.force_refresh)
            self.catalog_ready.emit(catalog)
        except Exception as e:
            self.error.emit(str(e))


class InstallWorker(QThread):
    """Download + validate a talent in a background thread (no disk write).

    The actual write happens on the GUI thread via commit_install() only
    after the user reviews the source, so nothing untrusted is persisted or
    imported without an explicit approval step.
    """
    install_done = pyqtSignal(dict)  # result dict from fetch_and_validate
    error = pyqtSignal(str)

    def __init__(self, marketplace_client, talent_entry):
        super().__init__()
        self.client = marketplace_client
        self.talent_entry = talent_entry

    def run(self):
        try:
            result = self.client.fetch_and_validate(self.talent_entry)
            self.install_done.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class UninstallWorker(QThread):
    """Remove a talent file in a background thread.

    uninstall_talent() scans and AST-parses every file in talents/user/ before
    deleting the match; running it inline would stall the dialog on a large
    user-talent directory. Kept symmetric with InstallWorker.
    """
    uninstall_done = pyqtSignal(dict)  # result dict from uninstall_talent
    error = pyqtSignal(str)

    def __init__(self, marketplace_client, talent_name, authorization=None):
        super().__init__()
        self.client = marketplace_client
        self.talent_name = talent_name
        self.authorization = authorization

    def run(self):
        try:
            result = self.client.uninstall_talent(
                self.talent_name, authorization=self.authorization)
            self.uninstall_done.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class TalentReviewDialog(QDialog):
    """Show downloaded talent source + integrity/danger findings for approval.

    The marketplace serves arbitrary Python that Talon will import and run.
    No static check can prove such code safe, so the final gate is the user
    reading what they're about to install. Cancel is the default button.
    """

    def __init__(self, talent_entry, source_code, warnings, verified, parent=None):
        super().__init__(parent)
        name = talent_entry.get("name", "talent")
        self.setWindowTitle(f"Review: {name}")
        self.setMinimumSize(640, 520)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        author = talent_entry.get("author", "unknown")
        version = talent_entry.get("version", "")
        head = QLabel(f"<b>{name}</b>  v{version}  —  by {author}")
        head.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(head)

        if verified:
            integrity = QLabel("Integrity: sha256 verified against catalog")
            integrity.setStyleSheet("color: #2e7d32;")
        else:
            integrity = QLabel(
                "Integrity: NOT verified — catalog entry has no sha256 hash. "
                "Only install if you trust the source.")
            integrity.setStyleSheet("color: #c62828;")
        integrity.setWordWrap(True)
        layout.addWidget(integrity)

        if warnings:
            warn = QLabel(
                "Risky constructs found (review carefully):\n  - "
                + "\n  - ".join(warnings))
            warn.setStyleSheet("color: #c62828;")
            warn.setWordWrap(True)
            layout.addWidget(warn)

        layout.addWidget(QLabel("Source code:"))
        viewer = QPlainTextEdit()
        viewer.setPlainText(source_code)
        viewer.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        viewer.setFont(mono)
        layout.addWidget(viewer, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok)
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("Install")
        cancel_btn = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_btn.setDefault(True)
        cancel_btn.setFocus()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class TalentCard(QFrame):
    """A card widget displaying one marketplace talent."""

    install_requested = pyqtSignal(dict)  # talent_entry
    uninstall_requested = pyqtSignal(str)  # talent_name

    def __init__(self, talent_entry, is_installed=False):
        super().__init__()
        self.talent_entry = talent_entry
        self._is_installed = is_installed
        self.setObjectName("marketplace_card")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedHeight(140)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        # Title row
        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        name_label = QLabel(talent_entry.get("name", "Unknown"))
        name_label.setObjectName("marketplace_card_title")
        name_font = QFont()
        name_font.setBold(True)
        name_font.setPointSize(11)
        name_label.setFont(name_font)
        title_row.addWidget(name_label)

        # Category badge
        category = talent_entry.get("category", "")
        if category:
            cat_label = QLabel(category)
            cat_label.setObjectName("marketplace_card_category")
            title_row.addWidget(cat_label)

        title_row.addStretch()

        # Version
        version = talent_entry.get("version", "")
        if version:
            ver_label = QLabel(f"v{version}")
            ver_label.setObjectName("marketplace_card_version")
            title_row.addWidget(ver_label)

        layout.addLayout(title_row)

        # Description
        desc = talent_entry.get("description", "No description available.")
        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setObjectName("marketplace_card_desc")
        layout.addWidget(desc_label)

        # Author
        author = talent_entry.get("author", "")
        if author:
            author_label = QLabel(f"by {author}")
            author_label.setObjectName("marketplace_card_author")
            layout.addWidget(author_label)

        layout.addStretch()

        # Action row
        action_row = QHBoxLayout()
        action_row.addStretch()

        if is_installed:
            self.action_btn = QPushButton("Remove")
            self.action_btn.setObjectName("marketplace_remove_btn")
            self.action_btn.clicked.connect(self._on_remove)
        else:
            self.action_btn = QPushButton("Install")
            self.action_btn.setObjectName("marketplace_install_btn")
            self.action_btn.clicked.connect(self._on_install)

        self.action_btn.setFixedWidth(90)
        self.action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        action_row.addWidget(self.action_btn)

        layout.addLayout(action_row)

    def _on_install(self):
        self.action_btn.setEnabled(False)
        self.action_btn.setText("Installing...")
        self.install_requested.emit(self.talent_entry)

    def _on_remove(self):
        self.action_btn.setEnabled(False)
        self.action_btn.setText("Removing...")
        self.uninstall_requested.emit(self.talent_entry.get("name", ""))

    def mark_installed(self):
        """Switch button from Install to Remove after successful install."""
        self._is_installed = True
        self.action_btn.setText("Remove")
        self.action_btn.setObjectName("marketplace_remove_btn")
        self.action_btn.setEnabled(True)
        self.action_btn.setStyleSheet(self.action_btn.styleSheet())
        try:
            self.action_btn.clicked.disconnect()
        except TypeError:
            pass
        self.action_btn.clicked.connect(self._on_remove)

    def mark_uninstalled(self):
        """Switch button back to Install after removal."""
        self._is_installed = False
        self.action_btn.setText("Install")
        self.action_btn.setObjectName("marketplace_install_btn")
        self.action_btn.setEnabled(True)
        self.action_btn.setStyleSheet(self.action_btn.styleSheet())
        try:
            self.action_btn.clicked.disconnect()
        except TypeError:
            pass
        self.action_btn.clicked.connect(self._on_install)

    def reset_button(self):
        """Re-enable button after error."""
        if self._is_installed:
            self.action_btn.setText("Remove")
        else:
            self.action_btn.setText("Install")
        self.action_btn.setEnabled(True)


class MarketplaceDialog(QDialog):
    """Main marketplace browsing dialog."""

    talent_installed = pyqtSignal(str)   # filepath of installed .py
    talent_removed = pyqtSignal(str)     # talent name removed

    def __init__(self, marketplace_client, installed_names=None, parent=None):
        super().__init__(parent)
        self.client = marketplace_client
        self._installed = installed_names or set()
        self._catalog = []
        self._cards = {}  # name -> TalentCard
        self._worker = None
        self._install_worker = None
        self._uninstall_worker = None
        self._active_workers = set()

        self.setWindowTitle("Talent Marketplace")
        self.setMinimumSize(700, 500)
        self.resize(800, 560)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Header ────────────────────────────────────────
        header = QLabel("Talent Marketplace")
        header.setObjectName("marketplace_header")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(14)
        header.setFont(header_font)
        layout.addWidget(header)

        # ── Search / filter bar ───────────────────────────
        filter_row = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search talents...")
        self.search_input.setObjectName("marketplace_search")
        self.search_input.textChanged.connect(self._apply_filters)
        filter_row.addWidget(self.search_input, 1)

        self.category_combo = QComboBox()
        self.category_combo.setObjectName("marketplace_category")
        self.category_combo.addItem("All Categories")
        self.category_combo.currentTextChanged.connect(self._apply_filters)
        filter_row.addWidget(self.category_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setObjectName("marketplace_refresh_btn")
        self.refresh_btn.clicked.connect(self._refresh_catalog)
        filter_row.addWidget(self.refresh_btn)

        layout.addLayout(filter_row)

        # ── Scrollable card grid ──────────────────────────
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setObjectName("marketplace_scroll")

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.grid_container)

        layout.addWidget(self.scroll, 1)

        # ── Status bar ────────────────────────────────────
        self.status_label = QLabel("Loading catalog...")
        self.status_label.setObjectName("marketplace_status")
        layout.addWidget(self.status_label)

        # ── Close button ──────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Start loading
        self._load_catalog()

    # ── Catalog loading ───────────────────────────────────

    def _load_catalog(self, force=False):
        self.status_label.setText("Fetching catalog...")
        self.refresh_btn.setEnabled(False)
        self._worker = CatalogWorker(self.client, force_refresh=force)
        self._worker.catalog_ready.connect(self._on_catalog_loaded)
        self._worker.error.connect(self._on_catalog_error)
        self._retain(self._worker)
        self._worker.start()

    def _refresh_catalog(self):
        self._load_catalog(force=True)

    @pyqtSlot(list)
    def _on_catalog_loaded(self, catalog):
        self._catalog = catalog
        self.refresh_btn.setEnabled(True)

        # Populate category dropdown
        categories = sorted(set(
            t.get("category", "Uncategorized") for t in catalog if t.get("category")
        ))
        self.category_combo.blockSignals(True)
        current = self.category_combo.currentText()
        self.category_combo.clear()
        self.category_combo.addItem("All Categories")
        for cat in categories:
            self.category_combo.addItem(cat)
        idx = self.category_combo.findText(current)
        if idx >= 0:
            self.category_combo.setCurrentIndex(idx)
        self.category_combo.blockSignals(False)

        self._render_cards(catalog)
        self.status_label.setText(f"{len(catalog)} talent(s) available")

    @pyqtSlot(str)
    def _on_catalog_error(self, error):
        self.refresh_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error}")

    # ── Card rendering ────────────────────────────────────

    def _render_cards(self, talents):
        """Clear and re-render talent cards in the grid."""
        # Clear existing cards
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()

        if not talents:
            empty = QLabel("No talents found. Try refreshing or adjusting your search.")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid_layout.addWidget(empty, 0, 0, 1, 2)
            return

        cols = 2  # 2-column grid
        for i, entry in enumerate(talents):
            name = entry.get("name", "")
            is_installed = name in self._installed
            card = TalentCard(entry, is_installed=is_installed)
            card.install_requested.connect(self._on_install_requested)
            card.uninstall_requested.connect(self._on_uninstall_requested)
            self._cards[name] = card
            self.grid_layout.addWidget(card, i // cols, i % cols)

    def _apply_filters(self):
        """Filter displayed cards by search text and category."""
        search = self.search_input.text().lower().strip()
        category = self.category_combo.currentText()

        filtered = []
        for entry in self._catalog:
            # Category filter
            if category != "All Categories":
                if entry.get("category", "Uncategorized") != category:
                    continue
            # Search filter
            if search:
                searchable = " ".join([
                    entry.get("name", ""),
                    entry.get("description", ""),
                    entry.get("author", ""),
                    " ".join(entry.get("keywords", [])),
                ]).lower()
                if search not in searchable:
                    continue
            filtered.append(entry)

        self._render_cards(filtered)
        self.status_label.setText(
            f"Showing {len(filtered)} of {len(self._catalog)} talent(s)")

    # ── Install ───────────────────────────────────────────

    def _on_install_requested(self, talent_entry):
        name = talent_entry.get("name", "")
        self.status_label.setText(f"Installing {name}...")

        self._install_worker = InstallWorker(self.client, talent_entry)
        self._install_worker.install_done.connect(
            lambda result, n=name: self._on_install_done(n, result))
        self._install_worker.error.connect(
            lambda err, n=name: self._on_install_error(n, err))
        # Retain like the uninstall path: clicking Install on a second card
        # reassigns self._install_worker, and without a held reference the
        # first worker's QThread is destroyed mid-run, which aborts the app.
        self._retain(self._install_worker)
        self._install_worker.start()

    def _on_install_done(self, name, result):
        if not result.get("success"):
            error = result.get("error", "Unknown error")
            if name in self._cards:
                self._cards[name].reset_button()
            self.status_label.setText(f"Install failed: {error}")
            return

        # Source downloaded + validated but NOT yet written. Make the user
        # review it (and any danger/integrity findings) before we commit.
        entry = self._cards[name].talent_entry if name in self._cards else {"name": name}
        authorization = self.client.request_plugin_change(
            "install", name, result.get("filename", ""))
        if authorization and not (
                authorization.allowed or authorization.confirmation_required):
            if name in self._cards:
                self._cards[name].reset_button()
            self.status_label.setText(
                self.client.capabilities.denial_message(authorization))
            return
        review = TalentReviewDialog(
            entry,
            result.get("source_code", ""),
            result.get("warnings", []),
            result.get("verified", False),
            parent=self,
        )
        if review.exec() != QDialog.DialogCode.Accepted:
            if authorization and authorization.confirmation_required:
                self.client.capabilities.cancel(
                    authorization.request.request_id,
                    source=self.client.command_source)
            if name in self._cards:
                self._cards[name].reset_button()
            self.status_label.setText(f"Install of '{name}' cancelled.")
            return

        commit = self.client.commit_install(
            result.get("filename", ""), result.get("source_code", ""),
            authorization=authorization)
        if commit.get("success"):
            self._installed.add(name)
            if name in self._cards:
                self._cards[name].mark_installed()
            self.status_label.setText(f"Installed '{name}' successfully!")
            self.talent_installed.emit(commit["filepath"])
        else:
            if name in self._cards:
                self._cards[name].reset_button()
            self.status_label.setText(
                f"Install failed: {commit.get('error', 'Unknown error')}")

    def _on_install_error(self, name, error):
        if name in self._cards:
            self._cards[name].reset_button()
        self.status_label.setText(f"Install error: {error}")

    # ── Uninstall ─────────────────────────────────────────

    def _on_uninstall_requested(self, talent_name):
        authorization = self.client.request_plugin_change(
            "remove", talent_name)
        if authorization and not (
                authorization.allowed or authorization.confirmation_required):
            if talent_name in self._cards:
                self._cards[talent_name].reset_button()
            self.status_label.setText(
                self.client.capabilities.denial_message(authorization))
            return
        reply = QMessageBox.question(
            self, "Remove Talent",
            f"Remove '{talent_name}'? This will delete the talent file.\n"
            f"(Requires restart to fully unload.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            if authorization and authorization.confirmation_required:
                self.client.capabilities.cancel(
                    authorization.request.request_id,
                    source=self.client.command_source)
            if talent_name in self._cards:
                self._cards[talent_name].reset_button()
            return

        self.status_label.setText(f"Removing {talent_name}...")
        self._uninstall_worker = UninstallWorker(
            self.client, talent_name, authorization=authorization)
        self._uninstall_worker.uninstall_done.connect(
            lambda result, n=talent_name: self._on_uninstall_done(n, result))
        self._uninstall_worker.error.connect(
            lambda err, n=talent_name: self._on_uninstall_error(n, err))
        self._retain(self._uninstall_worker)
        self._uninstall_worker.start()

    def _on_uninstall_done(self, talent_name, result):
        if result.get("success"):
            self._installed.discard(talent_name)
            if talent_name in self._cards:
                self._cards[talent_name].mark_uninstalled()
            self.status_label.setText(
                f"Removed '{talent_name}'. Restart to fully unload.")
            self.talent_removed.emit(talent_name)
        else:
            error = result.get("error", "Unknown error")
            if talent_name in self._cards:
                self._cards[talent_name].reset_button()
            self.status_label.setText(f"Remove failed: {error}")

    def _on_uninstall_error(self, talent_name, error):
        if talent_name in self._cards:
            self._cards[talent_name].reset_button()
        self.status_label.setText(f"Remove error: {error}")

    def _retain(self, worker):
        """Hold a reference to a running QThread until QThread.finished fires.

        Prevents the GC from destroying a worker mid-run when its attribute is
        reassigned ("QThread: Destroyed while thread is still running"). The
        receiver is a bound method, so the cross-thread connection is queued and
        the reference is dropped on the GUI thread after the worker terminates.
        """
        self._active_workers.add(worker)
        worker.finished.connect(self._discard_worker)

    def _discard_worker(self):
        worker = self.sender()
        if worker is not None:
            self._active_workers.discard(worker)
