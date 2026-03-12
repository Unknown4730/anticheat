"""
detector.py
===========
Pure-Python detection logic for the Anti-Cheat Surveillance system.

This module is intentionally free of GUI (tkinter) and vision (OpenCV)
imports so it can be imported and unit-tested in environments that lack
those native libraries.

Exported symbols used by ``gui.py`` and test utilities:
    - ``StudentState``         Per-student strike/movement/status tracker.
    - ``_nearest_centroid_id`` Centroid-based fallback ID assignment.
    - ``_log_strike``          Write a strike event row to SQLite.
    - ``init_db``              Create/connect to the detections SQLite DB.
    - ``STATUS_OK``            Constant: student has no strikes.
    - ``STATUS_WARNING``       Constant: student has 1..MAX_STRIKES-1 strikes.
    - ``STATUS_CHEATING``      Constant: student has >= MAX_STRIKES strikes.
    - ``CONFIG_LABEL_CHEATING``
    - ``CONFIG_LABEL_NOT_CHEATING``
"""

from __future__ import annotations

import math
import os
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Configurable label names (env-var overridable)
# ---------------------------------------------------------------------------

#: Model class label that represents cheating behaviour.
CONFIG_LABEL_CHEATING: str = os.environ.get(
    "CONFIG_LABEL_CHEATING", "students_cheating"
)

#: Model class label that represents normal (not-cheating) behaviour.
CONFIG_LABEL_NOT_CHEATING: str = os.environ.get(
    "CONFIG_LABEL_NOT_CHEATING", "students_not_cheating"
)

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_CHEATING = "cheating"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Create/connect to SQLite DB and ensure the *detections* table exists.

    The parent directory is created automatically so the ``runtime/`` folder
    does not need to exist in advance.

    Args:
        db_path: File-system path (or ``:memory:`` for in-memory testing).

    Returns:
        ``(connection, cursor)`` tuple ready for use.
    """
    # For in-memory DBs there is no parent directory to create.
    parent = os.path.dirname(os.path.abspath(db_path))
    if db_path != ":memory:":
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            timestamp  TEXT,
            label      TEXT,
            strikes    INTEGER
        )
        """
    )
    conn.commit()
    return conn, cursor


def _log_strike(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    student_id: str,
    label: str,
    strikes: int,
) -> None:
    """Insert a single strike event into the *detections* table.

    Args:
        cursor:     Active SQLite cursor.
        conn:       Active SQLite connection (used to commit).
        student_id: String representation of the student's tracking ID.
        label:      Detection class label (e.g. ``"students_cheating"``).
        strikes:    Cumulative strike count *after* this increment.
    """
    cursor.execute(
        """
        INSERT INTO detections (student_id, timestamp, label, strikes)
        VALUES (?, ?, ?, ?)
        """,
        (student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, strikes),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Per-student state
# ---------------------------------------------------------------------------

class StudentState:
    """Holds all per-student counters and status for the current session.

    All dictionaries are keyed by an integer student ID.  Positive IDs come
    from the tracker; negative IDs are temporary values assigned by the
    centroid-fallback logic in ``_nearest_centroid_id``.
    """

    def __init__(self) -> None:
        #: Cumulative cheating-event strike count per ID.
        self.strike_counts: Dict[int, int] = defaultdict(int)
        #: Last centroid (cx, cy) per ID, used to measure movement.
        self.last_positions: Dict[int, Tuple[int, int]] = {}
        #: Cumulative movement-event count per ID.
        self.movement_counts: Dict[int, int] = defaultdict(int)
        #: Derived status per ID: ``STATUS_OK``, ``STATUS_WARNING``, or
        #: ``STATUS_CHEATING``.
        self.status: Dict[int, str] = {}
        #: Whether the first-time-cheating alert has already been sent for
        #: this ID during the current session.
        self._alerted: Dict[int, bool] = defaultdict(bool)

    # ------------------------------------------------------------------
    # Status computation
    # ------------------------------------------------------------------

    def compute_status(self, student_id: int, max_strikes: int) -> str:
        """Return the current status string for *student_id*.

        Rules:
        - ``strikes == 0``               → ``STATUS_OK``
        - ``0 < strikes < max_strikes``  → ``STATUS_WARNING``
        - ``strikes >= max_strikes``     → ``STATUS_CHEATING``

        Args:
            student_id:  Tracking ID of the student.
            max_strikes: Threshold at which the student is flagged.

        Returns:
            One of ``STATUS_OK``, ``STATUS_WARNING``, or ``STATUS_CHEATING``.
        """
        strikes = self.strike_counts[student_id]
        if strikes >= max_strikes:
            return STATUS_CHEATING
        if strikes > 0:
            return STATUS_WARNING
        return STATUS_OK

    # ------------------------------------------------------------------
    # Strike recording
    # ------------------------------------------------------------------

    def record_strike(
        self,
        student_id: int,
        max_strikes: int,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        label: str,
        log_file: Optional[str] = None,
    ) -> str:
        """Increment the strike counter for *student_id* and return new status.

        Saves a DB row on every increment.  On the **first** transition to
        ``STATUS_CHEATING`` it also appends a line to *log_file* (if provided).

        Args:
            student_id:  Tracking ID of the student.
            max_strikes: Threshold at which the student is flagged.
            cursor:      Active SQLite cursor for DB logging.
            conn:        Active SQLite connection for DB logging.
            label:       Detection class label triggering this strike.
            log_file:    Optional path to a plain-text alert log.

        Returns:
            The new status string after incrementing.
        """
        self.strike_counts[student_id] += 1
        new_status = self.compute_status(student_id, max_strikes)
        self.status[student_id] = new_status
        strikes = self.strike_counts[student_id]

        # Persist every strike event to the database.
        _log_strike(cursor, conn, str(student_id), label, strikes)

        # One-time alert on first transition to CHEATING.
        if new_status == STATUS_CHEATING and not self._alerted[student_id]:
            self._alerted[student_id] = True
            if log_file:
                try:
                    with open(log_file, "a", encoding="utf-8") as fh:
                        fh.write(
                            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                            f"CHEATING_FLAGGED student_id={student_id} "
                            f"strikes={strikes}\n"
                        )
                except OSError:
                    pass  # Log file write is best-effort.

        return new_status

    # ------------------------------------------------------------------
    # Movement tracking
    # ------------------------------------------------------------------

    def update_movement(
        self,
        student_id: int,
        cx: int,
        cy: int,
        threshold_px: int,
    ) -> bool:
        """Update the centroid for *student_id* and check for movement.

        Args:
            student_id:    Tracking ID of the student.
            cx, cy:        Current centroid coordinates.
            threshold_px:  Pixel displacement that counts as movement.

        Returns:
            ``True`` if the centroid moved more than *threshold_px* pixels
            since the last update, ``False`` otherwise.
        """
        moved = False
        if student_id in self.last_positions:
            lx, ly = self.last_positions[student_id]
            dist = math.hypot(cx - lx, cy - ly)
            if dist > threshold_px:
                self.movement_counts[student_id] += 1
                moved = True
        self.last_positions[student_id] = (cx, cy)
        return moved

    # ------------------------------------------------------------------
    # Session reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all counters — call this at the start of a new exam session."""
        self.strike_counts.clear()
        self.last_positions.clear()
        self.movement_counts.clear()
        self.status.clear()
        self._alerted.clear()


# ---------------------------------------------------------------------------
# Centroid-based fallback ID matching
# ---------------------------------------------------------------------------

def _nearest_centroid_id(
    cx: int,
    cy: int,
    last_positions: Dict[int, Tuple[int, int]],
    next_temp_id: int,
    match_radius: int = 80,
) -> Tuple[int, int]:
    """Return the closest existing temp ID within *match_radius* pixels.

    If no existing ID is close enough a new one is allocated.  New IDs are
    negative integers so they cannot collide with positive tracker-assigned
    IDs.

    Args:
        cx, cy:         Centroid of the current detection.
        last_positions: Dict mapping existing IDs → last (cx, cy).
        next_temp_id:   Counter for allocating new negative IDs.
        match_radius:   Maximum pixel distance to consider a match.

    Returns:
        ``(matched_or_new_id, updated_next_temp_id)``
    """
    best_id: Optional[int] = None
    best_dist = float("inf")
    for existing_id, (lx, ly) in last_positions.items():
        d = math.hypot(cx - lx, cy - ly)
        if d < best_dist and d <= match_radius:
            best_dist = d
            best_id = existing_id

    if best_id is not None:
        return best_id, next_temp_id

    # Allocate a new temporary negative ID.
    new_id = -next_temp_id
    return new_id, next_temp_id + 1
