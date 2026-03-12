#!/usr/bin/env python3
"""
simulate_detections.py
======================
Replay a sequence of fake detection frames to exercise the per-student
strike logic and verify multi-student handling and the 3-strike behaviour.

This script does **not** require a webcam, YOLO model, or running GUI.
It imports only the pure-Python logic from ``src/detector.py`` (``StudentState``
and ``_nearest_centroid_id``).

Usage
-----
Run standalone::

    python scripts/simulate_detections.py

Run as a unittest::

    python -m unittest scripts/simulate_detections.py -v

Scenarios covered
-----------------
1. Single student accumulates 3 cheating strikes → flagged as CHEATING.
2. Two simultaneous students are tracked independently; only the one with 3
   strikes is flagged.
3. Movement events are counted but do not directly cause a strike.
4. Centroid-based ID fallback (no tracker IDs) maintains per-person counters.
"""

from __future__ import annotations

import sys
import os
import sqlite3
import unittest
import tempfile
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Make src/ importable from the scripts/ directory
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)

from src.detector import (  # noqa: E402
    StudentState,
    STATUS_OK,
    STATUS_WARNING,
    STATUS_CHEATING,
    _nearest_centroid_id,
    _log_strike,
    init_db,
    CONFIG_LABEL_CHEATING,
    CONFIG_LABEL_NOT_CHEATING,
)


# ---------------------------------------------------------------------------
# Helpers for fake DB
# ---------------------------------------------------------------------------

def _make_db() -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Return an in-memory SQLite connection with the detections table."""
    conn, cursor = init_db(":memory:")
    return conn, cursor


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

FakeBox = Tuple[int, int, int, int, float, str]
"""(x1, y1, x2, y2, confidence, label)"""


def simulate_frame(
    state: StudentState,
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    detections: List[Tuple[int, FakeBox]],
    max_strikes: int = 3,
    conf_thresh: float = 0.5,
    movement_threshold: int = 30,
) -> None:
    """Process one simulated frame of detections.

    Args:
        state:           The per-student state object being updated.
        cursor / conn:   SQLite connection for logging.
        detections:      List of ``(student_id, (x1,y1,x2,y2,conf,label))``.
        max_strikes:     Maximum strikes before flagging.
        conf_thresh:     Minimum confidence to process a box.
        movement_threshold: Pixel displacement counted as movement.
    """
    for student_id, (x1, y1, x2, y2, conf, label) in detections:
        if conf < conf_thresh:
            continue  # Skip low-confidence detections

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        state.update_movement(student_id, cx, cy, movement_threshold)

        if label == CONFIG_LABEL_CHEATING:
            state.record_strike(
                student_id, max_strikes, cursor, conn, label, log_file=None
            )
        else:
            state.status[student_id] = state.compute_status(student_id, max_strikes)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestStrikeLogic(unittest.TestCase):
    """Verify the 3-strike rule and multi-student handling."""

    def setUp(self) -> None:
        self.conn, self.cursor = _make_db()
        self.state = StudentState()

    def tearDown(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Single-student tests
    # ------------------------------------------------------------------

    def test_zero_strikes_is_ok(self) -> None:
        """A new student with no detections has STATUS_OK."""
        self.assertEqual(
            self.state.compute_status(42, max_strikes=3), STATUS_OK
        )

    def test_one_strike_is_warning(self) -> None:
        """One strike → STATUS_WARNING (not yet flagged)."""
        simulate_frame(
            self.state, self.cursor, self.conn,
            [(1, (10, 10, 50, 50, 0.9, CONFIG_LABEL_CHEATING))],
        )
        self.assertEqual(self.state.status[1], STATUS_WARNING)
        self.assertEqual(self.state.strike_counts[1], 1)

    def test_two_strikes_is_warning(self) -> None:
        """Two strikes → STATUS_WARNING (still not flagged)."""
        for _ in range(2):
            simulate_frame(
                self.state, self.cursor, self.conn,
                [(1, (10, 10, 50, 50, 0.9, CONFIG_LABEL_CHEATING))],
            )
        self.assertEqual(self.state.status[1], STATUS_WARNING)
        self.assertEqual(self.state.strike_counts[1], 2)

    def test_three_strikes_is_cheating(self) -> None:
        """A student with exactly MAX_STRIKES cheating events is flagged as CHEATING."""
        for _ in range(3):
            simulate_frame(
                self.state, self.cursor, self.conn,
                [(99, (10, 10, 50, 50, 0.9, CONFIG_LABEL_CHEATING))],
            )
        self.assertEqual(self.state.status[99], STATUS_CHEATING)
        self.assertEqual(self.state.strike_counts[99], 3)

    def test_beyond_three_strikes_remains_cheating(self) -> None:
        """Additional strikes beyond MAX_STRIKES keep the student as CHEATING."""
        for _ in range(5):
            simulate_frame(
                self.state, self.cursor, self.conn,
                [(7, (10, 10, 50, 50, 0.9, CONFIG_LABEL_CHEATING))],
            )
        self.assertEqual(self.state.status[7], STATUS_CHEATING)
        self.assertGreaterEqual(self.state.strike_counts[7], 3)

    def test_not_cheating_does_not_add_strike(self) -> None:
        """Frames where the label is 'students_not_cheating' do not add strikes."""
        for _ in range(5):
            simulate_frame(
                self.state, self.cursor, self.conn,
                [(3, (10, 10, 50, 50, 0.9, CONFIG_LABEL_NOT_CHEATING))],
            )
        self.assertEqual(self.state.strike_counts[3], 0)
        self.assertEqual(self.state.status[3], STATUS_OK)

    def test_low_confidence_box_is_skipped(self) -> None:
        """Detections below conf_thresh must NOT add strikes."""
        for _ in range(5):
            simulate_frame(
                self.state, self.cursor, self.conn,
                [(4, (10, 10, 50, 50, 0.3, CONFIG_LABEL_CHEATING))],
                conf_thresh=0.5,
            )
        self.assertEqual(self.state.strike_counts[4], 0)

    # ------------------------------------------------------------------
    # Multi-student tests
    # ------------------------------------------------------------------

    def test_multiple_students_independent_counters(self) -> None:
        """Two students have separate strike counters."""
        # Student A: 3 cheating events → CHEATING
        for _ in range(3):
            simulate_frame(
                self.state, self.cursor, self.conn,
                [(10, (10, 10, 50, 50, 0.9, CONFIG_LABEL_CHEATING))],
            )
        # Student B: 1 cheating event → WARNING
        simulate_frame(
            self.state, self.cursor, self.conn,
            [(20, (200, 200, 250, 250, 0.9, CONFIG_LABEL_CHEATING))],
        )

        self.assertEqual(self.state.status[10], STATUS_CHEATING)
        self.assertEqual(self.state.status[20], STATUS_WARNING)
        self.assertEqual(self.state.strike_counts[10], 3)
        self.assertEqual(self.state.strike_counts[20], 1)

    def test_simultaneous_students_same_frame(self) -> None:
        """Multiple students in the same frame are all tracked correctly."""
        detections = [
            (1, (10, 10, 50, 50, 0.9, CONFIG_LABEL_CHEATING)),
            (2, (200, 10, 250, 50, 0.9, CONFIG_LABEL_NOT_CHEATING)),
            (3, (400, 10, 450, 50, 0.9, CONFIG_LABEL_CHEATING)),
        ]
        for _ in range(3):
            simulate_frame(self.state, self.cursor, self.conn, detections)

        self.assertEqual(self.state.status[1], STATUS_CHEATING)
        self.assertEqual(self.state.status[2], STATUS_OK)       # never cheated
        self.assertEqual(self.state.status[3], STATUS_CHEATING)

    # ------------------------------------------------------------------
    # Movement tests
    # ------------------------------------------------------------------

    def test_movement_counted_but_no_strike(self) -> None:
        """Movement alone does not add a cheating strike."""
        # Student moves across frames (large centroid displacement)
        positions = [(10, 10, 50, 50), (100, 100, 140, 140), (200, 200, 240, 240)]
        for x1, y1, x2, y2 in positions:
            simulate_frame(
                self.state, self.cursor, self.conn,
                [(5, (x1, y1, x2, y2, 0.9, CONFIG_LABEL_NOT_CHEATING))],
                movement_threshold=30,
            )
        self.assertGreater(self.state.movement_counts[5], 0)
        self.assertEqual(self.state.strike_counts[5], 0)
        self.assertEqual(self.state.status[5], STATUS_OK)

    # ------------------------------------------------------------------
    # Centroid-fallback tests
    # ------------------------------------------------------------------

    def test_centroid_fallback_same_id_returned(self) -> None:
        """Re-appearing centroid within match_radius gets the same temp ID."""
        state = StudentState()
        # Place a detection at (100, 100)
        id1, next_id = _nearest_centroid_id(100, 100, state.last_positions, 1)
        state.last_positions[id1] = (100, 100)

        # Second detection nearby (same person, small movement)
        id2, _ = _nearest_centroid_id(110, 105, state.last_positions, next_id)
        self.assertEqual(id1, id2)

    def test_centroid_fallback_new_id_for_distant_box(self) -> None:
        """A centroid far from all existing tracks gets a new temp ID."""
        state = StudentState()
        id1, next_id = _nearest_centroid_id(100, 100, state.last_positions, 1)
        state.last_positions[id1] = (100, 100)

        # Detection far away (different person)
        id2, _ = _nearest_centroid_id(600, 400, state.last_positions, next_id)
        self.assertNotEqual(id1, id2)

    # ------------------------------------------------------------------
    # DB persistence tests
    # ------------------------------------------------------------------

    def test_db_rows_inserted_on_strike(self) -> None:
        """A DB row is written for each cheating strike event."""
        for _ in range(3):
            simulate_frame(
                self.state, self.cursor, self.conn,
                [(55, (10, 10, 50, 50, 0.9, CONFIG_LABEL_CHEATING))],
            )
        self.cursor.execute(
            "SELECT COUNT(*) FROM detections WHERE student_id = '55'"
        )
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 3)

    def test_db_no_rows_for_not_cheating(self) -> None:
        """No DB row is written when the label is 'students_not_cheating'."""
        for _ in range(5):
            simulate_frame(
                self.state, self.cursor, self.conn,
                [(66, (10, 10, 50, 50, 0.9, CONFIG_LABEL_NOT_CHEATING))],
            )
        self.cursor.execute(
            "SELECT COUNT(*) FROM detections WHERE student_id = '66'"
        )
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 0)

    # ------------------------------------------------------------------
    # Reset tests
    # ------------------------------------------------------------------

    def test_clear_resets_all_state(self) -> None:
        """StudentState.clear() removes all per-student data."""
        for _ in range(3):
            simulate_frame(
                self.state, self.cursor, self.conn,
                [(77, (10, 10, 50, 50, 0.9, CONFIG_LABEL_CHEATING))],
            )
        self.state.clear()
        self.assertEqual(len(self.state.strike_counts), 0)
        self.assertEqual(len(self.state.status), 0)
        self.assertEqual(len(self.state.movement_counts), 0)
        self.assertEqual(len(self.state.last_positions), 0)


# ---------------------------------------------------------------------------
# Narrative simulation printout (run when invoked directly, not via unittest)
# ---------------------------------------------------------------------------

def run_narrative_simulation() -> None:
    """Print a human-readable walkthrough of the strike logic."""
    conn, cursor = _make_db()
    state = StudentState()
    MAX = 3

    print("=" * 60)
    print("Anti-Cheat Simulation — 3-strike rule, multi-student")
    print("=" * 60)

    # Define a simple scenario
    scenario: List[Tuple[str, List[Tuple[int, FakeBox]]]] = [
        ("Frame 1 – two students detected; Alice cheating, Bob not",
         [(1, (10, 10, 50, 50, 0.92, CONFIG_LABEL_CHEATING)),
          (2, (200, 10, 240, 50, 0.88, CONFIG_LABEL_NOT_CHEATING))]),
        ("Frame 2 – Alice cheating again; Bob still ok",
         [(1, (12, 11, 52, 51, 0.90, CONFIG_LABEL_CHEATING)),
          (2, (202, 12, 242, 52, 0.85, CONFIG_LABEL_NOT_CHEATING))]),
        ("Frame 3 – Alice cheating a third time → FLAGGED",
         [(1, (14, 12, 54, 52, 0.91, CONFIG_LABEL_CHEATING)),
          (2, (204, 14, 244, 54, 0.87, CONFIG_LABEL_NOT_CHEATING))]),
        ("Frame 4 – Carol (ID 3) appears and cheats once",
         [(1, (14, 12, 54, 52, 0.89, CONFIG_LABEL_CHEATING)),
          (2, (204, 14, 244, 54, 0.86, CONFIG_LABEL_NOT_CHEATING)),
          (3, (400, 10, 440, 50, 0.95, CONFIG_LABEL_CHEATING))]),
        ("Frame 5 – low confidence detection ignored",
         [(1, (14, 12, 54, 52, 0.25, CONFIG_LABEL_CHEATING)),  # ignored
          (2, (220, 20, 260, 60, 0.85, CONFIG_LABEL_NOT_CHEATING))]),
    ]

    for description, detections in scenario:
        print(f"\n{description}")
        simulate_frame(state, cursor, conn, detections, max_strikes=MAX)
        for sid in sorted(set(state.strike_counts) | set(state.status)):
            st = state.status.get(sid, state.compute_status(sid, MAX))
            print(
                f"  Student {sid}: strikes={state.strike_counts[sid]}, "
                f"movement={state.movement_counts[sid]}, status={st.upper()}"
            )

    cursor.execute("SELECT COUNT(*) FROM detections")
    total_rows = cursor.fetchone()[0]
    print(f"\nTotal DB rows written: {total_rows}")
    conn.close()

    print("\n✓ Simulation complete.")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--unittest" in sys.argv or "-v" in sys.argv:
        sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--unittest"]
        unittest.main()
    else:
        run_narrative_simulation()
        print("\nRunning unit tests …\n")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestStrikeLogic)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
