"""
Unit tests for Arah.AI Schedule Extractor.

Covers the core utility functions in extractor.py and the validator module.
Run with:  python -m pytest tests/ -v
"""

import json
import os
import sys
import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from extractor import (
    fuzzy_match_day,
    parse_roman,
    add_minutes,
    split_time,
    is_likely_course_name,
    is_likely_class_code,
    ScheduleExtractor,
)
from validator import validate_extraction, ValidationResult


# ─────────────────────────────────────────────────────────────
# fuzzy_match_day
# ─────────────────────────────────────────────────────────────

class TestFuzzyMatchDay:
    """Tests for the 4-strategy fuzzy day matcher."""

    @pytest.mark.parametrize("inp,expected", [
        ("SENIN", "SENIN"),
        ("senin", "SENIN"),
        ("SELASA", "SELASA"),
        ("RABU", "RABU"),
        ("KAMIS", "KAMIS"),
        ("JUMAT", "JUMAT"),
        ("SABTU", "SABTU"),
        ("MINGGU", "MINGGU"),
    ])
    def test_exact_match(self, inp, expected):
        assert fuzzy_match_day(inp) == expected

    @pytest.mark.parametrize("inp,expected", [
        ("MONDAY", "SENIN"),
        ("FRIDAY", "JUMAT"),
        ("SATURDAY", "SABTU"),
    ])
    def test_english_days(self, inp, expected):
        assert fuzzy_match_day(inp) == expected

    @pytest.mark.parametrize("inp,expected", [
        ("NINES", "SENIN"),     # reversed
        ("KSAMI", "KAMIS"),     # sorted-char
        ("SAUBT", "SABTU"),     # sorted-char
        ("JUTMA", "JUMAT"),     # edit-distance
    ])
    def test_ocr_artifacts(self, inp, expected):
        assert fuzzy_match_day(inp) == expected

    def test_empty_and_none(self):
        assert fuzzy_match_day(None) is None
        assert fuzzy_match_day("") is None
        assert fuzzy_match_day("   ") is None

    def test_garbage_string(self):
        # Very short or completely unrelated should return None
        assert fuzzy_match_day("XZ") is None


# ─────────────────────────────────────────────────────────────
# parse_roman
# ─────────────────────────────────────────────────────────────

class TestParseRoman:

    @pytest.mark.parametrize("inp,expected", [
        ("I", 1), ("II", 2), ("III", 3), ("IV", 4),
        ("V", 5), ("X", 10), ("XII", 12), ("XVI", 16),
    ])
    def test_valid_numerals(self, inp, expected):
        assert parse_roman(inp) == expected

    def test_with_separator(self):
        assert parse_roman("III-IV") == 3  # takes first
        assert parse_roman("V/VI") == 5

    def test_none_and_invalid(self):
        assert parse_roman(None) == 0
        assert parse_roman("ABC") == 0
        assert parse_roman("") == 0


# ─────────────────────────────────────────────────────────────
# add_minutes
# ─────────────────────────────────────────────────────────────

class TestAddMinutes:

    def test_basic(self):
        assert add_minutes("07:00", 50) == "07:50"
        assert add_minutes("07:00", 100) == "08:40"

    def test_midnight_wrap(self):
        assert add_minutes("23:30", 60) == "00:30"

    def test_invalid_input(self):
        assert add_minutes("invalid", 10) is None


# ─────────────────────────────────────────────────────────────
# split_time
# ─────────────────────────────────────────────────────────────

class TestSplitTime:

    def test_dot_separator(self):
        assert split_time("07.00-07.50") == ("07:00", "07:50")

    def test_colon_separator(self):
        assert split_time("07:00-07:50") == ("07:00", "07:50")

    def test_spaced_ocr(self):
        assert split_time("0 7. 0 0 - 0 7. 5 0") == ("07:00", "07:50")

    def test_em_dash(self):
        assert split_time("08:00\u201309:30") == ("08:00", "09:30")

    def test_none_input(self):
        assert split_time(None) == (None, None)
        assert split_time("") == (None, None)


# ─────────────────────────────────────────────────────────────
# Heuristic helpers
# ─────────────────────────────────────────────────────────────

class TestHeuristics:

    def test_is_likely_course_name(self):
        assert is_likely_course_name("STRUKTUR DATA") is True
        assert is_likely_course_name("AB") is False
        assert is_likely_course_name(None) is False
        assert is_likely_course_name("12345") is False

    def test_is_likely_class_code(self):
        assert is_likely_class_code("A") is True
        assert is_likely_class_code("B1") is True
        assert is_likely_class_code("STRUKTUR DATA") is False
        assert is_likely_class_code(None) is False


# ─────────────────────────────────────────────────────────────
# ScheduleExtractor class
# ─────────────────────────────────────────────────────────────

class TestScheduleExtractor:

    def test_default_grid(self):
        grid = ScheduleExtractor._build_default_grid()
        assert grid[1] == ("07:00", "07:50")
        assert grid[12] == ("17:10", "18:00")

    def test_friday_grid_afternoon_shift(self):
        grid = ScheduleExtractor._build_friday_grid()
        assert grid[7] == ("13:30", "14:20")  # shifted from 13:00

    def test_evening_slots(self):
        slots = ScheduleExtractor._build_evening_slots()
        assert 13 in slots
        assert 16 in slots
        assert slots[16] == ("21:30", "22:20")

    def test_end_time_cap_configurable(self):
        ext = ScheduleExtractor.__new__(ScheduleExtractor)
        ext.end_time_cap = "21:00"
        assert ext.end_time_cap == "21:00"

    def test_to_json_returns_empty_list_without_data(self):
        ext = ScheduleExtractor.__new__(ScheduleExtractor)
        ext.df = None
        assert ext.to_json() == []


# ─────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────

class TestValidator:

    @staticmethod
    def _make_row(**overrides):
        """Create a minimal valid schedule row with optional overrides."""
        base = {
            "course_name": "Test Course",
            "day": "SENIN",
            "start_time": "07:00",
            "end_time": "08:40",
            "sks": 2,
            "class_name": "A",
            "lecturer": "Dosen A",
            "room_id": "1.01",
            "session_id": "I",
            "time_slot": "07:00-08:40",
            "is_online": False,
            "semester": 1,
        }
        base.update(overrides)
        return base

    def test_returns_validation_result_type(self):
        data = [self._make_row()]
        result = validate_extraction(data)
        assert isinstance(result, ValidationResult)

    def test_pass_with_valid_data(self):
        data = [self._make_row()]
        result = validate_extraction(data)
        assert result.success is True
        assert "PASSED" in result.report

    def test_fail_on_missing_fields(self):
        data = [self._make_row(course_name="", day="")]
        result = validate_extraction(data)
        assert "Missing fields" in result.report

    def test_time_logic_error(self):
        data = [self._make_row(start_time="10:00", end_time="09:00")]
        result = validate_extraction(data)
        assert "Time error" in result.report

    def test_invalid_day(self):
        data = [self._make_row(day="FOOBAR")]
        result = validate_extraction(data)
        assert "Invalid day" in result.report

    def test_sks_out_of_range(self):
        data = [self._make_row(sks=0)]
        result = validate_extraction(data)
        assert "SKS" in result.report

    def test_room_overlap_detected(self):
        data = [
            self._make_row(course_name="Math", start_time="07:00", end_time="09:00", room_id="R1"),
            self._make_row(course_name="Physics", start_time="08:00", end_time="10:00", room_id="R1"),
        ]
        result = validate_extraction(data)
        assert "Room overlap" in result.report or "room" in result.report.lower()

    def test_online_room_not_conflict(self):
        """DARING rooms should not trigger room conflicts."""
        data = [
            self._make_row(course_name="A", room_id="DARING"),
            self._make_row(course_name="B", room_id="DARING"),
        ]
        result = validate_extraction(data)
        assert "Room overlap" not in result.report

    def test_lecturer_overlap_detected(self):
        data = [
            self._make_row(
                course_name="Math", lecturer="Dosen A",
                start_time="07:00", end_time="09:00", room_id="R1",
            ),
            self._make_row(
                course_name="Physics", lecturer="Dosen A",
                start_time="08:00", end_time="10:00", room_id="R2",
            ),
        ]
        result = validate_extraction(data)
        assert "Lecturer overlap" in result.report or "lecturer" in result.report.lower()

    def test_empty_data(self):
        result = validate_extraction([])
        assert result.success is False

    def test_none_data(self):
        result = validate_extraction(None)
        assert result.success is False
