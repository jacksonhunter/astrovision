"""Unit tests for HistoryTracker class."""

import pytest


class TestHistoryTracker:
    """Test HistoryTracker for processing history recording."""

    def test_record_step(self, history_tracker):
        """Test recording a processing step."""
        history_tracker.record('normalize', {'method': 'zscale'})

        history = history_tracker.get_history()

        assert len(history) == 1
        assert history[0]['operation'] == 'normalize'
        assert history[0]['parameters']['method'] == 'zscale'

    def test_record_multiple_steps(self, history_tracker):
        """Test recording multiple processing steps."""
        history_tracker.record('normalize', {'method': 'zscale'})
        history_tracker.record('stretch', {'method': 'asinh', 'a': 0.1})
        history_tracker.record('composite', {'algorithm': 'lupton'})

        history = history_tracker.get_history()

        assert len(history) == 3
        assert history[0]['operation'] == 'normalize'
        assert history[1]['operation'] == 'stretch'
        assert history[2]['operation'] == 'composite'

    def test_history_order(self, history_tracker):
        """Test that history maintains chronological order."""
        operations = ['load', 'normalize', 'stretch', 'enhance', 'composite', 'export']

        for op in operations:
            history_tracker.record(op, {})

        history = history_tracker.get_history()

        for i, op in enumerate(operations):
            assert history[i]['operation'] == op

    def test_export_to_fits_header(self, history_tracker):
        """Test exporting history to FITS header format."""
        history_tracker.record('normalize', {'method': 'zscale'})
        history_tracker.record('stretch', {'method': 'asinh'})

        fits_history = history_tracker.export_to_fits_header()

        assert isinstance(fits_history, list)
        assert len(fits_history) >= 2
        # FITS header entries are strings
        assert all(isinstance(entry, str) for entry in fits_history)

    def test_export_to_text(self, history_tracker):
        """Test exporting history as human-readable text."""
        history_tracker.record('normalize', {'method': 'zscale'})
        history_tracker.record('stretch', {'method': 'asinh', 'a': 0.1})

        text = history_tracker.export_to_text()

        assert isinstance(text, str)
        assert 'normalize' in text
        assert 'stretch' in text
        assert 'zscale' in text
        assert 'asinh' in text

    def test_clear_history(self, history_tracker):
        """Test clearing history."""
        history_tracker.record('step1', {})
        history_tracker.record('step2', {})

        assert len(history_tracker.get_history()) == 2

        history_tracker.clear()

        assert len(history_tracker.get_history()) == 0

    def test_timestamps(self, history_tracker):
        """Test that history entries have timestamps."""
        history_tracker.record('normalize', {'method': 'zscale'})

        history = history_tracker.get_history()

        # Should have timestamp
        assert 'timestamp' in history[0] or 'time' in history[0]

    def test_reproducibility_info(self, history_tracker):
        """Test that history contains enough info for reproducibility."""
        history_tracker.record('normalize', {
            'method': 'percentile',
            'vmin': 5,
            'vmax': 95
        })
        history_tracker.record('stretch', {
            'method': 'asinh',
            'a': 0.1
        })

        history = history_tracker.get_history()

        # Each step should have operation and parameters
        for step in history:
            assert 'operation' in step
            assert 'parameters' in step

    def test_nested_parameters(self, history_tracker):
        """Test recording nested parameter structures."""
        history_tracker.record('composite', {
            'algorithm': 'lupton',
            'parameters': {
                'stretch': 0.5,
                'Q': 8
            },
            'channels': {
                'red': 'i_band',
                'green': 'r_band',
                'blue': 'g_band'
            }
        })

        history = history_tracker.get_history()

        assert len(history) == 1
        assert history[0]['parameters']['algorithm'] == 'lupton'

    def test_full_pipeline_history(self, history_tracker):
        """Test recording a complete pipeline."""
        steps = [
            ('load', {'files': ['f1.fits', 'f2.fits', 'f3.fits']}),
            ('assess_quality', {'method': 'statistical'}),
            ('align', {'reference': 'f1.fits'}),
            ('normalize', {'method': 'zscale'}),
            ('stretch', {'method': 'asinh', 'a': 0.1}),
            ('map_channels', {'mapping': 'chromatic'}),
            ('composite', {'algorithm': 'lupton', 'stretch': 0.5}),
            ('export', {'format': 'png', 'file': 'output.png'})
        ]

        for op, params in steps:
            history_tracker.record(op, params)

        history = history_tracker.get_history()

        assert len(history) == len(steps)
        assert all(history[i]['operation'] == steps[i][0] for i in range(len(steps)))
