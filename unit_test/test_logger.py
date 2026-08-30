import unittest
from mnts.mnts_logger import MNTSLogger
import logging
import pytest
from pathlib import Path
import os
import sys
import io
import multiprocessing
from unittest.mock import patch


# Module-level worker function — must be picklable for multiprocessing spawn
def _mpi_worker_create_loggers(worker_id):
    """Create a logger and a sub-logger inside the worker process."""
    try:
        logger = MNTSLogger[f'worker_{worker_id}']
        logger.info(f"Worker {worker_id} started")
        sub = MNTSLogger[f'worker_{worker_id}_sub']
        sub.info(f"Worker {worker_id} sub-logger ok")
        return True
    except Exception:
        return False

class Test_MNTSLogger(unittest.TestCase):
    def test_creating_logger_with_default_arguments(self):
        MNTSLogger.cleanup()
        logger = MNTSLogger()
        self.assertTrue(logger._verbose)
        self.assertEqual('info', logger._log_level)
        self.assertEqual(False, logger._keep_file)

    #  Tests that a logger can be created with custom arguments. Tags: [happy path]
    def test_creating_logger_with_custom_arguments(self):
        MNTSLogger.cleanup()
        logger = MNTSLogger(log_dir='test.log', logger_name='test', verbose=False, log_level='debug', keep_file=True)
        self.addCleanup(logger.cleanup)
        assert logger._logger_name == 'test'
        assert logger._log_dir == str(Path('test.log').absolute())
        assert logger._verbose == False
        assert logger._log_level == 'debug'
        assert logger._keep_file == True

    #  Tests that a warning message is not repeated when no_repeat=True. Tags: [general behavior]
    def test_logging_warning_message_with_no_repeat(self):
        logger = MNTSLogger()
        logger.warning('Test warning message', no_repeat=True)
        logger.warning('Test warning message', no_repeat=True)
        assert len(logger._warning_hash) == 1

    #  Tests that the log level can be set. Tags: [happy path]
    def test_setting_log_level(self):
        logger = MNTSLogger()

        # 模擬 stdout 和 stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        def clear_buffer():
            stdout_buffer.truncate(0)
            stdout_buffer.seek(0)
            stderr_buffer.truncate(0)
            stderr_buffer.seek(0)

        with patch('sys.stdout', stdout_buffer), patch('sys.stderr', stderr_buffer):
            MNTSLogger.set_global_log_level('warning')

            self.assertEqual(logger._logger.level, logging.WARNING)

            # Clear messages from changing log level
            clear_buffer()

            # This should not print anything
            logger.info("This shouldn't be seen")
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()
            clear_buffer()

            # Assert nothing was printed
            self.assertEqual(stdout_content, "")
            self.assertEqual(stderr_content, "")

            # This should print to stderr
            logger.warning("Warning is here")
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()

            # Assert something was printed
            self.assertIn("Warning is here", stderr_content)
            self.assertEqual(stdout_content, "")

    def test_creating_multiple_loggers_with_different_names(self):
        r"""Tests that multiple loggers can be created with different names. Tags: [happy path]"""
        logger1 = MNTSLogger(logger_name='logger1')
        logger2 = MNTSLogger(logger_name='logger2')
        assert logger1._logger_name == 'logger1'
        assert logger2._logger_name == 'logger2'

    def test_temp_log_file(self):
        r"""Test that when keep_file option is set to False or is default, the log_file attribute is created by
        tempfile."""
        with MNTSLogger(verbose=True, keep_file=False) as logger:
            logger.info("Logger ready")
            self.assertIsNone(logger._log_file)

    def test_keep_log_file(self):
        MNTSLogger.cleanup()
        with MNTSLogger('test.log', 'test_keep_log', verbose=True, keep_file=True) as logger:
            logger.info("Logger keep_file ready")
            self.assertIsNotNone(logger._log_file)
            p = Path(logger._log_dir)

            # See how error is logged
            with self.assertRaises(ZeroDivisionError):
                try:
                    10 / 0
                except Exception as e:
                    logger.exception(e)
                    raise e
        self.assertTrue(p.exists())
        p.unlink()

    def test_rich_traceback(self):
        logger = MNTSLogger(verbose=True, log_level='DEBUG')

        def not_good(x):
            x *= 10
            return x / 0

        try:
            not_good(20)
        except Exception as e:
            logger.exception(e)

    def test_rich_inspect(self):
        logger = MNTSLogger(verbose=True, log_level='DEBUG')

        obj = ("Abc", "DFE", [1, 2, 3])
        logger.inspect(obj)

class TestMNTSLoggerMPI(unittest.TestCase):
    """Tests covering multiprocessing / MPI logger interactions."""

    def setUp(self):
        MNTSLogger.cleanup()

    def tearDown(self):
        MNTSLogger.cleanup()

    def test_worker_loggers_independent_of_parent(self):
        """Worker subprocesses create their own global logger and do not
        interfere with the parent's lifecycle."""
        with MNTSLogger('./', verbose=False, keep_file=False) as logger:
            logger.info("Parent logger active")
            with multiprocessing.Pool(processes=2) as pool:
                results = pool.map(_mpi_worker_create_loggers, range(4))

        self.assertTrue(all(results), "One or more workers failed to create loggers")
        # Parent's with block should have reset state to 'Init'
        self.assertEqual(MNTSLogger.global_logger, 'Init',
                         "global_logger should be 'Init' after with block exits")
        self.assertEqual(len(MNTSLogger.all_loggers), 0,
                         "all_loggers should be empty after cleanup")

    def test_consecutive_with_blocks_are_independent(self):
        """Each with block gets a fresh global; settings must not bleed between blocks."""
        with MNTSLogger('./', verbose=False, keep_file=False) as logger1:
            logger1.info("First block")
            MNTSLogger['first_sub'].info("sub in first block")
            self.assertIsNone(logger1._log_file)

        self.assertEqual(MNTSLogger.global_logger, 'Init')
        self.assertEqual(len(MNTSLogger.all_loggers), 0)

        # Second block must not inherit the first block's sub-loggers or settings
        with MNTSLogger('./', verbose=False, keep_file=False) as logger2:
            logger2.info("Second block — must be fresh")
            self.assertIsNone(logger2._log_file)
            self.assertNotIn('first_sub', MNTSLogger.all_loggers)

        self.assertEqual(MNTSLogger.global_logger, 'Init')

    def test_sublogs_cleaned_up_on_exit(self):
        """Sub-loggers created inside a with block are destroyed when the block exits."""
        with MNTSLogger('./', verbose=False, keep_file=False) as logger:
            MNTSLogger['cleanup_sub_a'].info("a")
            MNTSLogger['cleanup_sub_b'].info("b")
            self.assertIn('cleanup_sub_a', MNTSLogger.all_loggers)

        self.assertEqual(MNTSLogger.global_logger, 'Init')
        self.assertEqual(len(MNTSLogger.all_loggers), 0)

    def test_worker_without_with_creates_own_global(self):
        """A worker that uses MNTSLogger[name] (no with) creates its own global
        that persists for the worker's lifetime and does not affect the parent."""
        with multiprocessing.Pool(processes=1) as pool:
            results = pool.map(_mpi_worker_create_loggers, [99])
        self.assertTrue(results[0])
        # Parent never touched global_logger — should still be 'Init'
        self.assertEqual(MNTSLogger.global_logger, 'Init')


class TestMNTSLoggerRichColor(unittest.TestCase):
    def setUp(self):
        self.logger_name = "test_logger"
        self.log_dir = "test.log"
        self.logger = MNTSLogger(log_dir=self.log_dir, logger_name='logger1', verbose=True, keep_file=False)
        self.logger2 = MNTSLogger(log_dir=self.log_dir, logger_name='logger2')

    def tearDown(self):
        MNTSLogger.cleanup()

    @unittest.skip("Not Implemented")
    def test_turn_off_rich_color(self):
        MNTSLogger.turn_off_rich_color()
        self.logger.info("Printing something without color")