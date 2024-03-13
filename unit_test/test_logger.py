import unittest
from mnts.mnts_logger import MNTSLogger
import logging
import pytest
from pathlib import Path
import os


class Test_MNTSLogger(unittest.TestCase):
    def test_creating_logger_with_default_arguments(self):
        logger = MNTSLogger()
        self.assertTrue(logger._verbose)
        self.assertEqual(logger._log_level, 'info')
        self.assertEqual(logger._keep_file, False)

    #  Tests that a logger can be created with custom arguments. Tags: [happy path]
    def test_creating_logger_with_custom_arguments(self):
        logger = MNTSLogger(log_dir='test.log', logger_name='test', verbose=False, log_level='debug', keep_file=True)
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
        logger.set_global_log_level('debug')
        assert logger._logger.level == logging.DEBUG

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
            self.assertIsNotNone(logger._log_file)
