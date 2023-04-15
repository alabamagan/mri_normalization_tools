import logging
import traceback
import os, sys, traceback
import hashlib
import tempfile
import atexit
from pathlib import Path
from tqdm import *

import torch.distributed as dist

__all__ = ['MNTSLogger', 'LogExceptions']

class MNTSLogger(object):
    global_logger = 'Init'
    all_loggers   = {}
    log_levels={
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR
    }
    DEBUG         = logging.DEBUG
    INFO          = logging.INFO
    WARNING       = logging.WARNING
    CRITICAL      = logging.CRITICAL
    FATAL         = logging.FATAL
    ERROR         = logging.ERROR
    is_verbose    = False
    log_level = os.getenv("MNT_LOGGER_LEVEL", default='info')

    def __new__(cls, log_dir='default.log', logger_name=__name__, verbose=True, log_level=log_level, keep_file=False):
        if cls.global_logger is 'Init':
            logger_name = 'global'
            cls.global_logger = None
            cls.global_logger = MNTSLogger(log_dir = log_dir,
                                           logger_name = logger_name,
                                           verbose = verbose,
                                           log_level =log_level,
                                           keep_file = keep_file)
            cls.is_verbose = cls.global_logger._verbose
            cls.global_logger.sys_hook = sys.excepthook
            sys.excepthook = cls.global_logger.exception_hook
            cls.global_logger.info(f"Created first logger. Exception hooked to this logger. "
                                   f"Log level is: {log_level}")
            return super().__new__(cls)
        elif logger_name in cls.all_loggers and logger_name != 'global':
            return cls.all_loggers[logger_name]
        else:
            return super().__new__(cls)

    def __init__(self, log_dir='default.log', logger_name=__name__, verbose=True, log_level=log_level, keep_file=False):
        """
        This is the logger. This is typically passed to all modules for logging. Use class method Logger['str'] to get a
        logger named 'str'.

        Args:
            log_dir (str):
                Filename of the log file.
            verbose (boolean, Optional):
                If True, messages will be printed to stdout alongside being logged. Default to False.

        Returns:
            :class:`Logger` object
        """

        super(MNTSLogger, self).__init__()
        # before anything, check if the logger is created within a subprocess initiated by torch
        if dist.is_initialized():
            logger_name = logger_name + f"-DDP-{dist.get_rank():02d}"
        self._logger_name = logger_name

        # Do nothing if logger already created because it would have been initialized previuosly
        if logger_name in MNTSLogger.all_loggers:
            self.warning(f"Trying to duplicate logger {logger_name}")
            return

        # Define attributes
        self._log_dir = str(Path(log_dir).absolute())
        self._verbose = verbose
        self._warning_hash = {}
        self._keep_file = keep_file
        self._logger_name = logger_name
        self._log_level = log_level


        assert log_level in self.log_levels, "Expected argument log_level in one of {}, got {} instead.".format(
            list(self.log_levels.keys()), log_level
        )

        # Check and create directory for log
        try:
            if not Path(log_dir).parent.is_dir():
                Path(log_dir).parent.mkdir()
        except:
            pass

        # if not keep_log use a temp file to hold the messages
        self.__enter__()

        formatter = LevelFormatter(fmt="[%(asctime)-12s-%(levelname)s] (%(name)s) %(message)s")

        handler = logging.StreamHandler(self._log_file)
        handler.setFormatter(formatter)
        self._file_handler = handler

        # create a new logger if requested logger was not already created
        if not logger_name in MNTSLogger.all_loggers:
            self._stream_handler = TqdmLoggingHandler(verbose=self._verbose)
            self._stream_handler.setFormatter(formatter)
            self._logger.addHandler(self._file_handler)
            self._logger.addHandler(self._stream_handler)
            self._logger.setLevel(level=self.log_levels[self._log_level] if MNTSLogger.global_logger is None else
                                                                MNTSLogger.global_logger._logger.level)
            # put this in all_logger
            MNTSLogger.all_loggers[logger_name] = self
            self.info(f"Loging to {self._log_dir} with log level: {self._logger.level}")
        else:
            msg = f"Duplicated MNTSLogger created with logger name: {self._logger_name}."
            raise ArithmeticError(msg)

    def __enter__(self):
        # if not keep file, log to tempfile under the parent directory of log_dir
        if not self._keep_file and MNTSLogger.global_logger is None:
            temp_file = tempfile.NamedTemporaryFile('w', dir=str(Path(self._log_dir).parent), suffix='.log')
            self._log_dir = temp_file.name
            self._log_file = temp_file
        elif MNTSLogger.global_logger is not None:
            self._log_file = MNTSLogger.global_logger._log_file
            self._log_dir = MNTSLogger.global_logger._log_dir
        else:
            self._log_file = open(self._log_dir, 'a')
            self._log_dir = self._log_file.name
        # Make sure its absolute
        self._log_dir = str(Path(self._log_dir).absolute())
        return self

    @property
    def _logger(self):
        return logging.getLogger(self._logger_name)

    @classmethod
    def set_log_level(cls, level):
        assert level in cls.log_levels, "Log levels available are: {}".format(','.join(cls.log_levels.keys()))
        for l in cls.all_loggers:
            cls.all_loggers[l]._logger.level = cls.log_levels[level]

    @classmethod
    def set_global_verbosity(cls, b):
        cls.is_verbose = b
        for loggers in cls.all_loggers:
            cls[loggers].set_verbose(b)

        try:
            self.global_logger.set_verbose(False)
        except AttributeError:
            pass

    def set_verbose(self, b):
        self._stream_handler.verbose=b
        self._verbose=b

    def log_traceback(self):
        self.exception()

    def log_print(self, msg, level=logging.INFO):
        self._logger.log(level, msg)
        # if self._verbose:
        #     print(msg)

    def log_print_tqdm(self, msg, level=logging.INFO):
        self._logger.log(level, msg)
        # if self._verbose:
        #     tqdm.write(msg)

    def info(self, msg):
        self.log_print_tqdm(msg, level=logging.INFO)

    def debug(self, msg):
        self.log_print_tqdm(msg, level=logging.DEBUG)

    def warning(self, msg: str, no_repeat=False):
        if no_repeat:
            h = hashlib.md5(msg.encode('utf-8')).hexdigest()
            if not h in self._warning_hash:
                self.log_print_tqdm(msg, level=logging.WARNING)
                self.log_print_tqdm("Warning message won't be shown again in this run",
                                    level=logging.WARNING)
                self._warning_hash[h] = 1
        else:
            self.log_print_tqdm(msg, level=logging.WARNING)

    def error(self, msg):
        self.log_print_tqdm(msg, level=logging.ERROR)

    def critical(self, msg):
        self.log_print_tqdm(msg, level=logging.CRITICAL)

    def exception(self, msg="", exec=False):
        self._logger.exception(msg)
        # exec = traceback.format_exc()
        # self._logger.debug(sys.exc_info()[2])

    def exception_hook(self, *args):
        gettrace = getattr(sys, 'gettrace', None)
        if not gettrace():
            self.error('Uncaught exception:')
            self._logger.exception(args[-1], exc_info=args)
            self.__class__.global_logger.sys_hook(*args)
            self.__del__()

    def __class_getitem__(cls, item):
        # If global logger haven't be created, casually use temp file to host the log
        if cls.global_logger is 'Init' or None:
            cls.global_logger = MNTSLogger('./default.log', logger_name='default',
                                           verbose=True, keep_file=False)
            return MNTSLogger[item]

        elif not item in cls.all_loggers:
            cls.global_logger.info("Requesting logger [{}] not exist, creating...".format(
                str(item)
            ))
            cls.all_loggers[item] = MNTSLogger(cls.global_logger._log_dir,
                                           logger_name=str(item),
                                           verbose=cls.is_verbose)
            return cls.all_loggers[item]
        else:
            return cls.all_loggers[item]

    @staticmethod
    def Log_Print(msg, level=logging.INFO):
        MNTSLogger.global_logger.log_print(msg, level)

    @staticmethod
    def Log_Print_tqdm(msg, level=logging.INFO):
        MNTSLogger.global_logger.log_print_tqdm(msg, level)

    @staticmethod
    def get_global_logger():
        if not MNTSLogger.global_logger is None:
            # Attempts to create a global logger
            MNTSLogger.global_logger = MNTSLogger('./default_logdir.log', logger_name='default_logname')
            return MNTSLogger.global_logger
        else:
            raise AttributeError("Global logger was not created.")

    @classmethod
    def cleanup(cls):
        # in DDP subprocess, the global logger is controlled by rank 0 process only
        if dist.is_initialized():
            if dist.get_rank() != 0:
                return
        g_l = cls.global_logger
        if g_l is None or isinstance(g_l, str):
            return
        for l in list(cls.all_loggers.keys()):
            if cls.all_loggers[l] == g_l:
                continue
            else:
                cls.all_loggers[l]._del()
        g_l._del()

        # Clean list
        del cls.global_logger, cls.all_loggers
        cls.global_logger = None
        cls.all_loggers = {}

    def _del(self):
        try:
            if not hasattr(MNTSLogger, 'global_logger'):
                return

            if (self == MNTSLogger.global_logger) or (len(MNTSLogger.all_loggers) == 1):
                self._logger.info("Deleting self...")
                self._logger.info("Removing log file...")
                MNTSLogger.all_loggers.pop(self._logger_name)
                self._log_file.close()
                del self._logger, self._log_file
            else:
                self._logger.info("Deleting self...")
                MNTSLogger.all_loggers.pop(self._logger_name)
                del self._logger
        except Exception as e:
            # i give up, MPI logger is a nightmare, this class is not thread safe 
            pass

    def __str__(self):
        msg = f"This logger: \n\t{self._logger_name}\n" \
              f"All loggers: \n" + "\n".join(["\t- "+l for l in self.all_loggers]) + \
              f"Logfile: \nt\t{self._log_dir}"
        return msg

    def __repr__(self):
        msg = f"{self.__class__.__name__}: {self._logger_name}"
        return msg

    def __del__(self):
        try:
            if self == MNTSLogger.global_logger:
                MNTSLogger.cleanup()
        except:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self == MNTSLogger.global_logger:
            self.info("Deleting global logger...")
            MNTSLogger.global_logger = None
            # Pop first to prevent infinite loop
            MNTSLogger.all_loggers.pop(self._logger_name)
            for loggers in dict.copy(MNTSLogger.all_loggers):
                # exist all existing logs
                MNTSLogger.all_loggers[loggers].__exit__(exc_type, exc_val, exc_tb)
            MNTSLogger.all_loggers.clear()
            self._log_file.close() # close to delete tempfile
        else:
            # If self is just an ordinary logger
            if self._logger_name in MNTSLogger.all_loggers and self == MNTSLogger[self._logger_name]:
                MNTSLogger.all_loggers.pop(self._logger_name)
            self.debug("Deleting this logger...")

        # Remove all handler from loggers
        try:
            self._logger.removeHandler(self._file_handler)
            self._logger.removeHandler(self._stream_handler)
        except:
            pass

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)
        except AttributeError:
            pass
        except Exception as e:
            # Supposedly also catch multi-thread errors, but not always working
            Logger['GGWP-OLZ']._logger.error(traceback.format_exc())
            raise

        # It was fine, give a normal answer
        return result

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET, verbose=False):
        super().__init__(level)
        self.verbose = verbose

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.verbose:
                tqdm.write(msg)
                self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class LevelFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, level_fmts={}):
        self._level_formatters = {}
        for level, format in level_fmts.items():
            # Could optionally support level names too
            self._level_formatters[level] = logging.Formatter(fmt=format, datefmt=datefmt)
        # self._fmt will be the default format
        super(LevelFormatter, self).__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        if record.levelno in self._level_formatters:
            return self._level_formatters[record.levelno].format(record)

        return super(LevelFormatter, self).format(record)


atexit.register(MNTSLogger.cleanup)