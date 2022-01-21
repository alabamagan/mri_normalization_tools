import logging
import traceback
import os, sys, traceback
import hashlib
import tempfile
from pathlib import Path
from tqdm import *

__all__ = ['MNTSLogger', 'LogExceptions']

class MNTSLogger(object):
    global_logger = None
    all_loggers = {}
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    log_level = os.getenv("MNT_LOGGER_LEVEL", default='info')

    def __init__(self, log_dir, logger_name=__name__, verbose=False, log_level=log_level, keep_file=True):
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
        self._log_dir = log_dir
        self._verbose = verbose
        self._warning_hash = {}
        self._keep_file = keep_file
        self._loggerkey = logger_name

        log_levels={
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR
        }
        assert log_level in log_levels, "Expected argument log_level in one of {}, got {} instead.".format(
            list(log_levels.keys()), log_level
        )

        # Check and create directory for log
        try:
            os.makedirs(os.path.dirname(log_dir), exist_ok=True)
        except:
            pass

        if MNTSLogger.global_logger is None:
            # prevent running this twice
            self.__enter__()

        self._logger = logging.getLogger(logger_name)
        formatter = LevelFormatter(fmt="[%(asctime)-12s-%(levelname)s] (%(name)s) %(message)s")

        handler = logging.FileHandler(self._log_dir)
        handler.setFormatter(formatter)

        self._stream_handler = TqdmLoggingHandler(verbose=verbose)
        self._stream_handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.addHandler(self._stream_handler)
        self._logger.setLevel(level=log_levels[log_level] if MNTSLogger.global_logger is None else
                                                            MNTSLogger.global_logger._logger.level)

        self.info("Loging to file at: {}".format(os.path.abspath(self._log_dir)))

        # First logger created is the global logger.
        if MNTSLogger.global_logger is None:
            MNTSLogger.global_logger = self
            print = self.info
            sys.excepthook= self.exception_hook
            self.info("Exception hooked to this logger.")

        # put this in all_logger
        MNTSLogger.all_loggers[logger_name] = self

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
            self._log_file = open(self._log_dir, 'w')
            self._log_dir = self._log_file.name
        # Make sure its absolute
        self._log_dir = os.path.abspath(self._log_dir)
        return self

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

    def exception(self, msg=""):
        self._logger.exception(msg)


    def exception_hook(self, *args):
        self.error('Uncaught exception:')
        self._logger.exception(args[-1], exc_info=args)

    def __class_getitem__(cls, item):
        if cls.global_logger is None:
            cls.global_logger = MNTSLogger('./default.log', logger_name='default', verbose=True, keep_file=False)
            return MNTSLogger[item]

        elif not item in cls.all_loggers:
            cls.global_logger.log_print("Requesting logger [{}] not exist, creating...".format(
                str(item)
            ))
            cls.all_loggers[item] = MNTSLogger(cls.global_logger._log_dir,
                                           logger_name=str(item),
                                           verbose=cls.global_logger._verbose)
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
        g_l = cls.global_logger
        for l in list(cls.all_loggers.keys()):
            g_l.info(f"Deleting logger: {l}")
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
        if (self == MNTSLogger.global_logger) or (len(MNTSLogger.all_loggers) == 1):
            self._logger.info("Deleting self...")
            self._logger.info("Removing log file...")
            MNTSLogger.all_loggers.pop(self._loggerkey)
            self._log_file.close()
            del self._logger, self._log_file
        else:
            self._logger.info("Deleting self...")
            MNTSLogger.all_loggers.pop(self._loggerkey)
            del self._logger

    def __del__(self):
        if self == MNTSLogger.global_logger:
            MNTSLogger.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self == MNTSLogger.global_logger:
            self._log_file.close()

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