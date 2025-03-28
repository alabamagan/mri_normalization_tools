import logging
import traceback
import os, sys, traceback
import hashlib
import tempfile
import atexit
from typing import Any, Optional, Tuple, Iterable
from pathlib import Path
from tqdm import *
from rich.text import Text
from rich._inspect import Inspect
from rich.logging import RichHandler
from rich.console import Console
import time
import shutil

global TORCH_EXIST
try:
    import torch.distributed as dist

    TORCH_EXIST = True
except ModuleNotFoundError:
    TORCH_EXIST = False

__all__ = ['MNTSLogger', 'LogExceptions']


class MNTSLogger(object):
    r"""Logger class that manages multiple logging instances.

    This class handles the creation and management of loggers with various
    configurations, such as log directory, log levels, and verbosity. It also
    hooks into the system's exception handling to log uncaught exceptions.

    Args:
        log_dir (str    , optional): The directory or file path for logging output. Defaults to 'default.log'.
        logger_name (str, optional): The name of the logger. Defaults to the module name.
        verbose (bool   , optional): Indicates if verbose logging is enabled. Defaults to True.
        log_level (str  , optional): The logging level to set. Defaults to class-level log_level.
        keep_file (bool , optional): If True

    Attributes:
        CRITICAL (int)                    : Logging level for critical messages.
        DEBUG (int)                       : Logging level for debug messages.
        ERROR (int)                       : Logging level for error messages.
        FATAL (int)                       : Logging level for fatal messages.
        INFO (int)                        : Logging level for informational messages.
        WARNING (int)                     : Logging level for warning messages.
        all_loggers (dict)                : A dictionary storing all created loggers by name.
        global_logger (MNTSLogger or str) : The singleton global logger or 'Init' if not yet created.
        is_verbose (bool)                 : Indicates if verbose logging is enabled.
        log_format (str)                  : Format string for log messages.
        log_level (str)                   : Default logging level obtained from environment variable or set to 'info'.
        log_levels (dict)                 : A mapping of log level names to logging module constants.

    """
    global_logger = 'Init'
    all_loggers = {}
    log_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR
    }
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    is_verbose = False
    log_level = os.getenv("MNT_LOGGER_LEVEL", default='info')
    log_format = "[%(asctime)-12s-%(levelname)s] (%(name)s) %(message)s"
    log_format_rich = "(%(name)s) %(message)s"
    use_rich_stream: bool = True
    use_rich_file: bool = False
    shared_handlers = {}

    def __new__(cls, log_dir='default.log', logger_name=__name__, verbose=True, log_level=log_level, keep_file=False):
        r"""Creates or retrieves a logger instance.

        Initializes and returns a logger instance, creating a global logger
        if none exists. Hooks the exception handler to the global logger.

        ..notes::
            The first instantiated logger becomes the global logger. If a logger
            with the same name exists, it will be reused instead of creating a new one.

        Args:
            log_dir (str    , optional): The directory or file path for logging output. Defaults to 'default.log'.
            logger_name (str, optional): The name of the logger. Defaults to the module name.
            verbose (bool   , optional): Indicates if verbose logging is enabled. Defaults to True.
            log_level (str  , optional): The logging level to set. Defaults to class-level log_level.
            keep_file (bool , optional): If True, retains the log file on new logger creation. Defaults to False.

        Returns:
            MNTSLogger: An instance of MNTSLogger configured with the specified parameters.

        """
        if cls.global_logger == 'Init':
            logger_name = 'global'
            cls.global_logger = None
            cls.global_logger = MNTSLogger(log_dir=log_dir,
                                           logger_name=logger_name,
                                           verbose=verbose,
                                           log_level=log_level,
                                           keep_file=keep_file)
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
            log_dir (str    , optional): The directory or file path for logging output. Defaults to 'default.log'.
            logger_name (str, optional): The name of the logger. Defaults to the module name.
            verbose (bool   , optional): Indicates if verbose logging is enabled. Defaults to True.
            log_level (str  , optional): The logging level to set. Defaults to class-level log_level.
            keep_file (bool , optional): If True, retains the log file on new logger creation. Defaults to False.


        Returns:
            :class:`Logger` object
        """
        global TORCH_EXIST

        super(MNTSLogger, self).__init__()
        # before anything, check if the logger is created within a subprocess initiated by torch
        if TORCH_EXIST:
            if dist.is_initialized():
                logger_name = logger_name + f"-DDP-{dist.get_rank():02d}"
        self._logger_name = logger_name

        # Do nothing if logger already created because it would have been initialized previuosly
        if logger_name in MNTSLogger.all_loggers:
            self.warning(f"Trying to duplicate logger {logger_name}")
            return

        # Define attributes
        self._log_file = None
        self._log_dir = str(Path(log_dir).absolute())
        self._verbose = verbose
        self._warning_hash = {}
        self._keep_file = keep_file
        self._logger_name = logger_name
        self._log_level = str(log_level).lower()

        assert self._log_level in self.log_levels, "Expected argument log_level in one of {}, got {} instead.".format(
            list(self.log_levels.keys()), log_level
        )

        # Check and create directory for log
        try:
            if not Path(log_dir).parent.is_dir():
                Path(log_dir).parent.mkdir()
        except:
            pass

        # if not keep_log use a temp file to hold the messages
        self.set_up_log_file()

        formatter = LevelFormatter(fmt=MNTSLogger.log_format)

        # if there's already a global logger, use it's handlers
        if MNTSLogger.global_logger is not None:
            for h in MNTSLogger.global_logger._logger.handlers:
                self._logger.addHandler(h)
        else:
            # otherwise create handlers and mark then as shared handlers
            if self._keep_file:
                if not 'file_handler' in MNTSLogger.shared_handlers:
                    file_handler = self._create_handler(self._logger, formatter, is_file_handler=True)
                    MNTSLogger.shared_handlers['file_handler'] = file_handler
                self._logger.addHandler(MNTSLogger.shared_handlers['file_handler'])

            # create a new logger if requested logger was not already created
            if not logger_name in MNTSLogger.all_loggers:
                # Logger must have a stream handler
                if not 'stream_handler' in MNTSLogger.shared_handlers:
                    stream_handler = self._create_handler(self._logger, formatter, is_file_handler=False)
                    MNTSLogger.shared_handlers['stream_handler'] = stream_handler
                self._logger.addHandler(MNTSLogger.shared_handlers['stream_handler'])

            else:
                msg = f"Duplicated MNTSLogger created with logger name: {self._logger_name}."
                raise ArithmeticError(msg)

        # Set logger level
        self._logger.setLevel(
            level=self.log_levels[self._log_level]
            if MNTSLogger.global_logger is None
            else MNTSLogger.global_logger._logger.level
        )

        # Put this logger in all_logger
        MNTSLogger.all_loggers[logger_name] = self

        # Messages
        if self._log_file is not None:
            self.info(f"Logging to {self._log_dir} with log level: {self._logger.level}")
        else:
            self.info(f"Logging to STDERR with log level: {self._logger.level}")

    def _create_handler(self, logger, formatter, is_file_handler=False):
        """
        Add appropriate handlers (file or stream) to the logger.

        Args:
            logger (logging.Logger): The logger instance to which handlers will be added.
            formatter (logging.Formatter): The formatter to be used by the handlers.
            is_file_handler (bool): If True, add file handlers. Otherwise, add stream handlers.
        """
        if is_file_handler and self._keep_file:
            assert self._log_file is not None
            # This "console" writes to the log file
            console = Console(
                color_system="truecolor",
                soft_wrap=True,
                width=max(shutil.get_terminal_size().columns, 160),
                file=self._log_file,
                highlight=self.use_rich_file
            )
            try:
                rich_handler = RichHandler(
                    console=console,
                    rich_tracebacks=True,
                    show_path=False,
                    show_time=True,
                    show_level=True,
                    markup=False,
                    tracebacks_show_locals=True,
                    log_time_format="[%Y-%m-%d %H:%M:%S]",
                    omit_repeated_times=False,
                )
                rich_formatter = LevelFormatter(fmt=MNTSLogger.log_format_rich)
                rich_handler.setFormatter(rich_formatter)
                return rich_handler
            except:
                handler = logging.FileHandler(self._log_file)
                handler.setFormatter(formatter)
                return handler
        else:
            if self.use_rich_stream:
                console = Console(
                    color_system="truecolor",
                    soft_wrap=True,
                    width=max(shutil.get_terminal_size().columns, 160),
                    stderr=True,  # Direct to STDERR
                )
                rich_handler = RichHandler(
                    console=console,
                    rich_tracebacks=True,
                    show_path=False,
                    show_time=True,
                    show_level=True,
                    markup=False,
                    tracebacks_show_locals=True,
                    log_time_format="[%Y-%m-%d %H:%M:%S]-R",
                    locals_max_length=10,
                    locals_max_string=80,
                )
                rich_formatter = LevelFormatter(fmt=MNTSLogger.log_format_rich)
                rich_handler.setFormatter(rich_formatter)
                return rich_handler
            else:
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                return stream_handler

    def set_up_log_file(self):
        r"""Changes the self._log_dir according to options"""
        if MNTSLogger.global_logger is not None:
            # Update local attributes with global logger's attribute
            original_log_file = self._log_file
            original_log_dir = self._log_dir
            original_keep_file = self._keep_file
            self._log_file = MNTSLogger.global_logger._log_file
            self._log_dir = MNTSLogger.global_logger._log_dir
            self._keep_file = MNTSLogger.global_logger._keep_file

            if original_log_file != self._log_file:
                self.warning(f"Log file changed from {original_log_file} to {self._log_file}")
            if original_log_dir != self._log_dir:
                self.warning(f"Log directory changed from {original_log_dir} to {self._log_dir}")
            if original_keep_file != self._keep_file:
                self.warning(f"Keep file setting changed from {original_keep_file} to {self._keep_file}")
        elif self._keep_file:
            self._log_file = open(self._log_dir, 'a')
            self._log_dir = self._log_file.name
        # Make sure its absolute
        self._log_dir = str(Path(self._log_dir).absolute())
        return self

    def __enter__(self):
        return self

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
        else:
            # If self is just an ordinary logger
            if self._logger_name in MNTSLogger.all_loggers and self == MNTSLogger[self._logger_name]:
                MNTSLogger.all_loggers.pop(self._logger_name)
            self.debug("Deleting this logger...")

            # If the final logger left is global, delete it too
            if len(MNTSLogger.all_loggers) == 1:
                MNTSLogger.global_logger.__exit__(exc_type, exc_val, exc_tb)

        # Remove all handler from loggers
        try:
            self._logger.handlers.clear()
        except:
            pass

    @property
    def _logger(self):
        return logging.getLogger(self._logger_name)

    @property
    def handlers(self):
        self._logger.handlers

    @classmethod
    def set_global_log_level(cls, level):
        assert level in cls.log_levels, "Log levels available are: {}".format(','.join(cls.log_levels.keys()))
        cls.log_level = cls.log_levels[level]
        for l in cls.all_loggers:
            cls.all_loggers[l].set_log_level(level)

    def set_log_level(self, level):
        assert level in self.__class__.log_levels, \
            "Log levels available are: {}".format(','.join(self.__class__.log_levels.keys()))
        self.info("Setting log level {} -> {}".format(self._log_level, level))
        self.log_levels = level
        self._logger.setLevel(self.__class__.log_levels[level])

    @classmethod
    def set_global_verbosity(cls, b):
        cls.is_verbose = b
        for loggers in cls.all_loggers:
            cls[loggers].set_verbose(b)

        try:
            cls.global_logger.set_verbose(False)
        except AttributeError:
            pass

    @classmethod
    def get_global_verbosity(cls):
        return cls.is_verbose

    @classmethod
    def set_global_logger(cls, logger: 'MNTSLogger'):
        assert isinstance(logger, MNTSLogger)
        cls.global_logger = logger
        sys.excepthook = logger.exception_hook

    @classmethod
    def set_verbose(cls, b):
        if 'stream_handler' in cls.shared_handlers:
            stream_handler = cls.shared_handlers['stream_handler']
            if hasattr(stream_handler, 'verbose'):
                stream_handler.verbose = b
        cls._verbose = b

    def log_traceback(self):
        self.exception()

    def log_print(self, msg, level=logging.INFO):
        self._logger.log(level, msg)

    def log_print_tqdm(self, msg, level=logging.INFO):
        self._logger.log(level, msg)

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
            try:
                self.__class__.global_logger.sys_hook(*args)
            except AttributeError:
                pass
            self.__del__()

    def __class_getitem__(cls, item):
        # If global logger haven't be created, casually use temp file to host the log
        if cls.global_logger == 'Init' or cls.global_logger is None:
            cls.global_logger = MNTSLogger('./default.log', logger_name='default',
                                           verbose=True, keep_file=False)
            return MNTSLogger[item]

        elif not item in cls.all_loggers:
            if cls.global_logger is None:
                cls.global_logger = MNTSLogger('./default.log', logger_name='default',
                                               verbose=True, keep_file=False)
                return MNTSLogger[item]

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
        global TORCH_EXIST
        if TORCH_EXIST:
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

    def inspect(self, obj: Any, *args, **kwargs):
        r"""If a rich handler is used for logging, this inspect function will write the inspect result to the log file
        and/or console output.
        """
        default_kwargs = {
            'private': False,
            'title': f"Inspection ({type(obj)})",
            'all': False
        }
        default_kwargs.update(kwargs)

        # Get the rich handler from self._logger
        rich_handler = next((h for h in self._logger.handlers if isinstance(h, RichHandler)), None)

        if rich_handler:
            # Use the rich console to print the inspected object
            console = rich_handler.console
            console.print(Inspect(obj, *args, **default_kwargs))  # Print the object using the rich console
        else:
            # Fallback to default logging if no rich handler is found
            self.warning(f"Try to inspect object but no RichHanlder is found.", no_repeat=True)

    def _del(self):
        try:
            if not hasattr(MNTSLogger, 'global_logger'):
                return

            # If this is the global logger or this is the final logger
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
        msg = f"This logger: \n\t{self._logger_name} ({self._log_level})\n" + \
              f"All loggers: \n" + "\n".join(["\t- " + l for l in self.all_loggers]) + '\n' + \
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