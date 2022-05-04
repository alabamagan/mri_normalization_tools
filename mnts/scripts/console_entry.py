import argparse
import os
from ..mnts_logger import MNTSLogger

class MNTS_ConsoleEntry(argparse.ArgumentParser):
    def __init__(self, addargs: str = ''):
        super(MNTS_ConsoleEntry, self).__init__()

        default_arguments = {
            'i': (['-i', '--input'],    {'type': str, 'help': 'Input directory that contains nii.gz or DICOM files.'}),
            'o': (['-o', '--output'],   {'type': str, 'help': 'Directory for generated output.'}),
            'O': (['-o', '--outfile'],  {'type': str, 'help': 'Directory for generated file.'}),
            'g': (['-g', '--idglobber'],{'type': str, 'help': 'Globber for globbing case IDs.', 'default': None}),
            'L': (['-l', '--idlist'],   {'type': str, 'help': 'List or txt file directory for loading only specific ids.', 'default': None}),
            'n': (['-n', '--numworker'],{'type': int, 'help': 'Specify number of workers.', 'default': 10}),
            'v': (['-v', '--verbose'],  {'action': 'store_true', 'help': 'Verbosity.'}),
            'l': (['-l', '--save-log'], {'action': 'store', 'help': 'Save log file to designated path', 'default': "./default.log"})
        }

        for k in addargs:
            args, kwargs = default_arguments[k]
            self.add_argument(*args, **kwargs)



    @staticmethod
    def make_console_entry_io():
        return MNTS_ConsoleEntry('iogLv')


    def parse_args(self, *args, **kwargs):
        a = super(MNTS_ConsoleEntry, self).parse_args(*args, **kwargs)

        # Create output dir
        if hasattr(a, 'output'):
            if not os.path.isdir(a.output):
                os.makedirs(a.output, exist_ok=True)
        return a