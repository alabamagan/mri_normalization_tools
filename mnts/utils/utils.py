import pandas as pd
from pathlib import Path
import re
import warnings
from shutil import *
from pprint import pformat, pprint
from ..mnts_logger import MNTSLogger
from typing import *

__all__ = ['repeat_zip']

def repeat_zip(*args):
    r"""
    Zip with shorter columns repeated until the longest column is fully iterated.

    Args:
        *args (list of iterables)

    Examples:
        >>> x = [tuple([1]), ('a', 'b'), ('Z', 'D', 'E', 'F')]
        >>> for row in repeat_zip(*x):
        >>>    print(row)
        >>> #(1, 'a', 'Z')
        >>> #(1, 'b', 'D')
        >>> #(1, 'a', 'E')
        >>> #(1, 'b', 'F')
    """
    iterators = [iter(it) for it in args]
    finished = {i: False for i in range(len(iterators))}
    while True:
        values = []
        for i, it in enumerate(iterators):
            try:
                value = next(it)
            except StopIteration:
                finished[i] = True
                iterators[i] = iter(args[i])
                value = next(iterators[i])
            values.append(value)
            if all([x[1] for x in finished.items()]):
                return
        yield tuple(values)


def unify_modality_name(m: str) -> str:
    r"""Unifies the modality name based on predefined criteria.

    This function processes a given modality name and attempts to normalize
    it by matching against predefined regex patterns. It determines the
    sequence weight, scan plane, whether it includes contrast, and if it
    employs fat suppression. It then constructs a new modality name with
    this information.

    Args:
        m (str): The original modality name to be unified.

    Returns:
        str: The unified modality name that has been standardized according
        to the predefined criteria. The format of the unified name is
        constructed as follows: 'CE-' prefix if contrast is present,
        followed by the sequence weight (or 'MISC' if none is found),
        '-FS' if fat suppression is present, and ending with the scan
        plane if identified.

    Raises:
        ValueError: If `m` is not a string, a ValueError will be raised
        when regex matching is attempted.

    Examples:
        >>> unify_modality_name("t1_coronal+fs")
        'CE-T1W-FS_COR'

        >>> unify_modality_name("T2_sagittal")
        'T2W_SAG'

        >>> unify_modality_name("SomeRandomModalityName")
        'MISC'

    .. note::
        The function does not currently use the 'DIXON' regex pattern
        for modality unification, as explained in the function's inline
        comments. This is because DIXON is considered a fat suppression
        technique rather than a unique weighted sequence and is already
        included in the file name.
    """
    # error check
    if not isinstance(m, str):
        raise TypeError(f"Input should be string, got {type(m)} instead.")

    # Now this is hardcoded, maybe we can have a more flexible impelmentation
    weights = {
        'NECK': r"(?i)(.*NECK.*)",
        'T1W': r"(?i)(.*T1.*)",
        'T2W': r"(?i)(.*T2.*)",
        'DWI': r"(?i)(.*DWI.*)",
        'IVIM': r"(?i)(.*IVIM.*)",
        'DCE': r"(?i)(.*DCE.*)",
        'SURVEY': r"(?i)(.*SURVEY.*)",
        'eTHRIVE': r"(?i)(.*e-thrive.*)"
    }
    planes = {
        'COR': r"(?i)(.*cor.*)",
        'SAG': r"(?i)(.*sag.*)",
        'TRA': r"(?i)(.*(tra|ax).*)",
    }
    contrast = r"(?i).*(\+C|C\+).*"
    fat_sat = r"(?i).*fs.*"
    dixon = r"(?i).*dixon.*"    # not used

    """
    Notes: After due consideration, I decide not to add DIXON into directory sorting criteria because
    DIXON is already carried in the file name. The modality is also clinically used as FS technique 
    rather than a unique weighted sequence. 
    """

    w = None    # sequence weight
    p = None    # scan plane
    c = False   # contrast
    fs = False  # fat suppression
    for _w, regpat in weights.items():
        mo = re.match(regpat, m)
        if mo is not None:
            w = _w
            break

    for _p, regpat in planes.items():
        mo = re.match(regpat, m)
        if mo is not None:
            p = _p
            break

    c = re.match(contrast, m) is not None
    fs = re.match(fat_sat, m) is not None
    dx = re.match(dixon, m) is not None

    # reconstruct modality name to unify it
    if w is None:
        w = 'MISC'

    new_name = f"{'CE-' if c else ''}" + \
               f"{w}" + \
               (f"-FS" if fs else "")+ \
               (f"_{p}" if not p is None else "")
    return new_name


def organize_directory(d: Union[Path, str], warn_duplicate: bool = False):
    r"""Organizes a directory of NIfTI image files into subdirectories.

    This function takes a directory path as input and organizes NIfTI image
    files (.nii.gz) found in it by their metadata. It uses a predefined regex
    pattern to extract the patient ID, modality, and sequence ID from each
    file name. It then applies a unified modality name to each file, creates
    subdirectories for each modality, and moves files to their respective
    locations. Optionally, it can warn about duplicate entries for the same
    patient ID and modality combination.

    Args:
        d (Union[Path, str]):
            The directory path where the NIfTI image files
            are located. It can be a `Path` object or a string path.
        warn_duplicate (bool, Optional):
            If True, issues a warning when
            duplicate image files for the same patient ID and modality are
            found. Defaults to `False`.

    Raises:
        TypeError:
            If the provided `d` argument is neither a `Path` object
            nor a string.
        FileNotFoundError:
            If the provided path `d` does not exist or is not
            a directory.

    .. note::
        This function relies on the :func:`unify_modality_name` function to rename
        the modalities. The regex pattern used for extraction is defined
        within the function and can be adjusted based on the file naming
        conventions.

    Examples:
        Suppose we have a directory `./nii_images` with the following files:
        `patient1-T1+001_tra+C.nii.gz`, `patient1-T2+002_ax.nii.gz`.

        >>> organize_directory('./nii_images')
        This will organize the files into subdirectories named after their
        unified modalities, for instance: `./nii_images/CE-T1W-TRA` and
        `./nii_images/T2W-TRA`.

    """
    # Error check
    if not isinstance(d, (Path, str)):
        raise TypeError(f"Expect path or string input, got {type(d)} instead.")
    if not isinstance(d, Path):
        d = Path(d)
    if not d.is_dir():
        raise FileNotFoundError(f"Cannot found directory: {str(p.absolute())}")

    repat = r"(?P<PatientID>[\w\d]+)-(?P<Modality>[-\w\+\.\(\)]+)\+(?P<SequenceID>\d+).*"

    forganize_dir = d

    # Glob all nii files within `d`
    rows = []
    for _nii_file in forganize_dir.rglob("*nii.gz"):
        _fname = _nii_file.name
        mo = re.search(repat, _fname)
        groupdict = mo.groupdict()
        groupdict['fname'] = _fname
        groupdict['file_dir'] = _nii_file
        rows.append(pd.Series(groupdict))
    df = pd.concat(rows, axis=1).T
    df.set_index('fname', inplace=True, drop=True)

    # Map sequence name to unified modality name
    mapping = {}
    df['Unified Modality'] = df['Modality'].apply(unify_modality_name)

    # Sanity check if there's image of same patient having same unified modliaty name
    if warn_duplicate:
        df_new = df.reset_index().set_index(['PatientID', 'Modality'])
        duplicates = df_new.loc[df_new.index[df_new.index.duplicated()]]

        if len(duplicates):
            msg = "The following cases presents duplicates: \n"
            msg += pformat(duplicates.to_string())
            warnings.warn(msg)

    # iterate and put files where they belong
    for name, row in df.iterrows():
        src_dir = row['file_dir']
        mov_to = target_dir / row['Unified Modality']
        if not mov_to.is_dir():
            mov_to.mkdir(exist_ok=True, parents=True)

        # print(row)
        if not src_dir.parent == (mov_to / name):
            MNTSLogger['utils'].info(f"Moving {src_dir} -> {mov_to / name}")
            try:
                move(str(src_dir), mov_to)
            except Exception as e:
                warnings.warn(f"Error when trying to move: {name}; {e}")
