import pprint
import re, os
import pandas as pd
from typing import Optional, AnyStr, List, Union, Dict, Any, Tuple
from pathlib import Path
from ..mnts_logger import MNTSLogger

__all__ = ['get_unique_IDs', 'get_fnames_by_globber', 'get_fnames_by_IDs', 'load_supervised_pair_by_IDs',
           'check_ID_duplicates']


def get_unique_IDs(fnames: List[Union[Path,str]],
                   globber: Optional[str]=None,
                   return_dict: Optional[Dict]=False):
    iddict = {}
    for fname in fnames:
        if globber is None:
            globber = "([0-9]{3,5})"

        if isinstance(fname, Path):
            f = fname.name
        else:
            f = fname

        mo = re.search(globber, str(f))
        if not mo is None:
            id = mo.group()
            if id in iddict.keys():
                iddict[id].append(fname)
            else:
                iddict[id] = [fname]

    if not return_dict:
        idlist =list(iddict.keys())
        idlist.sort()
        return idlist
    else:
        return iddict

def get_fnames_by_IDs(fnames: List[Union[Path, str]],
                      idlist: List[str],
                      globber: Optional[str]=None,
                      return_dict: Optional[bool]=False,
                      raise_error: Optional[bool]=False) -> Union[Dict[str, Any], List[Any]]:
    """Matches filenames with a list of IDs using a regex pattern.

    Given a list of filenames and a list of IDs, this function uses a regular
    expression pattern to extract unique IDs from the filenames. It then checks
    for any missing IDs or duplicate filenames associated with the same ID.

    Args:
        fnames (List[Union[Path, str]]):
            A list of filenames or Path objects to be matched with IDs.
        idlist (List[str]):
            A list of string IDs to be matched with the filenames.
        globber (Optional[str]):
            A regular expression pattern used to match and extract IDs from the filenames.
            Defaults to "([0-9]{3,5})", which matches 3 to 5 consecutive digits.
        return_dict (Optional[bool]):
            If True, returns a dictionary mapping IDs to filenames. If False,
            returns a list of filenames that match the IDs in `idlist`.
            Defaults to False.
        raise_error (Optional[bool]):
            If True, raises a ValueError if any ID in `idlist` cannot be found
            in `fnames`. If False, logs a warning. Defaults to False.

    Returns:
        Union[Dict[str, Any], List[Any]]:
            If `return_dict` is True, returns a dict mapping IDs to the corresponding
            filenames. If `return_dict` is False, returns a list of filenames that
            correspond to the IDs in `idlist`.

    Raises:
        ValueError:
            If `raise_error` is True and any ID in `idlist` cannot be found in `fnames`.

    Examples:
        >>> fnames = ['123_file.txt', '124_file.txt', '125_file.txt']
        >>> idlist = ['123', '125']
        >>> get_fnames_by_IDs(fnames, idlist)
        >>> # ['123_file.txt', '125_file.txt']
        >>> get_fnames_by_IDs(fnames, idlist, return_dict=True)
        >>> # {'123': ['123_file.txt'], '125': ['125_file.txt']}


    """
    _logger = MNTSLogger['algorithm.utils']
    globber = globber or "([0-9]{3,5})"

    outfnames = {}
    id_fn_pair = get_unique_IDs(fnames, globber, return_dict=True)
    ids_in_fn = id_fn_pair.keys()

    id_not_in_fnames = set(idlist) - set(ids_in_fn)
    if len(id_not_in_fnames) > 0:
        msg = (f"Cannot find anything for the following ids:\n "
               f"{pprint.pformat(id_not_in_fnames)}")
        if not raise_error:
            _logger.warning(msg)
        else:
            raise ValueError(msg)
    overlap = set(ids_in_fn) & set(idlist)

    # Check if there are repeated ids
    for k, v in id_fn_pair.items():
        if not k in overlap:
            continue
        if len(v) > 1:
            _logger.warning(f"Found more than 1 file for ID: {k}. "
                            f"Files found are: {v}")

    if return_dict:
        return {k: id_fn_pair[k] for k in list(overlap)}
    else:
        overlap_files = [id_fn_pair[key][0] for key in overlap]
        return overlap_files


def get_fnames_by_globber(fnames, globber):
    assert isinstance(fnames, list)

    copy = list(fnames)
    for f in fnames:
        if re.match(globber, f) is None:
            copy.remove(f)
    return copy


def load_supervised_pair_by_IDs(source_dir: Path,
                                target_dir: Path,
                                idlist: List[str],
                                globber: Optional[str]=None,
                                return_pairs: Optional[bool]=False) -> Union[List[Path], Tuple[List[Path], List[Path]]]:
    """Loads and pairs supervised file names from source and target directories by IDs.

    This function pairs file names from the given source and target directories,
    matching them according to the provided list of IDs. It can optionally use
    a globbing pattern to filter file names.

    Args:
        source_dir (Path):
            The directory containing the source files.
        target_dir (Path):
            The directory containing the target files.
        idlist (List[str]):
            A list of IDs to match files to.
        globber (Optional[str]):
            A regex pattern to filter file names based on IDs. Defaults to None,
            which means no filtering is applied.
        return_pairs (Optional[bool]):
            If True, returns a list of (source_file, target_file) pairs. If False,
            returns two lists of file names from source and target respectively.
            Defaults to False.

    Returns:
        Tuple[Any, Any]:
            If `return_pairs` is True, returns a list of (source_file, target_file) pairs.
            If `return_pairs` is False, returns two lists of file names from source and
            target directories respectively.

    Raises:
        ValueError:
            If there is a mismatch in the number of files after pairing.

    .. note::
        This function requires `get_fnames_by_IDs`, `get_fnames_by_globber`,
        and `MNTSLogger` to be defined elsewhere in the codebase.
    """
    _logger = MNTSLogger['algorithm.utils']

    # List files in source and target directories
    source_files = list(source_dir.iterdir())
    target_files = list(target_dir.iterdir())

    # Match files by IDs
    source_list = get_fnames_by_IDs(source_files, idlist, globber=globber, return_dict=True)
    target_list = get_fnames_by_IDs(target_files, idlist, globber=globber, return_dict=True)

    # Ensure matched files are present in both source and target
    source_keys = set(source_list.keys())
    target_keys = set(target_list.keys())
    common_keys = list(source_keys & target_keys)
    common_keys.sort()

    if len(common_keys) != len(idlist):
        _logger.error("Dimension mismatch when pairing.")
        missing = {'Src': list(source_keys - target_keys), 'Target': list(target_keys - source_keys)}
        _logger.debug(f"{missing}")
        raise ValueError(f"Dimension mismatch! Src: {len(source_keys)} vs Target: {len(target_keys)}")

    # Pair files from source and target
    #! Note that if there are multiple matches for the same ID, only the first file path is returned.
    paired_source_files = [source_list[key][0] for key in common_keys]
    paired_target_files = [target_list[key][0] for key in common_keys]

    # Sort pairs by IDs to ensure matching order
    pairs = list(zip(paired_source_files, paired_target_files))
    pairs.sort(key=lambda x: list(idlist).index(re.search(globber, Path(x[0]).stem).group()))

    # Return pairs or separate lists
    if return_pairs:
        return pairs
    else:
        source_list, target_list = zip(*pairs)
        return list(source_list), list(target_list)

def check_ID_duplicates(target_dir: Path,
                        globber: AnyStr = None) -> pd.DataFrame:
    r"""
    Check if there are any files with duplicated IDs in the file and output a dataframe
    Args:
        target_dir (Path):
            Files to check
        globber (str):
            Regex string to glob the ID.

    Returns:
        pd.DataFrame
    """
    _logger = MNTSLogger['algorithm.utils']
    target_dir = Path(target_dir)

    assert target_dir.is_dir(), "Cannot open target_dir."
    if len(list(target_dir.iterdir())) == 0:
        _logger.info(f"Nothing is in {str(target_dir.absolute())}")

    dup_keys = []
    ids = get_unique_IDs([str(f) for f in target_dir.iterdir()],
                         return_dict=True, globber=globber)
    for key in ids:
        if len(ids[key]) > 1:
            dup_keys.append(key)

    out_frame = []
    for key in dup_keys:
        row = pd.Series(ids[key], index=[f"Filename {i}" for i in range(len(ids[key]))], name=key)
        out_frame.append(row)

    if len(out_frame) == 0:
        _logger.info(f"Searched {len(target_dir.iterdir())}, no ID duplication found.")
        return None
    else:
        out_frame = pd.concat(out_frame, axis=1, sort=False)
        out_frame.fillna('-', inplace=True)
        return out_frame.T