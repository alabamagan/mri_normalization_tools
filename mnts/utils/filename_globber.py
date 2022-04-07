import pprint
import re, os
import pandas as pd
from typing import Optional, AnyStr
from pathlib import Path
from ..mnts_logger import MNTSLogger

__all__ = ['get_unique_IDs', 'get_fnames_by_globber', 'get_fnames_by_IDs', 'load_supervised_pair_by_IDs',
           'check_ID_duplicates']


def get_unique_IDs(fnames, globber=None, return_dict=False):
    iddict = {}
    for f in fnames:
        if globber is None:
            globber = "([0-9]{3,5})"

        mo = re.search(globber, f)
        if not mo is None:
            id = mo.group()
            if id in iddict.keys():
                iddict[id].append(f)
            else:
                iddict[id] = [f]

    if not return_dict:
        idlist =list(iddict.keys())
        idlist.sort()
        return idlist
    else:
        return iddict

def get_fnames_by_IDs(fnames,
                      idlist,
                      globber=None,
                      return_dict=False):
    _logger = MNTSLogger['algorithm.utils']
    if globber is None:
        globber = "([0-9]{3,5})"

    outfnames = {}
    id_fn_pair = get_unique_IDs(fnames, globber, return_dict=True)
    ids_in_fn = id_fn_pair.keys()

    id_not_in_fnames = set(idlist) - set(ids_in_fn)
    if len(id_not_in_fnames) > 0:
        _logger.warning(f"Cannot find anything for the following ids:\n"
                        f"{pprint.pformat(id_not_in_fnames)}")
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


def load_supervised_pair_by_IDs(source_dir, target_dir, idlist, globber=None, return_pairs=False):
    source_list = get_fnames_by_globber(os.listdir(source_dir), globber) \
        if not globber is None else os.listdir(source_dir)
    _logger = MNTSLogger['algorithm.utils']

    source_list = get_fnames_by_IDs(source_list, idlist, globber=globber, return_dict=True)
    source_keys = source_list.keys()
    source_list = [source_list[key][0] for key in source_list]
    target_list = get_fnames_by_IDs(os.listdir(target_dir), idlist, globber=globber, return_dict=True)
    target_keys = target_list.keys()
    target_list = [target_list[key][0] for key in source_keys]

    if len(source_list) != len(target_list):
        _logger.error("Dimension mismatch when pairing.")
        missing = {'Src': [], 'Target': []}
        for src in source_keys:
            if src not in target_keys:
                missing['Src'].append(src)
        for tar in target_keys:
            if tar not in source_keys:
                missing['Target'].append(src)
        _logger.debug(f"{missing}")
        raise ValueError("Dimension mismatch! Src: %i vs Target: %i"%(len(source_list), len(target_list)))

    pairs = list(zip(source_list, target_list))
    pairs.sort()
    if return_pairs:
        return pairs
    else:
        source_list, target_list = zip(*pairs)
        return source_list, target_list

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