import re, os
from ..mnts_logger import MNTSLogger

__all__ = ['get_unique_IDs', 'get_fnames_by_globber', 'get_fnames_by_IDs', 'load_supervised_pair_by_IDs']


def get_unique_IDs(fnames, globber=None):
    idlist = []
    for f in fnames:
        if globber is None:
            globber = "([0-9]{3,5})"

        mo = re.search(globber, f)
        if not mo is None:
            idlist.append(f[mo.start():mo.end()])

    idlist = list(set(idlist))
    idlist.sort()
    return idlist


def get_fnames_by_IDs(fnames, idlist, globber=None):
    _logger = MNTSLogger['algorithm.utils']
    if globber is None:
        globber = "([0-9]{3,5})"

    outfnames = {}
    for id in idlist:
        flist = []
        for f in fnames:
            _f = os.path.basename(f)
            l = re.findall(globber, _f)
            if not len(l):
                continue
            if l[0] == id:
                flist.append(f)
        # skip if none is found
        if len(flist) == 0:
            _logger.warning(f"Can't found anything for key {id}. Skipping..")
            continue
        outfnames[id] = flist
    return outfnames


def get_fnames_by_globber(fnames, globber):
    assert isinstance(fnames, list)

    copy = list(fnames)
    for f in fnames:
        if re.match(globber, f) is None:
            copy.remove(f)
    return copy


def load_supervised_pair_by_IDs(source_dir, target_dir, idlist, globber=None):
    source_list = get_fnames_by_globber(os.listdir(source_dir), globber) \
        if not globber is None else os.listdir(source_dir)
    _logger = MNTSLogger['algorithm.utils']

    source_list = get_fnames_by_IDs(source_list, idlist)
    source_keys = source_list.keys()
    source_list = [source_list[key][0] for key in source_list]
    target_list = get_fnames_by_IDs(os.listdir(target_dir), idlist)
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

    return source_list, target_list