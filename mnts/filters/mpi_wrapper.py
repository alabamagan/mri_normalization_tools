import multiprocessing as mpi
import multiprocessing.pool as Pool
import copyreg
from typing import Union, Callable, Iterable, Any
from ..utils import repeat_zip
from ..mnts_logger import MNTSLogger

__all__ = ['mpi_wrapper']

def mpi_wrapper(func: Callable,
                args: Iterable[Any],
                num_worker: int = None):
    r"""

    .. warning::
        This wrapper does not manage memory usage for you so beware.

    Args:
        func (callable):
            Class function could work here.
        args (list of iterables):
            Should have a structure of [(args1), (args21, args22, args23...), ...].
        num_worker (int, Optional):
            If None, use `mpi.cpu_count()` to calculate nubmer of workers required.
    Returns:

    """
    num_worker = int(mpi.cpu_count() if num_worker is None else num_worker)
    assert num_worker >= 2, "Number of worker less than 2 should not be executed with this function."

    logger = MNTSLogger['MPI']
    pool = Pool.ThreadPool(num_worker)
    res = pool.starmap_async(func, repeat_zip(*args))
    pool.close()
    pool.join()
    return res.get()

