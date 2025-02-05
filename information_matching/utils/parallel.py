"""Some custom classes to do parallelization. We might not need to use any of these, but
they are available if we need them.
"""

import sys
import random

import numpy as np
import multiprocessing as mp
import multiprocessing.pool
import queue
from multiprocessing.reduction import ForkingPickler, AbstractReducer

# We set to use pickle 4 in case if we need to deal with large data. However,
# it is recommended not to deal with large data directly if possible.


class ForkingPickler4(ForkingPickler):
    def __init__(self, *args):
        if len(args) > 1:
            args[1] = 2
        else:
            args.append(2)
        super().__init__(*args)

    @classmethod
    def dumps(cls, obj, protocol=4):
        return ForkingPickler.dumps(obj, protocol)


def dump(obj, file, protocol=4):
    ForkingPickler4(file, protocol).dump(obj)


class Pickle4Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler4
    register = ForkingPickler4.register
    dump = dump


ctx = mp.get_context()
ctx.reducer = Pickle4Reducer()
np.random.seed(2021)


def mymp(target, njobs, nprocesses, args=(), keys=()):
    """My wrapper to do multiprocessing, to make my job easier so I don't need
    to write this routine over and over when I need it. Here, I also set to use
    pickle 4 in case if I need to deal with large data and pass it accross
    processes. However, it is recommended not to deal with large data directly
    if possible, instead we can load the data globally and just include the
    index in the parallel processes.

    Parameters
    ----------
    target: callable
        Target function that does the computation. See the example below on the
        requirements of this function.
    njobs: int
        Number of jobs to do.
    nprocesses: int
        Number of processes to ccreate.
    args: tuple
        Arguments that the target take.
    keys: list or tuple of str
        Keys of the final dictionary result.

    Returns
    -------
    result_dict: dict
        Dictionary of the results of the computation.

    Notes
    -----
    In principle, calling this function is equivalent to running the following
    commands in parallel ::

        result_dict = {}
        for ii in range(njobs):
            result_dict[keys[ii]] = target(ii, *args)
    """
    # Setup multiprocessing
    task_to_accomplish = mp.Queue()
    task_done = mp.Queue()
    Processes = []
    manager = mp.Manager()
    result_dict = manager.dict()

    # Check the keys
    if not keys:
        keys = np.arange(njobs)

    # Creating jobs
    for ii in range(njobs):
        task_to_accomplish.put(f"Task no. {ii} for job index {ii}")

    # creating processes
    fargs = (target, result_dict, task_to_accomplish, task_done, keys, args)
    for ii in range(nprocesses):
        Proc = mp.Process(
            target=_target_wrapper,
            args=fargs,
        )
        Processes.append(Proc)

    for Proc in Processes:
        Proc.start()

    for Proc in Processes:
        Proc.join()

    while not task_done.empty():
        task_done.get()

    return dict(result_dict)


def _target_wrapper(target, result_dict, task_to_accomplish, task_done, keys, args):
    """Wrapper to the target function.

    Parameters
    ----------
    target: callable
        Target function that does the computation. See the example below on the
        requirements of this function.
    result_dict: mp.Manager.dict dictionary
        Dictionary to store the results.
    task_to_accomplish: mp.Queue object
        Task to accomplish.
    task_done: mp.Queue object
        Task done.
    args: tuple
        Additional argument that will be passed into the target.
    """
    while True:
        try:
            task = task_to_accomplish.get_nowait()
        except queue.Empty:
            break
        else:
            ii = np.array(task.split(" "))[-1].astype(int)
            temp_mat = target(ii, *args)
            result_dict[keys[ii]] = temp_mat

            task_done.put(
                f"Task for job index {ii} " f"is done by {mp.current_process().name}"
            )


class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(mp.get_context())):
    Process = NoDaemonProcess


# We sub-class mp.pool.Pool instead of mp.Pool because the latter is only a
# wrapper function, not a proper class.
class NonDaemonicPool(multiprocessing.pool.Pool):
    """This multiprocessing pool can be used for nested multiprocessing. I am not sure
    if this practice has any catastrophic consequence.
    """

    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(NonDaemonicPool, self).__init__(*args, **kwargs)


# Custom multiprocessing pool to handle unpicklable KIM object.
class MyPool(multiprocessing.pool.Pool):
    """An alternative to ``mp.Pool``.

    The difference is that this pool class can be used for non picklable
    functions. I inherited from ``mp.Pool``, with modification in the ``map``
    method, which is a copy of the parallelization function in KLIFF.

    Notes
    -----
    This implementation might not be the best or optimized, but this is
    temporary solution (if nothing wrong happens, it can be a permanent fix).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctx = self.get_context()

    def map(self, f, X, *args, tuple_X=False):
        """This is to mimic ``multiprocessing.Pool.map``, which requires the
        function ``f`` to be picklable. This function does not have this
        restriction and allows extra arguments to be used for the function
        ``f``.

        Parameters
        ----------
        f: function
            The function that operates on the data.

        X: list
            Data to be parallelized.

        args: args
            Extra positional arguments needed by the function ``f``.

        tuple_X: bool
            This depends on ``X``. It should be set to ``True`` if multiple
            arguments are parallelized and set to ``False`` if only one
            argument is parallelized.

        Return
        ------
        list
            A list of results, corresponding to ``X``.
        """

        # shuffle and divide into nprocs equally-numbered parts
        if tuple_X:
            # to make array_split work
            pairs = [(i, *x) for i, x in enumerate(X)]
        else:
            pairs = [(i, x) for i, x in enumerate(X)]
        random.shuffle(pairs)
        groups = np.array_split(pairs, self._processes)

        processes = []
        managers = []
        for i in range(self._processes):
            manager_end, worker_end = self.ctx.Pipe(duplex=False)
            p = self.ctx.Process(target=self._func, args=(f, groups[i], args, worker_end))
            p.daemon = False
            p.start()
            processes.append(p)
            managers.append(manager_end)
        results = []
        for m in managers:
            results.extend(m.recv())
        for p in processes:
            p.join()

        return [r for i, r in sorted(results)]

    @staticmethod
    def _func(f, iX, args, worker_end):
        results = []
        for ix in iX:
            i = ix[0]
            x = ix[1:]
            results.append((i, f(*x, *args)))
        worker_end.send(results)

    @staticmethod
    def get_context():
        if sys.platform == "darwin":
            ctx = mp.get_context("fork")
        else:
            ctx = mp.get_context()
        return ctx
