import logging
import os
from collections import OrderedDict
from sys import stdout
from typing import Optional, Union

import h5py  # type: ignore
import numpy as np
from dedalus.tools import post  # type: ignore
from dedalus.tools.general import natural_sort  # type: ignore
from dedalus.tools.parallel import Sync  # type: ignore
from mpi4py import MPI
from tqdm import tqdm

logger = logging.getLogger(__name__.split(".")[-1])


def match_basis(dset: Union[h5py.Dataset, "RolledDset"], basis: str) -> np.ndarray:
    """Returns a 1D numpy array of the requested basis given a Dedalus dataset and basis name (string)."""
    for i in range(len(dset.dims)):
        if dset.dims[i].label == basis:
            return dset.dims[i][0][:].ravel()
    raise ValueError(f"Basis '{basis}' not found in dataset in match_basis.")


class RolledDset:
    """A wrapper for data which has been rolled in time to make it behave like a Dedalus dataset."""

    def __init__(self, dset: h5py.Dataset, ni: int, rolled_data: np.ndarray):
        self.dims = dset.dims
        self.ni = ni
        self.data = rolled_data

    def __getitem__(self, ni: Union[int, tuple]) -> np.ndarray:
        if isinstance(ni, tuple) and ni[0] == self.ni:
            return self.data[ni[1:]]
        if ni == self.ni:
            return self.data
        else:
            raise ValueError("Wrong index value used for currently stored rolled data")


class FileReader:
    """
    A general class for reading and interacting with Dedalus output data.
    This class takes a list of dedalus files and distributes them across MPI processes according to a specified rule.
    """

    def __init__(
        self,
        run_dir: str,
        distribution: str = "even-write",
        sub_dirs: list[str] = [
            "slices",
        ],
        num_files: list[Optional[int]] = [
            None,
        ],
        start_file: int = 1,
        global_comm: MPI.Intracomm = MPI.COMM_WORLD,
        chunk_size: int = 1000,
    ):
        """
        Initializes the file reader.

        # Arguments
        run_dir : Root directory of the Dedalus run.
        distribution : Type of MPI file distribution ['single', 'even-write', 'even-file', 'even-chunk']
        sub_dirs : Directories corresponding to the names of dedalus file handlers.
        num_files : Max number of files to read in each subdirectory. If None, read them all.
        start_file : File number to start reading from (1 by default)
        global_comm : MPI communicator to use for distributing files.
        chunk_size : If distribution == 'even-chunk', this is the number of writes per chunk.
        """
        self.run_dir = os.path.expanduser(run_dir)
        self.sub_dirs = sub_dirs
        self.file_lists = OrderedDict()
        self.global_comm = global_comm

        # Get all files ending in _s*.h5
        for d, n in zip(sub_dirs, num_files):
            files = []
            for f in os.listdir("{:s}/{:s}/".format(self.run_dir, d)):
                if f.endswith(".h5"):
                    file_num = int(f.split(".h5")[0].split("_s")[-1])
                    if file_num < start_file:
                        continue
                    if n is not None and file_num >= start_file + n:
                        continue
                    files.append("{:s}/{:s}/{:s}".format(self.run_dir, d, f))
            self.file_lists[d] = natural_sort(files)

        self.file_starts: dict[str, np.ndarray] = (
            OrderedDict()
        )  # Start index of files for each file type
        self.file_counts: dict[str, np.ndarray] = (
            OrderedDict()
        )  # Number of counts per file for each file type
        self.comms: dict[str, MPI.Comm] = (
            OrderedDict()
        )  # MPI communicators for each file type
        self.idle: dict[str, bool] = (
            OrderedDict()
        )  # Whether or not this processor is idle for each file type
        self._distribute_writes(distribution, chunk_size=chunk_size)

    def _distribute_writes(self, distribution: str, chunk_size: int = 100) -> None:
        """
        Distribute writes (or files) across MPI processes according to the specified rule.

        Currently, these types of file distributions are implemented:
            1. 'single'       : First process takes all file tasks
            2. 'even-write'   : evenly distribute total number of writes over all mpi processes
            3. 'even-file'    : evenly distribute total number of files over all mpi processes
            4. 'even-chunk'   : evenly distribute chunks of writes over all mpi processes

        # Arguments
            distribution (string) :
                Type of MPI file distribution
            chunk_size (int, optional) :
                If distribution == 'even-chunk', this is the number of writes per chunk.
        """
        # Loop over each file handler type
        for k, files in self.file_lists.items():
            writes = np.array(post.get_all_writes(files))
            set_ends = np.cumsum(writes)
            set_starts = set_ends - writes
            self.idle[k] = False

            # Distribute writes
            if distribution.lower() == "single":
                # Process 0 gets all files, all other processes are idle
                num_procs = 1
                if self.global_comm.rank == 0:
                    self.file_starts[k] = np.zeros_like(writes)
                    self.file_counts[k] = np.copy(writes)
                else:
                    self.file_starts[k] = np.copy(writes)
                    self.file_counts[k] = np.zeros_like(writes)
                    self.idle[k] = True
            elif distribution.lower() == "even-write":
                # Evenly distribute writes over all processes using Dedalus logic
                self.file_starts[k], self.file_counts[k] = post.get_assigned_writes(
                    files
                )
                writes_per = np.ceil(np.sum(writes) / self.global_comm.size)
                num_procs = int(np.ceil(np.sum(writes) / writes_per))
            elif distribution.lower() == "even-file":
                # Evenly distribute files over all processes; files are not split by write.
                self.file_starts[k] = np.copy(writes)
                self.file_counts[k] = np.zeros_like(writes)
                if len(files) <= self.global_comm.size:
                    # Some processes will be idle
                    num_procs = len(files)
                    if self.global_comm.rank < len(files):
                        self.file_starts[k][self.global_comm.rank] = 0
                        self.file_counts[k][self.global_comm.rank] = writes[
                            self.global_comm.rank
                        ]
                else:
                    # All processes will have at least one file
                    file_per = int(np.ceil(len(files) / self.global_comm.size))
                    proc_start = self.global_comm.rank * file_per
                    self.file_starts[k][proc_start : proc_start + file_per] = 0
                    self.file_counts[k][proc_start : proc_start + file_per] = writes[
                        proc_start : proc_start + file_per
                    ]
                    num_procs = int(np.ceil(len(files) / file_per))
            elif distribution.lower() == "even-chunk":
                num_procs = int(np.ceil(np.sum(writes) / chunk_size))
                chunk_adjust = 1
                if num_procs > self.global_comm.size:
                    # If there are more chunks than processes, figure out how many chunks each process will get
                    # TODO: Check if this is correct.
                    chunk_adjust = int(np.floor(num_procs / self.global_comm.size))
                    if self.global_comm.rank < num_procs % self.global_comm.size:
                        chunk_adjust += 1
                    num_procs = self.global_comm.size
                chunk_size *= chunk_adjust
                proc_start = self.global_comm.rank * chunk_size
                self.file_starts[k] = np.clip(
                    proc_start, a_min=set_starts, a_max=set_ends
                )
                self.file_counts[k] = (
                    np.clip(proc_start + chunk_size, a_min=set_starts, a_max=set_ends)
                    - self.file_starts[k]
                )
                self.file_starts[k] -= set_starts
            else:
                raise ValueError(
                    "invalid distribution choice. Please choose 'single', 'even-write', 'even-file', or 'even-chunk'"
                )

            # Distribute comms
            if num_procs == self.global_comm.size:
                self.comms[k] = self.global_comm
            else:
                if self.global_comm.rank < num_procs:
                    self.comms[k] = self.global_comm.Create(
                        self.global_comm.Get_group().Incl(range(num_procs))
                    )
                else:
                    self.comms[k] = MPI.COMM_SELF
                    self.idle[k] = True


class RollingFileReader(FileReader):
    """
    Distributes files for even processing, but also keeps track of surrounding writes
    and takes rolling averages of the data.

    TODO: Write unit tests for this class' distribution method. Possibly use pandas to roll the data.
    """

    def __init__(
        self,
        run_dir: str,
        distribution: str = "even-write",
        sub_dirs: list[str] = [
            "slices",
        ],
        num_files: list[Optional[int]] = [
            None,
        ],
        start_file: int = 1,
        global_comm: MPI.Intracomm = MPI.COMM_WORLD,
        chunk_size: int = 1000,
        roll_writes: int = 10,
    ):
        """
        Initialize RollingFileReader. roll_writes is the number of writes (before and after current write) to average over.
        So if roll_writes = 10, then the average is over 20 writes (current, 10 before, 9 after).
        """
        self.roll_writes = roll_writes
        super().__init__(
            run_dir,
            distribution=distribution,
            sub_dirs=sub_dirs,
            num_files=num_files,
            start_file=start_file,
            global_comm=global_comm,
            chunk_size=chunk_size,
        )

    def _distribute_writes(self, distribution: str, chunk_size: int = 100) -> None:
        super()._distribute_writes(distribution=distribution, chunk_size=chunk_size)
        self.roll_starts, self.roll_counts = OrderedDict(), OrderedDict()

        for k, files in self.file_lists.items():
            writes = np.array(post.get_all_writes(files))
            set_ends = np.cumsum(writes)
            set_starts = set_ends - writes
            global_writes = np.sum(writes)
            file_indices = np.arange(len(set_starts))

            base_starts, base_counts = self.file_starts[k], self.file_counts[k]
            local_writes = np.sum(base_counts)
            self.roll_starts[k] = np.zeros(
                (local_writes, len(base_counts)), dtype=np.int32
            )
            self.roll_counts[k] = np.zeros(
                (local_writes, len(base_counts)), dtype=np.int32
            )

            # Build array containing global indices of all writes if all files were concatenated.
            global_indices = np.zeros(local_writes, dtype=np.int32)
            counter = 0
            for i, counts in enumerate(base_counts):
                if counts > 0:
                    for j in range(counts):
                        global_indices[counter] = set_starts[i] + base_starts[i] + j
                        counter += 1

            for i in range(local_writes):
                # Find start index, decrement by roll_writes
                roll_start_global = global_indices[i] - self.roll_writes
                roll_end_global = global_indices[i] + self.roll_writes
                if roll_start_global < 0:
                    roll_start_global = 0
                elif roll_end_global > global_writes - 1:
                    roll_end_global = global_writes - 1
                file_index = file_indices[
                    (roll_start_global >= set_starts) * (roll_start_global < set_ends)
                ][0]
                self.roll_starts[k][i, file_index] = (
                    roll_start_global - set_starts[file_index]
                )
                remaining_writes = roll_end_global - roll_start_global
                while remaining_writes > 0:
                    remaining_this_file = (
                        writes[file_index] - self.roll_starts[k][i, file_index]
                    )
                    if remaining_writes > remaining_this_file:
                        counts = remaining_this_file
                    else:
                        counts = remaining_writes
                    self.roll_counts[k][i, file_index] = counts
                    remaining_writes -= counts
                    file_index += 1


class SingleTypeReader:
    """
    A class for reading Dedalus data from a single file handler.
    """

    def __init__(
        self,
        run_dir: str,
        sub_dir: str,
        out_name: str,
        distribution: str = "even-write",
        num_files: Optional[int] = None,
        roll_writes: Optional[int] = None,
        start_file: int = 1,
        global_comm: MPI.Intracomm = MPI.COMM_WORLD,
        chunk_size: int = 1000,
    ):
        """
        Initializes the reader.

        # Arguments
            run_dir (str) :
                Root file directory of output files
            sub_dir (str) :
                subdirectory of run_dir where the data to make PDFs is contained
            out_name (str) :
                The name of an output directory to create inside run_dir. Also the base name of output figures
            num_files  (int, optional) :
                Number of files to process. If None, all of them.
            roll_writes (int, optional) :
                Number of writes to average over. If None, no rolling average is taken.
            kwargs (dict) :
                Additional keyword arguments for FileReader()
        """

        self.roll_counts: Optional[np.ndarray] = None
        self.roll_starts: Optional[np.ndarray] = None
        if roll_writes is None:
            self.reader = FileReader(
                run_dir,
                sub_dirs=[
                    sub_dir,
                ],
                num_files=[
                    num_files,
                ],
                distribution=distribution,
                start_file=start_file,
                global_comm=global_comm,
                chunk_size=chunk_size,
            )
        else:
            self.reader = RollingFileReader(
                run_dir,
                sub_dirs=[
                    sub_dir,
                ],
                num_files=[
                    num_files,
                ],
                roll_writes=roll_writes,
                distribution=distribution,
                start_file=start_file,
                global_comm=global_comm,
                chunk_size=chunk_size,
            )
            self.roll_counts = self.reader.roll_counts[sub_dir]
            self.roll_starts = self.reader.roll_starts[sub_dir]
        self.out_name = out_name
        self.out_dir: str = "{:s}/{:s}/".format(run_dir, out_name)
        if self.reader.global_comm.rank == 0 and not os.path.exists(
            "{:s}".format(self.out_dir)
        ):
            os.mkdir("{:s}".format(self.out_dir))

        self.files = self.reader.file_lists[sub_dir]
        self.idle = self.reader.idle[sub_dir]
        self.comm = self.reader.comms[sub_dir]
        self.my_sync = Sync(self.comm)
        self.starts = self.reader.file_starts[sub_dir]
        self.counts = self.reader.file_counts[sub_dir]
        self.writes = np.sum(self.counts)
        self.output: dict[str, Union[h5py.Dataset, RolledDset]] = OrderedDict()

        if not self.idle:
            file_num = []
            local_indices = []
            for i, c in enumerate(self.counts):
                if c > 0:
                    local_indices.append(np.arange(c, dtype=np.int64) + self.starts[i])
                    file_num.append(i * np.ones(c, dtype=np.int64))
            if len(local_indices) >= 1:
                self.file_index = np.array(
                    np.concatenate(local_indices), dtype=np.int64
                )
                self.file_num = np.array(np.concatenate(file_num), dtype=np.int64)
            else:
                raise ValueError("No merged or virtual files found")

            self.current_write: int = -1
            self.current_file_handle: Optional[h5py.File] = None
            self.current_file_number: Optional[int] = None
            self.current_file_name: Optional[str] = None

        self.pbar: Optional[tqdm] = None

    def writes_remain(self) -> bool:
        """
        Increments to the next write on the local MPI process.
        Returns False if there are no writes left and True if a write is found.

        For use in a while statement (e.g., while writes_remain(): do stuff).
        """
        if not self.idle:
            if (
                self.current_write >= self.writes - 1
                and self.current_file_handle is not None
            ):
                self.current_write = -1
                self.current_file_handle.close()
                self.current_file_handle = None
                self.current_file_number = None
                self.current_file_name = None
                return False
            else:
                self.current_write += 1
                next_file_number = self.file_num[self.current_write]
                if self.current_file_number is None:
                    # First iter
                    self.current_file_number = next_file_number
                    self.current_file_name = self.files[self.current_file_number]
                    self.current_file_handle = h5py.File(self.current_file_name, "r")
                elif (
                    self.current_file_number != next_file_number
                    and self.current_file_handle is not None
                ):
                    self.current_file_handle.close()
                    self.current_file_number = next_file_number
                    self.current_file_name = self.files[self.current_file_number]
                    self.current_file_handle = h5py.File(self.current_file_name, "r")
                return True
        else:
            return False

    def get_dsets(
        self, tasks: list[str], verbose: bool = True
    ) -> tuple[dict[str, Union[h5py.Dataset, RolledDset]], int]:
        """Given a list of task strings, returns a dictionary of the associated datasets and the dset index of the current write."""
        if not self.idle:
            if self.comm.rank == 0 and verbose:
                if self.pbar is None:
                    self.pbar = tqdm(total=self.writes)
                self.pbar.update(1)
                self.pbar.set_description(
                    "gathering {} tasks; write {}/{} on process 0".format(
                        tasks, self.current_write + 1, self.writes
                    )
                )
                stdout.flush()
                if self.pbar.n == self.writes:
                    self.pbar.close()
                    self.pbar = None

            assert self.current_file_handle is not None, ValueError(
                "No file handle found."
            )

            f = self.current_file_handle
            ni = self.file_index[self.current_write]
            for k in tasks:
                if isinstance(self.reader, RollingFileReader):
                    assert self.roll_counts is not None, ValueError(
                        "No roll counts found."
                    )
                    assert self.roll_starts is not None, ValueError(
                        "No roll starts found."
                    )
                    base_dset = f["tasks/{}".format(k)]
                    rolled_data = np.zeros_like(base_dset[0, :])
                    rolled_counter = 0
                    ri = self.current_write
                    for i, c in enumerate(self.roll_counts[ri, :]):
                        if c > 0:
                            local_indices = (
                                np.arange(c, dtype=np.int64) + self.roll_starts[ri, i]
                            )
                            with h5py.File(self.files[i], "r") as rf:
                                dset = rf["tasks/{}".format(k)]
                                rolled_data += np.sum(dset[local_indices], axis=0)
                                rolled_counter += c
                    rolled_data /= rolled_counter
                    self.output[k] = RolledDset(base_dset, ni, rolled_data)
                else:
                    self.output[k] = f["tasks/{}".format(k)]
            return self.output, ni
        else:
            return {}, -1
