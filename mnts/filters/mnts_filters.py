import copy
from abc import ABCMeta, abstractmethod, abstractproperty

import SimpleITK as sitk
import numpy as np
import multiprocessing as mpi

from ..mnts_logger import MNTSLogger
from typing import Union, Iterable, List, Any
from pathlib import Path
from cachetools import cached, LRUCache
from copy import deepcopy

import networkx as nx

__all__ = ['MNTSFilterGraph', 'MNTSFilter', 'MNTSFilterPipeline']

class MNTSFilter(object):
    def __init__(self):
        r"""
        Base class of filter
        """
        self._logger = MNTSLogger[self.get_name()]

    @abstractmethod
    def filter(self, *args, **kwargs):
        raise NotImplemented("This is an abstract method.")

    @property
    def name(self):
        return self.get_name()

    def get_name(self):
        return self.__class__.__name__

    def get_all_properties(self):
        n = [(name, self.__getattribute__(name)) for name, value in vars(self.__class__).items() if isinstance(value, property)]
        return n

    @staticmethod
    def read_image(input: Union[str, Path, sitk.Image]):
        if isinstance(input, (str, Path)):
            input = sitk.ReadImage(str(input))

        if not isinstance(input, sitk.Image):
            raise IOError
        return input


    def __call__(self, *args, **kwargs):
        return self.filter(*args, **kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}: \n\t" + \
               '\n\t'.join(["{: >10}: {}".format(item[0], item[1]) for item in self.get_all_properties()])

class MNTSFilterRequireTraining(MNTSFilter):
    def __init__(self):
        r"""
        Base class of filter
        """
        super(MNTSFilterRequireTraining, self).__init__()

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    def save_state(self):
        pass

    def load_state(self):
        pass


class MNTSFilterPipeline(list):
    def __init__(self,
                 *args: Iterable[MNTSFilter]
                 ):
        r"""
        A list of filter that will be called in sequential order
        Args:
            *args:
        """
        super(MNTSFilterPipeline, self).__init__(*args)

        self._logger = MNTSLogger[self.__class__.__name__]

        if len(args) > 1:
            assert all([isinstance(f, MNTSFilter) for f in args])

    def __str__(self):
        return '->'.join([f.name for f in self])

    def execute(self, *args):
        for f in iter(self):
            args = f(*args)
            if not isinstance(args, [tuple, list]):
                args = [args]
        return args

    def sort(self):
        # Disable sort
        pass

class MNTSFilterGraph(object):
    r"""
    A directed graph that links the filters together to enable more complex use of filters.
    """
    def __init__(self):
        super(MNTSFilterGraph, self).__init__()
        self._graph = nx.DiGraph()
        self._entrance = []
        self._exits = []
        self._logger = MNTSLogger[self.__class__.__name__]
        self._nodemap = {}
        self._nodes_cache = {}

    @property
    def nodes(self):
        return self._graph.nodes

    @nodes.setter
    def nodes(self, nodes):
        r"""
        Setting the nodes.

        .. warning::
            This method will clear all the nodes and edges. Set the nodes before you set the edges!
        """
        assert isinstance(nodes, (list, tuple))
        assert all([isinstance(a, MNTSFilter) for a in nodes]), "There are non-MNTSFitler elements in the nodes."
        self._graph.clear()
        for i, f in enumerate(nodes):
            self._graph.add_node(i, filter=f)

    @property
    def edges(self):
        return self._graph.edges

    def _node_search(self,
                     attr_key: str,
                     attr_val: Any) -> List[int]:
        r"""
        Search for node where its attribute with key:`attr_key` equal `attr_val`. There could be multiple nodes that
        fits the search so output is always a list.

        Args:
            attr_key (str):
                The key of the attribute.
            attr_val (Any):
                The value used as searching parameter.

        Returns:
            out (list of int):
                The list of nodes that fits the search criteria.
        """
        out = []
        for n, v in self.nodes.data(attr_key):
            if v == attr_val:
                out.append(n)
        if len(out) == 0:
            return -1
        else:
            if len(out) > 1:
                self._logger.warning(f"Same filter is reused in the graph: {out}")
            return out

    def add_node(self,
                 node: MNTSFilter,
                 upstream: Union[None, int, MNTSFilter, Iterable[Union[MNTSFilter, int]]] = None,
                 is_exit: bool =False):
        r"""
        Add a node to the filter graph and connect it to the upstream nodes. If no upstream nodes are provided this
        node is automatically treated as the input node.

        Exit nodes, where marks the end of the normalization pipeline must be labeled using `is_exit` argument.

        .. note::
            The order of `upstream` is important and decides the argument order when inputs are required from the
            upstream.

        Args:
            node (MNTSFilter):
                The filter that is to be put into the list. Note that filter cannot be reused as it forms looped
                graphs that are trouble to deal with.
            upstream (MNTSFitler or int or list(MNTSFitler or int)):
                The upstream source of data of this node. Permit multiple upstream, however, one node cannot have
                more than one downstream.

        """
        assert isinstance(node, MNTSFilter), f"Wrong input type: {node}"

        if self._node_search('filter', node) == -1:
            _node_index = self._graph.number_of_nodes()
            self._graph.add_node(_node_index, filter=node)
            self._nodemap[_node_index] = node.get_name()
            if is_exit:
                self._exits.append(_node_index)
        else:
            _node_index = self._node_search('filter', node)[0]

        if upstream is None:
            self._entrance.append(_node_index)

        elif isinstance(upstream, MNTSFilter):
            try:
                _upstream_index = self._node_search('filter', upstream)
            except ValueError:
                raise ValueError("Specified upstream is not in the list of nodes.")
            # Check if upstream node has another downstream, which is forbiddened
            if self._graph.out_edges(_upstream_index) > 0:
                raise ArithmeticError("Nodes cannot have multiple downstreams.")
            self._graph.add_edge(_upstream_index, _node_index)

        elif isinstance(upstream, int):
            if not upstream in self.nodes:
                raise IndexError("Upstream specified not in list.")
            # Check if upstream node has another downstream, which is forbiddened
            # if len(self._graph.out_edges(upstream)) > 0:
            #     raise ArithmeticError(f"Nodes cannot have multiple downstreams: {upstream}")
            self._graph.add_edge(upstream, _node_index)

        elif isinstance(upstream, (list, tuple)):
            for us in upstream:
                self.add_node(node, us)

    def _request_output(self, node_id):
        r"""
        Use a bottom up request to walk the graph finding the paths and then generate the output. Request are
        recursively requested to get data from the upstream.
        """
        in_edges = self._graph.in_edges(node_id)
        upstream_nodes = [t[0] for t in in_edges]

        cur_filter = self.nodes[node_id]['filter']
        if not node_id in self._entrance:
            data = [self._request_output(i) for i in upstream_nodes]
            self._logger.info(f"Finished step: {self._nodemap[node_id]}")
            if self._nodes_cache.get(node_id, None) is not None: # put this in cache if used many times
                return self._nodes_cache[node_id]
            elif node_id in self._nodes_cache:
                self._nodes_cache[node_id] = cur_filter(*data)
                return self._nodes_cache[node_id]
            else:
                return cur_filter(*data)
        else:
            # If it an entrance node, acquire input from dictionary.
            self._logger.info(f"Finished step: {self._nodemap[node_id]}")
            return cur_filter(self._inputs[node_id])

    def plot_graph(self):
        #TODO: write this nodemap as the legend of the graph.
        print(self._nodemap)
        nx.draw(self._graph, with_labels=True)

    def execute(self,
                *args,
                force_request: Union[int, MNTSFilter] = None) -> sitk.Image:
        r"""

        Args:
            *args:
                Arguments are passed to the nodes in list `self._entrance`. If the node require multiple input
                parameters, pass them as a tuple.
            force_request:
                If True, request output from the specified node instead of that in `self._exit`.

        Returns:

        """
        assert len(args) == len(self._entrance), "Inputs do not align with number of entrance."

        # mark all nodes with multiple downstreams
        num_of_downstream = {n: len(self._graph.out_edges(n)) for n in self.nodes}
        for key in num_of_downstream:
            if num_of_downstream[key] >= 2:
                self._nodes_cache[key] = None

        self._inputs = {n: args[i] for i, n in enumerate(self._entrance)}

        # gather_output
        output = {}
        if force_request is None:
            for i in self._exits:
                output[i] = self._request_output(i)
        else:
            if isinstance(force_request, MNTSFilter):
                force_request = self._node_search('filter', force_request)
            output[force_request] = self._request_output(force_request)
        return output

    def prepare_training_files(self,
                               nodelist: List[Union[int, MNTSFilter]],
                               output_prefix: str,
                               output_directory: Union[str, Path],
                               *args) -> None:
        r"""
        Call the `train` method with *args passed to the methods. This method process one input at a time, thus, need
        to call this repeatedly to fully prepare the files.

        Example of folder structure of the intermediate files
        .
        └── working_dir/
            └── temp_file_dir/
                ├── 5_trained_node/
                │   ├── 3_upstream_node/
                │   │   ├── output_1
                │   │   ├── output_2
                │   │   └── ...
                │   └── 2_upstream_node/
                │       ├── output_1
                │       ├── output_2
                │       └── ...
                └── 6_trained_node_B/
                    └── 3_upstream_node/ # Same upstream node will make duplications
                        ├── output_1
                        ├── output_2
                        └── ...

        Args:
            nodelist (list of int or MNTSFilter):
                The (list of) nodes that require training. The output from their upstream(s) are collected and
                computed and them put into the corresponding locations stated above.
            output_prefix (str):
                Name for the output. They are saved as `{output_prefix}`.nii.gz.
            output_directory (str or Path):
                Specify where to store the output from the upstream nodes.
            *args:
                See `:method:self.Execute`.
            **kwargs:

        Returns:
            None
        """
        temp_dir = Path(output_directory).absolute()
        if not isinstance(nodelist, (list, tuple)):
            nodelist = [nodelist]

        #!! Might get some memory issue here, but should be managible.
        # Create a copy of this class to separate
        cls_obj = copy.deepcopy(self)

        for n in nodelist:
            # Create directories first
            n = cls_obj._node_search('filter', n) if isinstance(n, MNTSFilter) else n
            node_name = f"{n}_" + cls_obj.nodes[n]['filter'].get_name()
            node_dir = temp_dir.joinpath(f"{node_name}/")
            if not node_dir.is_dir():
                node_dir.mkdir(exist_ok=True, parents=True) # sometimes mpi will mkdir a few times, exist_ok to
                                                            # prevent it from reporting error

            # Get upstream nodes
            u_nodes = [x[0] for x in cls_obj._graph.in_edges(n)]
            for u_node in u_nodes:
                u_node_name = f"{u_node}_" + cls_obj.nodes[u_node]['filter'].get_name()
                u_node_dir = node_dir.joinpath(f"{u_node_name}/")
                if not u_node_dir.is_dir():
                    u_node_dir.mkdir(exist_ok=True, parents=True)

                # Get output from these upstream nodes and save them into corresponding temp folders
                out = cls_obj.execute(*args, force_request=u_node)
                out_name = u_node_dir.joinpath(f"{output_prefix}")
                sitk.WriteImage(out[u_node], str(out_name.with_suffix('.nii.gz')))

    def train_node(self,
                   nodelist: List[Union[int, MNTSFilter]],
                   training_inputs: Union[str, Path]):
        r"""
        
        """

        input_path = Path(training_inputs).resolve()
        assert input_path.is_dir(), f"Cannot open training inputs at {input_path.__str__()}"
        if not isinstance(nodelist, (list, tuple)):
            nodelist = [nodelist]

        out = {}
        # Collect list of training inputs first.
        for n in nodelist:
            trained_node = self._node_search('filter', n) if isinstance(n, MNTSFilter) else n
            trained_node_filter = self.nodes[n]['filter']
            trained_node_name = f"{n}_" + trained_node_filter.get_name()
            # check if the target node is trainable
            if not isinstance(trained_node_filter, MNTSFilterRequireTraining):
                raise ArithmeticError(f"Specified node {trained_node_name} is not trainable.")

            trained_node_files_dir = input_path.joinpath(trained_node_name)
            if not trained_node_files_dir.is_dir():
                msg = f"Cannot open directory for training node {trained_node_name} at: " \
                      f"{trained_node_files_dir.resolve().__str__()}"
                raise IOError(msg)

            # Get upsteam nodes from which the training images are prepared by
            # calling `self.prepare_training_files`.
            u_nodes = [nn[0] for nn in self._graph.in_edges(n)]
            u_nodes_files = []
            for u_node in u_nodes:
                u_node_name = f"{u_node}_" + self.nodes[u_node]['filter'].get_name()
                u_node_dir = trained_node_files_dir.joinpath(u_node_name)
                if not u_node_dir.is_dir():
                    msg = f"Cannot open directory for training node {u_node_name} at: " \
                          f"{u_node_dir.resolve().__str__()}"
                    raise IOError(msg)

                u_nodes_files.append([str(r.resolve()) for r in u_node_dir.iterdir()
                                      if r.name.find('nii') != -1])
            trained_node_filter.train(*u_nodes_files)
        return out


    def __deepcopy__(self, memodict={}):
        r"""
        Note that every thing is copied except for the logger and the graph because deepcopying existing
        graph that has attributes is problematic given they are filters.

        This might be solvable through implementing __deepcopy__ for the filters.
        """
        cpyobj = type(self)()
        attr_copy = ['_entrance', '_exits', '_nodemap', '_nodes_cache']
        for attr in attr_copy:
            cpyobj.__setattr__(attr, deepcopy(self.__getattribute__(attr), memodict))
        cpyobj._graph = self._graph.copy()
        return cpyobj

    def __str__(self):
        msg = ""
        for n in self.nodes:
            u_nodes = [x[1] for x in self._graph.out_edges(n)]
            msg += f"{n: >2}: {self.nodes[n]['filter'].get_name()} -> {u_nodes} \n"
        return msg
