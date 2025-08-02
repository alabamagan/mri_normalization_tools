import copy
import operator
import pprint
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union
from threading import Lock

import SimpleITK as sitk
import networkx as nx
import numpy as np
import yaml
from cachetools import LRUCache, cachedmethod
from functools import wraps

from ..mnts_logger import MNTSLogger

from ..filters import geom, intensity, MNTSFilter, MNTSFilterRequireTraining
from .data_node import *
from ..filters.geom import *
from ..filters.intensity import *

_avail_filters = []
_avail_filters.extend(dir(geom))
_avail_filters.extend(dir(intensity))

__all__ = ['MNTSFilterGraph']

def _update_progress(func):
    """Decorator to update progress bar after function call."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        with self._thread_lock:
            if self.progress_bar is not None:
                self.progress_bar.update(1)
        return result

    return wrapper


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
        self._output = {}
        self._nodes_upstream = {}
        self._nodes_cache = LRUCache(maxsize=8)
        self._thread_lock = Lock()
        # This will be set if multi-thread is used
        self.progress_bar = None

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

    @property
    def requires_training(self):
        if not len(self._nodemap):
            self._logger.warning("Trying to access `requires_training` before initializing the model!")
            return None
        return any(isinstance(f, MNTSFilterRequireTraining) for f in self)

    @staticmethod
    def CreateGraphFromYAML(yaml_file: Union[Path, str]):
        r"""

        Args:
            yaml_file (Path or str):
                Text file containing

        .. codeblock::

        """
        _logger = MNTSLogger['MNTSFilterGraph']
        yaml_file = Path(yaml_file)
        if not yaml_file.exists():
            raise IOError("Cannot open transform file.")

        # Read file
        with open(yaml_file, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        _logger.info(f"Creating transform from content:\n {data_loaded}")

        graph = MNTSFilterGraph()
        steps = []
        for key in data_loaded:
            # Check if the specified filter exist
            try:
                _filter_class = eval(key)
            except AttributeError:
                _logger.exception(f"Cannot load filter: {key}")
                _logger.error(f"{key} is not in the available list {_avail_filters}")

            # Get and parse the attributes

            _content = data_loaded.get(key, None)
            _logger.debug(f"YAML content {_content = }")
            if _content is None:
                steps.append(_filter_class())
            else:
                # For list, there could be both args and kwargs.
                if isinstance(_content, list):
                    _ext_kwargs = _content.pop('_ext', {})
                    _logger.debug(f"{_ext_kwargs = }")
                    _args = [i for i in _content if not isinstance(i, dict)]
                    _kwargs = [i for i in _content if isinstance(i, dict)]
                    _kwargs = {} if len(_kwargs) == 0 else _kwargs[0]
                    graph.add_node(_filter_class(*_args, **_kwargs), **_ext_kwargs)

                # If its just a dict, its kwargs
                elif isinstance(_content, dict):
                    _ext_kwargs = _content.pop('_ext', {})
                    _logger.debug(f"{_ext_kwargs = }")
                    _filter = _filter_class(**_content)
                    graph.add_node(_filter, **_ext_kwargs)

        # check if result has exit
        if len(graph._exits) == 0:
            msg = "Your graph has no exit node and will do nothing."
            raise ArithmeticError(msg)
        return graph


    def _node_search(self,
                     attr_key: str,
                     attr_val: Any) -> List[int]:
        r"""
        Search for node where its attribute of key:`attr_key` equal `attr_val`. There could be multiple nodes that
        fits the search so output is always a list unless no node is find, in such case -1 is returned.

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
                 is_exit: bool =False) -> None:
        r"""
        Add a node to the filter graph and connect it to the upstream nodes. If no upstream nodes are provided this
        node is automatically treated as the input node.

        Exit nodes, where marks the end of the normalization pipeline must be labeled using `is_exit` argument.

        .. note::
            The order of `upstream` is important and decides the argument order when inputs are required from the
            upstream in some filters, such as `NyulNormlize`

        Args:
            node (MNTSFilter):
                The filter that is to be put into the list. Note that filter cannot be reused as it forms looped
                graphs that are trouble to deal with.
            upstream (MNTSFitler or int or list(MNTSFitler or int)):
                The upstream source of data of this node. Permit multiple upstream, however, one node cannot have
                more than one downstream.

        """
        assert isinstance(node, MNTSFilter), f"Wrong input type: {node}"
        self._logger.info(f"Adding node {node} to the filter graph.")
        self._logger.debug(f"Args: {node = }, {upstream = }, {is_exit = }")
        self._logger.debug(f"ArgsTypes: {type(node) = }, {type(upstream) = }, {type(is_exit) = }")

        # if there's no other node with a filter key associates with this current node (i.e., this node has been added
        # before), add this node to the graph as a new one.
        if self._node_search('filter', node) == -1:
            _node_index = self._graph.number_of_nodes()
            self._graph.add_node(_node_index, filter=node)
            self._nodemap[_node_index] = node.get_name()
            if is_exit:
                self._logger.debug(f"Node: {node} added as exits")
                self._exits.append(_node_index)
                self._logger.debug(f"{self._exits = }")
        else:
            # Otherwise, just add the node index because the node is already here
            _node_index = self._node_search('filter', node)[0]

        if upstream is None:
            self._entrance.append(_node_index)
            self._nodes_upstream[_node_index] = []

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
            self._graph.add_edge(upstream, _node_index)
            self._nodes_upstream[_node_index] = tuple([upstream])

        elif isinstance(upstream, (list, tuple)):
            for us in upstream:
                self._graph.add_edge(us, _node_index)
            # Store the upstream nodes in correct order (`in_edges` method doesn't retain edge order after `deep_copy`)
            self._nodes_upstream[_node_index] = tuple(upstream)

    @cachedmethod(operator.attrgetter('_nodes_cache'))
    def _request_output(self,
                        node_id: int) -> Any:
        r"""
        Use a bottom up request to walk the graph finding the paths and then generate the output. Request are
        recursively requested to get data from the upstream.

        .. note::
            `in_edges` method did not always preserve the order of edges by when they were added.
            E.g., if you add edge (2, 1) then (0, 1), and then call `in_edges`, it could return
            [(0, 1), (2, 1)] sometimes. This is easy to happen when the _graph is copied using the
            `deepcopy` method. However, the flow of inputs depends on this order, so another method
            is used to find the path and the edges dring input.
        """
        upstream_nodes = self._nodes_upstream[node_id]

        cur_filter = self.nodes[node_id]['filter']
        if not node_id in self._entrance:
            # If not an entrance node, require all inputs from upstream nodes first, than generate output for downstream
            data = [self._request_output(i) for i in upstream_nodes]
            self._logger.info(f"Executing step: {self._nodemap[node_id]}")
            out = cur_filter(*data)
        else:
            # If it an entrance node, acquire input from dictionary.
            self._logger.info(f"Executing step: {self._nodemap[node_id]}")
            out = cur_filter(self._inputs[node_id])

        # Put this result into output cache because cachetools to save some time.
        if node_id in self._exits:
            self._output[node_id] = out
        return out

    def plot_graph(self) -> None:
        r"""
        Plot the filter graph. This function requires `netgraph` package. Input nodes are labelled in green and output
        nodes are labeled in blue.

        .. note::
            Although the graph labels tries to avoids edges and node, it still overlap quite often.

        Returns:
            None
        """
        from netgraph import Graph
        # nx.draw(self._graph, with_labels=True)

        nodes_color = {}
        for n in self.nodes:
            if n in self._entrance:
                nodes_color[n] = 'tab:green'
            elif n in self._exits:
                nodes_color[n] = 'tab:blue'
            else:
                nodes_color[n] = 'white'

        nodes_label = {n: f'{n}: {self._nodemap[n]}' for n in self._nodemap}

        try:
            # In `netgraph` version < 4.0.5, there is a problem with long distance edges in Sugiyama, solved in 4.0.5
            Graph(graph=self._graph, arrows=True, node_layout='dot', node_labels=nodes_label,
                  node_label_fontdict={'size':9}, node_label_offset=0.1, node_color=nodes_color)
        except AttributeError:
            # Fall back to spring arrangement
            msg = f"Cannot plot using Sugiyama layout, retreating to spring."
            self._logger.warning(msg, no_repeat=True)
            Graph(graph=self._graph, arrows=True, node_layout='spring', node_labels=nodes_label,
                  node_label_fontdict={'size': 8}, node_label_offset=0.1, node_color=nodes_color)

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
        assert len(args) == len(self._entrance), \
            f"Inputs ({len(args)}) do not align with number of entrance ({len(self._entrance)})."

        self._inputs = {n: args[i] for i, n in enumerate(self._entrance)}

        # Clear cache to avoid messing the results
        self._output.clear()
        self._nodes_cache.clear()

        # gather_output
        if force_request is None:
            # Later exit nodes has more chance to walk through all exit nodes. This is good for caching.
            for i in reversed(self._exits):
                if i in self._output:
                    continue
                else:
                   self._output[i] = self._request_output(i)
        else:
            if isinstance(force_request, MNTSFilter):
                force_request = self._node_search('filter', force_request)
            self._output[force_request] = self._request_output(force_request)
        # Clear cache
        return self._output


    def close_progress_bar(self):
        """Closes the tqdm progress bar."""
        if self.progress_bar is not None:
            self.progress_bar.close()

    @_update_progress
    def mpi_execute(self,
                    output_prefix: Iterable[str],
                    output_directory: Union[str, Path, List[Union[str, Path]], Dict],
                    *args,
                    **kwargs) -> None:
        r"""
        For use with :func:`mnts.filters.mpi_wrapper`. This copy the MNTSFilterGraph object and call the `execute`
        method on different inputs. Note that this should be called AFTER the states of the filters are loaded.


        The outputs folder structure:

            output_directory_exitnode_1/
            ├── output_prefix_1.nii.gz
            ├── output_prefix_2.nii.gz
            └── ...
            output_directory_exitnode_2/
            ├── output_prefix_1.nii.gz
            ├── output_prefix_2.nii.gz
            └── ...

        .. note::
            Use this with :func:`mpi_wrapper` otherwise, just use execute to save some memory

        See Also:
            * :method:`execute`
            * :

        Args:
            output_prefix (list of str):
                A list of names that will be used to name the outputs.
            output_directory (list of str):
                A list of directories corresponds to the number of output nodes in the graph.
            *args:
                Arguements that will be passed to :func:`execute`.

        Returns:
            None

        Example:
        >>> from pathlib import Path
        >>> # Assume MNTSFilterGraph is properly imported and initialized
        >>> filter_graph = MNTSFilterGraph()
        >>>
        >>> # Example output prefixes and directories
        >>> output_prefixes = ["output_1", "output_2"]
        >>> output_directories = [
        >>>     Path("output_directory_exitnode_1"),
        >>>     Path("output_directory_exitnode_2"),
        >>> ]
        >>>
        >>> # Example inputs for the entrance nodes
        >>> input_data_1 = ...  # some input data
        >>> input_data_2 = ...  # some other input data
        >>>
        >>> # Execute the function
        >>> filter_graph.mpi_execute(
        >>>     output_prefixes,
        >>>     output_directories,
        >>>     input_data_1,
        >>>     input_data_2
        >>> )
        """
        #!! Might get some memory issue here, but should be managible.
        # Create a copy of this class to separate
        cls_obj = copy.deepcopy(self)

        if 'force_request' in kwargs:
            output_directory = {kwargs['force_request']: output_directory}
            res = cls_obj.execute(*args, **kwargs)
        else:
            # check inputs are properly specified.
            if isinstance(output_directory, (str, Path)):
                output_directory = [output_directory]
            # If `output_directory` is list, tuple
            if isinstance(output_directory, (list, tuple)):
                if len(output_directory) != len(cls_obj._exits):
                    if len(output_directory) == 1:
                        _od = Path(output_directory[0])
                        output_directory = {v: _od.joinpath(self.nodes[v]['filter'].get_name()) for v in self._exits}
                    else:
                        raise IndexError("The lenth of output directories specified do not match the number of "
                                         f"exit nodes: {output_directory = }")
                else:
                    output_directory = {v: output_directory[n] for n, v in enumerate(self._exits)}
            elif isinstance(output_directory, dict):
                if list(output_directory.keys()) != list(self._exits):
                    msg = f"Keys of specified output does not match the exist nodes. " \
                          f"Got: {output_directory.keys()} and {self._exits}"
                    raise IndexError(msg)

            try:
                if cls_obj.check_output(output_directory, output_prefix):
                    self._logger.info(f"All outputs exist for input {args}, skipping.")
                    return 0
                else:
                    res = cls_obj.execute(*args, **kwargs)
            except Exception as e:
                self._logger.exception(f"Got unexpected error {e} for input: {pprint.pformat(args)}")
                return 1

        exit_nodes = cls_obj._exits if not 'force_request' in kwargs else [kwargs['force_request']]
        for n in exit_nodes:
            out_d = output_directory[n]
            if not out_d.exists():
                out_d.mkdir(parents=True, exist_ok=True)

            # how to name the output?
            out_im = res[n]
            out_im_name = out_d.joinpath(output_prefix).with_suffix('.nii.gz').resolve().__str__()
            cls_obj._logger.info(f"Writing to output {out_im_name}")
            sitk.WriteImage(out_im, str(out_im_name))
        return 0

    def set_progress_bar(self, total_tasks):
        """Sets the progress bar for the graph."""
        self.progress_bar = tqdm(total=total_tasks, desc="Processing", unit="task")

    def check_output(self, output_directory, output_prefix):
        r"""
        Return 1 if all output files exists, and 0 if any is missing.
        """
        targets = []
        for n in self._exits:
            out_d = output_directory[n]
            out_im_name = out_d.joinpath(output_prefix).resolve().__str__()
            targets.append(Path(out_im_name))
        return all([t.exists() for t in targets])

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
            nodelist (list of int or MNTSFilter, Optional):
                The (list of) nodes that require training. The output from their upstream(s) are collected and
                computed and them put into the corresponding locations stated above. If None locate and train all nodes
                that are child of MNTSFilterRequireTraining
            output_prefix (str):
                Name for the output. They are saved as `{output_prefix}`_[1,2,3...].nii.gz. Default to "output".
            output_directory (str or Path):
                Specify where to store the output from the upstream nodes.
            *args:
                See `:method:self.Execute`.
            **kwargs:

        Returns:
            None
        """
        temp_dir = Path(output_directory).absolute()
        # if nodelist is not provided, train all nodes that require training
        if nodelist is None:
            # starts from the last node that needs training
            nodelist = list(np.argwhere([isinstance(f, MNTSFilterRequireTraining) for f in self]).flatten()[::-1])

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
                try:
                    out_name = u_node_dir.joinpath(f"{output_prefix}")
                    # skip if it exist:
                    if out_name.is_file():
                        self._logger.info(f"File: {str(out_name)} already exist, skipping...")
                        continue
                    out = cls_obj.execute(*args, force_request=u_node)
                    sitk.WriteImage(out[u_node], str(out_name.with_suffix('.nii.gz')))
                except Exception as e:
                    self._logger.exception(f"Error during the processing of input: {args}, for node {n}. Original error"
                                           f" message is {e}")

    def train_node(self,
                   nodelist: List[Union[int, MNTSFilter]],
                   training_inputs: Union[str, Path],
                   save_dir: Union[str, Path]) -> None:
        r"""
        This method trains the selected node(s). Must call `prepare_training_files` before if this node is not an
        entrance node. If it is an entrance node, train the node separately it self. Contrary, you can also make use of
        the `DataNode` to make this api work.

        The trained states will be saved by passing the path:
            Path(save_dir).joinpath([node_index]_[node_name])

        Args:
            nodelist (list):
                The list of nodes to train. If `None` is specified, all nodes that are `MNTSFilterRequireTraining`
                instances will be trained.
            training_inputs (Path or str):
            save_dir (Path or str):

        See Also:
            :class:`MNTSFilterRequireTraining`

        """
        input_path = Path(training_inputs).resolve()
        assert input_path.is_dir(), f"Cannot open training inputs at {input_path.__str__()}"

        # if nodelist is not provided, train all nodes that require training
        if nodelist is None:
            # starts from the last node that needs training
            nodelist = list(np.argwhere([isinstance(f, MNTSFilterRequireTraining) for f in self]).flatten()[::-1])

        if not isinstance(nodelist, (list, tuple)):
            nodelist = [nodelist]

        self._logger.info(f"Start training nodes.")

        # Collect list of training inputs first.
        for n in nodelist:
            trained_node = self._node_search('filter', n) if isinstance(n, MNTSFilter) else n
            trained_node_filter = self.nodes[n]['filter']
            trained_node_name = f"{n}_" + trained_node_filter.get_name()
            self._logger.info(f"Training {trained_node_name}")
            # check if the target node is trainable
            if not isinstance(trained_node_filter, MNTSFilterRequireTraining):
                raise ArithmeticError(f"Specified node {trained_node_name} is not trainable.")

            trained_node_files_dir = input_path.joinpath(trained_node_name)
            if not trained_node_files_dir.is_dir():
                msg = f"Cannot open directory for training node {trained_node_name} at: " \
                      f"{trained_node_files_dir.resolve().__str__()} \n" \
                      f"Have you ran prepare_training_files()?"
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
                          f"{u_node_dir.resolve().__str__()}\n" \
                          f"Have you ran prepare_training_files()?"
                    raise IOError(msg)

                # append the gloobed files.
                u_nodes_files.append([str(r.resolve()) for r in u_node_dir.iterdir()
                                      if r.name.find('nii') != -1])
            trained_node_filter.train(*u_nodes_files)
            trained_state_path = save_dir.joinpath(trained_node_name)
            trained_node_filter.save_state(trained_state_path)
            self._logger.info(f"Saving the state for {trained_node_name} to:"
                              f" {str(trained_state_path)}")

    def load_node_states(self,
                         nodelist: List[Union[int, MNTSFilter]],
                         save_dir: Union[str, Path]) -> None:
        """Loads the saved states for specified nodes.

        This method loads the trained states for nodes specified in the `nodelist`
        from the given directory or file path.

        Args:
            nodelist (List[Union[int, MNTSFilter]]):
                A list of nodes for which the states need to be loaded. If `None`,
                it defaults to all nodes that require training.
            save_dir (Union[str, Path]):
                The path to the directory or file containing the saved states.

        Raises:
            IOError: If the specified `save_dir` does not exist.
            ArithmeticError: If a specified node is not trainable.

        """
        save_dir = Path(save_dir)
        if not save_dir.exists() :
            raise IOError(f"Cannot open directory/file to load the states, got {save_dir}")
        if nodelist is None:
            # starts from the last node that needs training
            nodelist = list(np.argwhere([isinstance(f, MNTSFilterRequireTraining) for f in self]).flatten()[::-1])
        if not isinstance(nodelist, (list, tuple)):
            nodelist = [nodelist]

        self._logger.info(f"Loading state from {save_dir}.")
        for n in nodelist:
            trained_node = self._node_search('filter', n) if isinstance(n, MNTSFilter) else n
            trained_node_filter = self.nodes[n]['filter']
            trained_node_name = f"{n}_" + trained_node_filter.get_name()
            # check if the target node is trainable
            if not isinstance(trained_node_filter, MNTSFilterRequireTraining):
                raise ArithmeticError(f"Specified node {trained_node_name} is not trainable.")

            self._logger.info(f"Loading state for: {trained_node_name}")
            if save_dir.is_dir():
                trained_node_filter.load_state(save_dir.joinpath(trained_node_name))
            elif save_dir.is_file():
                trained_node_filter.load_state(save_dir)

    def __deepcopy__(self, memodict={}):
        r"""
        Note that every thing is copied except for the logger and the graph because deepcopying with the graph will
        make this object unable to be pickled by multi-processing APIs.

        This might be solvable through implementing __deepcopy__ for the filters.
        """
        cpyobj = type(self)()
        attr_copy = ['_entrance', '_exits', '_nodemap', '_nodes_upstream']
        for attr in attr_copy:
            cpyobj.__setattr__(attr, deepcopy(self.__getattribute__(attr), memodict))
        cpyobj._graph = self._graph.copy()
        return cpyobj

    def __str__(self):
        msg = "{:=^25}\n".format("Graph Structure")
        for n in self.nodes:
            u_nodes = [x[1] for x in self._graph.out_edges(n)]
            msg += f"{n: >2}: {self.nodes[n]['filter'].get_name()} -> {u_nodes} \n"
        msg += "{:=^50}\n".format("Filter Details")
        for n in self.nodes:
            msg += str(self.nodes[n]['filter']) + '\n\n'
        return msg

    def __iter__(self):
        r"""
        As an iterator, this class return the filters based on the sequence they are added.
        """
        for i in self.nodes:
            yield self.nodes[i]['filter']