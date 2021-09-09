from abc import ABCMeta, abstractmethod, abstractproperty
from ..mnts_logger import MNTSLogger
from typing import Union, Iterable
import networkx as nx
from cachetools import cached, LRUCache

__all__ = ['MNTSFilterGraph', 'MNTSFilter', 'MNTSFilterPipeline']

class MNTSFilter(object):
    def __init__(self):
        r"""
        Base class of filter
        """
        self._logger = MNTSLogger[self.get_name()]

    @property
    @abstractmethod
    def filter(self, *args, **kwargs):
        pass

    def get_name(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        self.filter(*args, **kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}: \n\t" + \
               '\n\t'.join(["{: >10} - {}".format(item[0], item[1]) if item[0].find('logger') == -1 else ""
                            for item in vars(self).items()])


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

    @property
    def nodes(self):
        return self._graph.nodes

    @nodes.setter
    def nodes(self, nodes):
        r"""
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

    def _node_search(self, attr_key, attr_val):
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
        Add a node to the filter graph

        Args:
            node (MNTSFilter):
                The filter that is to be put into the list. Note that filter cannot be reused as it forms looped
                graphs that are trouble to deal with.
            upstream (MNTSFitler or int or list(MNTSFitler or int)):
                The upstream source of data of this node. Permit multiple upstream, however, one node cannot have
                more than one downstream.

        Returns:

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

    @cached(cache=LRUCache(maxsize=5))
    def _request_output(self, node_id):
        r"""
        Use for recursively request data from upstream node
        """
        in_edges = self._graph.in_edges(node_id)
        upstream_nodes = [t[0] for t in in_edges]

        cur_filter = self.nodes[node_id]['filter']
        if not node_id in self._entrance:
            data = [self._request_output(i) for i in upstream_nodes]
            self._logger.info(f"Finished step: {self._nodemap[node_id]}")
            return cur_filter.filter(*data)
        else:
            self._logger.info(f"Finished step: {self._nodemap[node_id]}")
            return cur_filter.filter(self._inputs[node_id])

    def plot_graph(self):
        print(self._nodemap)
        nx.draw(self._graph, with_labels=True)

    def execute(self, *args):
        assert len(args) == len(self._entrance), "Inputs do not align with output"

        self._inputs = {n: args[i] for i, n in enumerate(self._entrance)}

        # gather_output
        output = {}
        for i in self._exits:
            output[i] = self._request_output(i)
        return output
