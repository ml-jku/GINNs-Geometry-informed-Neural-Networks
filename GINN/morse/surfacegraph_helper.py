import copy
import logging

import networkx as nx
import numpy as np
import torch
from GINN.helpers.mp_manager import MPManager
from models.point_wrapper import PointWrapper

class SurfaceGraphHelper():

    def __init__(self, config, mp_manager: MPManager, plot_helper, timer_helper):
        self.config = config
        self.mpm = mp_manager
        self.plot_helper = plot_helper
        self.timer_helper = timer_helper
        self.logger = logging.getLogger('surfnet_helper')

    def construct_surface_net(self, edges_cpu, xc_cpu):
        # surfnet graph
        graph = nx.Graph()
        ## 2) Add graph nodes
        graph.add_nodes_from(range(len(xc_cpu)))
        ## 3) Add graph edges
        graph.add_edges_from(edges_cpu.tolist())
        return graph

    def get_sub_CC(self, edges_list, pyc_cpu):
        surfnet_graph_sub_CCs_list = []
        
        for i_shape, edges_cpu in enumerate(edges_list):
            
            if len(edges_cpu) == 0:
                surfnet_graph_sub_CCs_list.append(None)
                continue
            yc_cpu = pyc_cpu.pts_of_shape(i_shape)
            
            ## Sub-graph below the level-set
            sublevel_mask_nodes = yc_cpu < self.config['level_set']
            sublevel_mask_edges = sublevel_mask_nodes[edges_cpu].all(1)
            edges_sub = edges_cpu[sublevel_mask_edges]
            # fix dimensionality error if there is only one edge
            if len(edges_sub.shape) == 1:
                edges_sub = edges_sub.unsqueeze(0)
            nodes_sub = torch.where(sublevel_mask_nodes)[0]

            surfnet_graph_sub = nx.Graph()
            surfnet_graph_sub.add_nodes_from(nodes_sub.tolist())
            surfnet_graph_sub.add_edges_from(edges_sub.tolist())
            c_cs = [surfnet_graph_sub.subgraph(c).copy() for c in nx.connected_components(surfnet_graph_sub)]
            self.logger.debug("Sub-graph has ")
            for i, CC in enumerate(c_cs):
                self.logger.debug(f'component {i} with nodes {CC.nodes()} and edges {CC.edges()}')
                
            surfnet_graph_sub_CCs_list.append(c_cs)
        return surfnet_graph_sub_CCs_list

    def get_cps_and_penalties_between_CCs_list(self, surfnet_graph_list, surfnet_graph_sub_CCs_list, pc, indexes_all, pyc, **kwargs):
        
        w_graph_list = []
        x_neighbor_node_list = []
        penalties_list = []
        plot_dict_list = []
        edges_list = []
        edge_weights_list = []
        
        for i_shape in range(self.config['batch_size']):
            surfnet_graph = surfnet_graph_list[i_shape]
            surfnet_graph_sub_CCs = surfnet_graph_sub_CCs_list[i_shape]
            if surfnet_graph_sub_CCs is None:
                ## set default values for plotting
                ## TODO: could refactor from lists to dict for not having to set the default values
                w_graph_list.append(None)
                edges_list.append(None)
                edge_weights_list.append(None)
                x_neighbor_node_list.append(torch.zeros(0, device='cpu'))
                penalties_list.append(torch.zeros(0, device='cpu'))
                plot_dict_list.append({
                    'Intermediate weighted graph edges':{ 'indexes': None, 'edges': None, 'edge_weights': None},
                    'sub and neighbor weighted graph':{ 'indexes': None, 'edges': None, 'edge_weights': None, 'neighbor_nodes': None, 'penalties': None}
                })
                continue
            
            xc = pc.pts_of_shape(i_shape)
            indexes = indexes_all[pc.get_idcs(i_shape)]
            yc = pyc.pts_of_shape(i_shape)
            w_graph, neighbor_node_idcs, penalties, plot_state_dict = self.get_cps_and_penalties_between_CCs(surfnet_graph, surfnet_graph_sub_CCs, xc, indexes, yc, **kwargs)
            w_graph_list.append(w_graph)
            x_neighbor_node_list.append(xc[neighbor_node_idcs])
            plot_dict_list.append(plot_state_dict)
            edges_list.append(torch.tensor([list(t) for t in w_graph.edges()], device='cpu'))
            edge_weights_list.append([data['weight'] for _, _, data in w_graph.edges.data()])
            
            penalties = torch.tensor(penalties, device=xc.device)
            penalties = penalties / (penalties.sum() + self.config['scc_penalty_norm_eps'])
            penalties_list.append(penalties)
        
        if self.plot_helper.do_plot('plot_intermediate_wd_graph'):
            tmp_dicts_list = [d['Intermediate weighted graph edges'] for d in plot_dict_list]
            self.mpm.plot(kwargs['plot_helper'].plot_edges, 'plot_intermediate_wd_graph', [pc],
                                      kwargs_dict={
                                          'indexes_all': [d['indexes'] for d in tmp_dicts_list],
                                          'edges_list': [d['edges'] for d in tmp_dicts_list],
                                            'edge_weights': [d['edge_weights'] for d in tmp_dicts_list],
                                          'fig_label': "Intermediate weighted graph edges",
                                          'show_v_id': False,
                                      })

        if self.plot_helper.do_plot('plot_sub_and_neighbor_wd_graph') and 'plot_helper' in kwargs:
            tmp_dicts_list = [d['sub and neighbor weighted graph'] for d in plot_dict_list]
            self.mpm.plot(kwargs['plot_helper'].plot_edges,
                                        'plot_sub_and_neighbor_wd_graph',
                                        [pc],
                                        kwargs_dict={
                                            'indexes_all': [d['indexes'] for d in tmp_dicts_list],
                                            'edges_list': [d['edges'] for d in tmp_dicts_list],
                                            'edge_weights': [d['edge_weights'] for d in tmp_dicts_list],
                                            'pts_to_penalize': [d['neighbor_nodes'] for d in tmp_dicts_list],
                                            'penalties': [d['penalties'] for d in tmp_dicts_list],
                                            'fig_label': "sub and neighbor weighted graph",
                                            'show_v_id': False,
                                        })
        
        if self.plot_helper.do_plot('plot_wd_graph'):
            # with self.timer_helper.record('plot_helper.plot_edges (augm surfnet graph)'):``
            self.mpm.plot(kwargs['plot_helper'].plot_edges,
                                        'plot_wd_graph',
                                        [pc],
                                        kwargs_dict={
                                            'indexes_all': [indexes_all[pc.get_idcs(i_shape)] for i_shape in range(self.config['batch_size'])],
                                            'edges_list': edges_list,
                                            'edge_weights': edge_weights_list,
                                            'pts_to_penalize': x_neighbor_node_list,
                                            'penalties': penalties_list,
                                            'fig_label': "Weighted graph edges",
                                        })
        
        # convert to pointwrapper instances
        p_penalize = PointWrapper.create_from_pts_per_shape_list(x_neighbor_node_list)
        p_penalties = PointWrapper.create_from_pts_per_shape_list(penalties_list)
        return p_penalize, p_penalties
        

    def get_cps_and_penalties_between_CCs(self, surfnet_graph: nx.Graph, surfnet_graph_sub_CCs, xc, indexes, yc, **kwargs):
        '''
        algorithm
        - construct a weighted graph as follows
        - for all nodes within a CC
        -- edges within the same CC must be zero
        -- edges going outside of the shape ("outside neighbors") must be connected via the y-value of the node that it's connected to
        - for all nodes outside the shape
        -- connect to all other nodes outside the shape with euclidean distance (TODO: could maybe improve this by using CC of outside shapes or via a distance threshold)

        - for all "outside neighbors" of each CC
        -- find all shortest distances between all pairs of CCs
        -- choose this as a inverse-weight
        ---> O(n^2 * OC), where OC is the number of outside neighbors
        ---> O(n^3)

        :param indexes:
        :param surfnet_graph:
        :param surfnet_graph_sub_CCs:
        :param xc:
        :param yc:
        :return:
        '''
        xc = np.copy(xc)  # make a copy to not change the original
        neighbor_node_idcs = []
        first_nodes_of_CC = []
        plot_state_dict = {}

        # construct graph
        w_graph = nx.Graph()
        # add edges within the connected components with weight 0
        self.logger.debug(f'add sub-edges to w_graph')
        for i, CC in enumerate(surfnet_graph_sub_CCs):
            is_first_node_in_CC = True
            self.logger.debug(f'component {i} with nodes {CC.nodes()} and edges {CC.edges()}')
            for u in CC.nodes():
                if is_first_node_in_CC:
                    first_nodes_of_CC.append(u)
                    is_first_node_in_CC = False
                for v in surfnet_graph.neighbors(u):
                    if u == v:
                        continue
                    elif v in CC.nodes():
                        weight = 0
                    else:
                        weight = yc[v].item() - self.config['level_set']
                        neighbor_node_idcs.append(v)
                    self.logger.debug(f'add edge ({u},{v},{weight:0.2f})')
                    w_graph.add_edge(u, v, weight=weight)

        if self.plot_helper.do_plot('plot_intermediate_wd_graph') and 'plot_helper' in kwargs:
            plot_state_dict["Intermediate weighted graph edges"] = {
                # 'xc': xc,
                'indexes': indexes,
                'edges': torch.tensor([list(t) for t in w_graph.edges()], device='cpu'),
                'edge_weights': [data['weight'] for _, _, data in w_graph.edges.data()],
            }

        # add edges from all outside nodes to all others with euclidean distance
        super_mask_nodes = yc > self.config['level_set']
        nodes_super = torch.where(super_mask_nodes)[0]
        for u in nodes_super:
            for v in nodes_super:
                if u == v:
                    continue
                weight = np.linalg.norm(xc[u] - xc[v])
                w_graph.add_edge(u.item(), v.item(), weight=weight)

        # plot all connected components
        c_cs = [w_graph.subgraph(c).copy() for c in nx.connected_components(w_graph)]
        self.logger.debug("w_graph has the following CCs")
        for i, CC in enumerate(c_cs):
            self.logger.debug(f'component {i} with nodes {CC.nodes()} and edges {CC.edges()}')

        # iterate over all neighbor_nodes and find path lengths to all nodes in first_nodes
        neighbor_node_idcs = list(set(neighbor_node_idcs))
        penalties = []
        for n_node in neighbor_node_idcs:
            self.logger.debug(f'Neighbor node: {n_node}')
            # note: all pairs dijkstra calls single source dijkstra internally; so it's better to call single_source dijkstra directly
            # https://networkx.org/documentation/networkx-1.10/_modules/networkx/algorithms/shortest_paths/weighted.html#all_pairs_dijkstra_path_length
            path_lengths_dict = nx.single_source_dijkstra_path_length(w_graph, source=n_node, weight='weight')
            penalty = 0
            prior_path_lengths = []
            for cc_1_node in first_nodes_of_CC:
                if cc_1_node in path_lengths_dict:
                    path_len_to_cur_cc = path_lengths_dict[cc_1_node]
                else:
                    path_len_to_cur_cc = 1.0 / self.config['graph_algo_inv_dist_eps']
                    
                # iterate over all previous nodes, to get all pairs of cc_nodes
                for i, prior_length in enumerate(prior_path_lengths):
                    total_path_len = path_len_to_cur_cc + prior_length
                    delta_penalty = 1 / (total_path_len ** 2 + self.config['graph_algo_inv_dist_eps'])
                    self.logger.debug(f'CC {cc_1_node} <--> {first_nodes_of_CC[i]}: {total_path_len:0.2f} / {delta_penalty:00.2f}')
                    penalty += delta_penalty
                prior_path_lengths.append(path_len_to_cur_cc)
            # TODO: maybe take nth root for geometric mean or harmonic mean
            penalties.append(penalty)

        for i, n_node in enumerate(neighbor_node_idcs):
            self.logger.debug(f'Neighbor node {n_node} has penalty: {penalties[i]:0.2f}')

        if self.plot_helper.do_plot('plot_sub_and_neighbor_wd_graph') and 'plot_helper' in kwargs:
            super_graph = w_graph.subgraph(nodes_super.numpy().tolist())
            w_graph_simple = copy.deepcopy(w_graph)
            w_graph_simple.remove_edges_from(super_graph.edges())
            plot_state_dict["sub and neighbor weighted graph"] = {
                'xc': xc,
                'indexes': indexes,
                'edges': torch.tensor([list(t) for t in w_graph.edges()], device='cpu'),
                'edge_weights': [data['weight'] for _, _, data in w_graph_simple.edges.data()],
                'neighbor_nodes': [xc[idx] for idx in neighbor_node_idcs],
                'penalties': penalties
            }

        return w_graph, neighbor_node_idcs, penalties, plot_state_dict




    def get_super_CC(self, edges, yc):
        ## Super-graph below the level-set
        superlevel_mask_nodes = yc > self.config['level_set']
        superlevel_mask_edges = superlevel_mask_nodes[edges].all(1)
        edges_super = edges[superlevel_mask_edges]
        nodes_super = torch.where(superlevel_mask_nodes)[0]

        surfnet_graph_super = nx.Graph()
        surfnet_graph_super.add_nodes_from(nodes_super.tolist())
        surfnet_graph_super.add_edges_from(edges_super.tolist())
        c_cs = [surfnet_graph_super.subgraph(c).copy() for c in nx.connected_components(surfnet_graph_super)]
        self.logger.debug("Super-graph has ")
        for i, CC in enumerate(c_cs):
            self.logger.debug(f'component {i} with nodes {CC.nodes()} and edges {CC.edges()}')
        return c_cs

    def get_common_children(self, surfnet_graph, surfnet_graph_sub_CCs):
        common_children = set()
        ijs = np.triu_indices(len(surfnet_graph_sub_CCs), 1)
        for i, j in zip(*ijs):
            CC0 = surfnet_graph_sub_CCs[i]
            CC1 = surfnet_graph_sub_CCs[j]
            # self.logger.debug(f'{CC0.nodes()} X {CC1.nodes()}')
            for node0 in CC0.nodes():
                for node1 in CC1.nodes():
                    common_child = set(surfnet_graph.neighbors(node0)) & set(surfnet_graph.neighbors(node1))
                    if len(common_child):
                        self.logger.debug(f'{node0} /\\ {node1} = {common_child}')
                        common_children = common_children.union(common_child)
        self.logger.debug(f"Saddles to minimize: {common_children}")
        return common_children






