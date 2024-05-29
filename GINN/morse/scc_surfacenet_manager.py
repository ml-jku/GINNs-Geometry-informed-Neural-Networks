import logging
import torch

from GINN.morse.cp_helper import CriticalPointsHelper
from GINN.morse.surfacegraph_helper import SurfaceGraphHelper
from utils import set_and_true


class SCCSurfaceNetManager:

    def __init__(self, config, netp, mp_manager, plotter, timer_helper, p_sampler, device):
        self.logger = logging.getLogger('SCCSurfaceNetManager')
        self.config = config
        self.mpm = mp_manager
        self.plotter = plotter
        self.timer_helper = timer_helper
        self.p_sampler = p_sampler
        self.device = device
        
        self.cp_helper = CriticalPointsHelper(config, netp, mp_manager, plotter, timer_helper, device)
        self.surfgraph_helper = SurfaceGraphHelper(self.config, self.mpm, self.plotter, self.timer_helper)
        

    def get_surface_pts(self, z, epoch):
        success, p_surface = self.cp_helper.get_surface_pts(z, epoch)
        if not success:
            return None, None

        weights_surf_pts = torch.ones(len(p_surface)) / p_surface.data.shape[0]
        if set_and_true('reweigh_surface_pts_close_to_interface', self.config):
            dist = torch.min(torch.norm(p_surface.data[:, None, :] - self.x_interface[None, :, :], dim=2), dim=1)[0]
            dist = torch.clamp(dist, max=self.config['reweigh_surface_pts_close_to_interface_cutoff'])
            weights_surf_pts = torch.pow(dist, self.config['reweigh_surface_pts_close_to_interface_power'])
            weights_surf_pts = weights_surf_pts / weights_surf_pts.sum()  ## normalize to sum to 1

        self.mpm.plot(self.plotter.plot_shape_and_points, 'plot_surface_points', arg_list=[p_surface.detach().cpu(), 'Surface points'])
        return p_surface, weights_surf_pts

    def get_scc_pts_to_penalize(self, z, epoch):
        ## Critical points
        success, res_tup = self.cp_helper.get_critical_points(z, epoch, plot_helper=self.plotter,
                                                              timer_helper=self.timer_helper)
        if not success:
            return False, None

        (pc, pyc, edges_list, indexes) = res_tup
        # move to cpu
        pc_cpu, pyc_cpu, indexes_cpu = pc.cpu(), pyc.cpu(), indexes.cpu()
        edges_list = [edges.cpu() for edges in edges_list]
        indexes_list = [indexes_cpu[pc_cpu.get_idcs(i_shape)] for i_shape in range(self.config['batch_size'])]

        self.mpm.plot(self.plotter.plot_edges,'plot_edges', arg_list=[pc_cpu, indexes_list, edges_list, "Surface graph edges"])

        ## Surface graphs
        surfnet_graph_list = []
        with self.timer_helper.record('surfgraph_helper.construct_surfnet_graph'):
            for edges_cpu in edges_list:
                surfnet_graph_list.append(self.surfgraph_helper.construct_surface_net(edges_cpu, pc_cpu))

        with self.timer_helper.record('surfgraph_helper.get_sub_CC'):
            surfnet_graph_sub_CCs_list = self.surfgraph_helper.get_sub_CC(edges_list, pyc_cpu)
        self.mpm.plot(self.plotter.plot_CCs_of_surfnet_graph, 'plot_surfnet_graph', arg_list=[surfnet_graph_sub_CCs_list, pc_cpu.numpy(), pyc_cpu.numpy(), edges_list])

        with self.timer_helper.record('surfgraph_helper.get_cps_and_penalties_between_CCs'):
            p_penalize, p_penalties = self.surfgraph_helper.get_cps_and_penalties_between_CCs_list(surfnet_graph_list,
                                                                                              surfnet_graph_sub_CCs_list,
                                                                                              pc_cpu,
                                                                                              indexes_cpu, pyc_cpu,
                                                                                              plot_helper=self.plotter)

        if p_penalties.data.sum() == 0:
            return False, None

        self.mpm.plot(self.plotter.plot_shape_and_points, 'plot_penalty_points', arg_list=[p_penalize.cpu().numpy(), 'Penalty points'])

        p_penalize = p_penalize.to(self.device)
        p_penalties = p_penalties.to(self.device)

        if set_and_true('remove_penalty_points_outside_envelope', self.config):
            is_in_env_mask = self.p_sampler.is_inside_envelope(p_penalize.cpu().numpy())
            is_in_env_mask = torch.from_numpy(is_in_env_mask).to(torch.bool).to(self.device)
            self.logger.debug(
                f'penalize DCs with {is_in_env_mask.sum()} points inside envelope and {(~is_in_env_mask).sum()} points outside')
            p_penalize = p_penalize.select_w_mask(is_in_env_mask)
            p_penalties = p_penalties.select_w_mask(is_in_env_mask)
            if p_penalize.data.sum() == 0:
                self.logger.debug(f'No points left after removing points outside envelope')
                return False, None

            self.mpm.plot(self.plotter.plot_shape_and_points, 'plot_penalty_points_only_in_envelope',
                                          arg_list=[p_penalize.cpu().numpy(), 'Penalty points w/o points outside envelope'])

        return True, (p_penalize, p_penalties)