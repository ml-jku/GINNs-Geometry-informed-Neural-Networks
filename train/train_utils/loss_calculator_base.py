import torch

from models.point_wrapper import PointWrapper
from util.misc import idx_of_tensor_in_list
from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector,apply_gradient_vector



class LossCalculator:
    def __init__(self):
        self.sub_grad_field_dict = {}
        self.loss_weighted_dict = {}

    def compute_loss_and_save_sublosses(self):
        raise NotImplementedError("calculate_losses_field method must be implemented in the subclass")

    def adaptive_update(self):
        raise NotImplementedError("adaptive_update method must be implemented in the subclass")

    def backward(self, field_dict):
        # SCALAR LOSSES
        loss = sum(self.loss_weighted_dict.values())
        # possibly we only have TO losses which do not have grads, in this case we do not backpropagate
        if loss.requires_grad:
            # TODO: AR IMPORTANT: double-check if retain_graph=True is the right choice
            # we resample points each iteration, so it should be fine
            loss.backward() #retain_graph=True)
        
        # FIELD LOSSES
        # Here we need to be careful. Some of the y_fields have 2 grad_fields to backpropagate through.
        # We want to avoid retain_graph=True, as we would need to clean it up after the optimization step.
        # Solution: we search for unique y_field in field_dict, sum up the grad_fields and backpropagate through the y_field
        y_fields = []
        key_lists = []
        for key, y_field in field_dict.items():
            idx = idx_of_tensor_in_list(y_field, y_fields)
            if idx >= 0:
                key_lists[idx].append(key)
            else:
                y_fields.append(y_field)
                key_lists.append([key])
        #Chamer: (n_batch, n_chamer, 3) <--> (n_batch, n_to, 3)
        # Now we do the backward pass
        for y_field, key_list in zip(y_fields, key_lists):
            grad_field = sum([self.sub_grad_field_dict[key] for key in key_list])
            if not torch.equal(grad_field, torch.zeros_like(grad_field)):
                # only backpropagate if we have a non-zero grad_field, e.g. avoiding backpropagating through non-existing surface points    
                y_field.backward(gradient=grad_field)

    def backward_config(self, field_dict, network):
        grads = []
        network.zero_grad()
        # SCALAR LOSSES
        loss = sum(self.loss_weighted_dict.values())
        # possibly we only have TO losses which do not have grads, in this case we do not backpropagate
        if loss.requires_grad:
            # TODO: AR IMPORTANT: double-check if retain_graph=True is the right choice
            # we resample points each iteration, so it should be fine
            loss.backward() #retain_graph=True)
            grad = get_gradient_vector(network)
            network.zero_grad()
            if not torch.equal(grad, torch.zeros_like(grad)):
                grads.append(grad)
        
        # FIELD LOSSES
        # Here we need to be careful. Some of the y_fields have 2 grad_fields to backpropagate through.
        # We want to avoid retain_graph=True, as we would need to clean it up after the optimization step.
        # Solution: we search for unique y_field in field_dict, sum up the grad_fields and backpropagate through the y_field
        y_fields = []
        key_lists = []
        for key, y_field in field_dict.items():
            idx = idx_of_tensor_in_list(y_field, y_fields)
            if idx >= 0:
                key_lists[idx].append(key)
            else:
                y_fields.append(y_field)
                key_lists.append([key])
        #Chamer: (n_batch, n_chamer, 3) <--> (n_batch, n_to, 3)
        # Now we do the backward pass
        for y_field, key_list in zip(y_fields, key_lists):
            for key in key_list:
                grad_field = self.sub_grad_field_dict[key] 
                if not torch.equal(grad_field, torch.zeros_like(grad_field)):
                    # only backpropagate if we have a non-zero grad_field, e.g. avoiding backpropagating through non-existing surface points    
                    y_field.backward(gradient=grad_field, retain_graph=(len(key_list) > 1))
                    grads.append(get_gradient_vector(network))
                    network.zero_grad()

        grad_config = ConFIG_update(grads)
        apply_gradient_vector(network, grad_config)

    def batch_grad_field_sqrt_derivative(self, x_value_batch, x_field_grad_batch, scalar_batch):
        """
        assume x_field_grad_batch torch.Size([batch_size, n*m])
        assume scalar_batch torch.Size([batch_size])
        """
        if type(x_field_grad_batch) == torch.Tensor:
            #return torch.sum(0.5 * torch.sqrt(scalar_batch) * x_field_grad_batch / torch.sqrt(x_value_batch), dim=0)
            if torch.equal(x_value_batch, torch.zeros_like(x_value_batch)):
                return torch.zeros_like(x_field_grad_batch)
            
            res = 0.5 * torch.sqrt(scalar_batch) * x_field_grad_batch / torch.sqrt(x_value_batch)
            return res
            
        if type(x_field_grad_batch) == PointWrapper:
            res_list = []
            for i in range(x_field_grad_batch.bz):
                if torch.equal(x_value_batch, torch.zeros_like(x_value_batch)):
                    res_list.append(torch.zeros_like(x_field_grad_batch.pts_of_shape(i)))
                else:
                    res_list.append(0.5 * torch.sqrt(scalar_batch) * x_field_grad_batch.pts_of_shape(i) / torch.sqrt(x_value_batch))
            res = PointWrapper.create_from_pts_per_shape_list(res_list)
            return res.data

    def batch_grad_field_linear_derivative(self, x_value_batch, x_field_grad_batch, scalar_batch):
        """
        assume x_field_grad_batch torch.Size([batch_size, n*m])
        assume scalar_batch torch.Size([batch_size])
        """
        if type(x_field_grad_batch) == torch.Tensor:
            #return torch.sum(scalar_batch * x_field_grad_batch, dim=0)
            return scalar_batch * x_field_grad_batch
        
        if type(x_field_grad_batch) == PointWrapper:
            res_list = []
            for i in range(x_field_grad_batch.bz):
                res_list.append(scalar_batch * x_field_grad_batch.pts_of_shape(i))
            res = PointWrapper.create_from_pts_per_shape_list(res_list)
            return res.data
    
    def batch_grad_field_square_derivative(self, x_value_batch, x_field_grad_batch, scalar_batch):
        """
        assume x_field_grad_batch torch.Size([batch_size, n*m])
        assume scalar_batch torch.Size([batch_size])
        """
        #return torch.sum(2. * scalar_batch**2 * x_value_batch * x_field_grad_batch, dim=0)
        if type(x_field_grad_batch) == torch.Tensor:
            return 2. * scalar_batch**2 * x_value_batch * x_field_grad_batch
        
        if type(x_field_grad_batch) == PointWrapper:
            res_list = []
            for i in range(x_field_grad_batch.bz):
                res_list.append(2. * scalar_batch**2 * x_value_batch * x_field_grad_batch.pts_of_shape(i))
            res = PointWrapper.create_from_pts_per_shape_list(res_list)
            return res.data

