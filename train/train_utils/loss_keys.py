class LossKey:
    def __init__(self, key: str):
        self.base_key = key
        self.mu_key = f"mu_{key}"
        self.nu_key = f"nu_{key}"
        self.lagrangian_key = f"lagrangian_{key}"
        self.lambda_key = f"lambda_{key}"
        self.loss_key = f"loss_{key}"
        self.loss_unweighted_key = f"loss_unweighted_{key}"

    def __repr__(self):
        return f"LossKey(base_key={self.base_key})"
    
    def __eq__(self, other):
        """Overrides the default == implementation"""
        if isinstance(other, LossKey):
            return self.base_key == other.base_key
        return False
    
    # for hashing
    def __hash__(self):
        return hash(self.base_key)

    

def get_loss_keys_and_lambdas(config: dict)-> list:
    scalar_loss_keys = []
    field_loss_keys = []
    field_losses = ['lambda_'+key for key in config['field_losses']]
    lambda_dict = {}
    
    for key, value in config.items():
        if 'lambda' not in key or value <= 0.:
            continue
        
        lkey = LossKey(key.replace('lambda_', ''))
        if key in field_losses:
            field_loss_keys.append(lkey)
        else:
            scalar_loss_keys.append(lkey)
        lambda_dict[lkey] = value
    
    obj_key = _get_objective_key(config)
    if obj_key.base_key == 'null':
        lambda_dict[obj_key] = 1
    
    return scalar_loss_keys, field_loss_keys, lambda_dict, obj_key


def _get_objective_key(config):
    assert config.get('max_'+config['objective'], 0) == 0, f'Max value for objective loss not allowed in config'
    objective = config['objective']
    assert objective == 'null' or config.get('lambda_'+objective, 0) > 0, f'Weighting for objective loss not found in config or not positive'
    return LossKey(objective)