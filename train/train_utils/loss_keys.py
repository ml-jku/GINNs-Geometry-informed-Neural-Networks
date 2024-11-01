class LossKey:
    def __init__(self, key: str):
        self.base_key = key
        self.mu_key = f"mu_{key}"
        self.nu_key = f"nu_{key}"
        self.lambda_key = f"lambda_{key}"
        self.lagrangian_key = f"lagrangian_{key}"
        self.loss_key = f"loss_{key}"
        self.loss_unweighted_key = f"loss_unweighted_{key}"

    def __repr__(self):
        return f"LossKey(base_key={self.base_key})"
    
    def __eq__(self, other):
        """Overrides the default == implementation"""
        if isinstance(other, LossKey):
            return self.base_key == other.base_key
        return False

    

def get_loss_key_list(config: dict)-> list:
    loss_keys = []
    for key, value in config.items():
        if 'lambda' in key and value>0.:
            loss_keys.append(LossKey(key.replace('lambda_', '')))
    return loss_keys

