from torch.optim import AdamW

def optimizer(model, lr, weight_decay):
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
    return optimizer

def optimizer_update(optimizer, model, lr, weight_decay):
    existing_parameters = {p for group in optimizer.param_groups for p in group['params']}
    new_parameters = [p for p in model.parameters() if p.requires_grad and p not in existing_parameters]
    optimizer.add_param_group({'params': new_parameters, 'lr': lr, 'weight_decay': weight_decay}) # optionally set a different lr to train early layers
    return optimizer