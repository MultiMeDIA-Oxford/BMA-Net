import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # class balance weight
        self.gamma = gamma # focusing parameter that reduces the loss of easy and confident samples within every class
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    # forward method is only compatible with PyTorch
    def forward(self, inputs, targets): # inputs (data) and targets (true labels) returned by DataLoader
        ce_loss = self.ce(inputs, targets)
        p_t = torch.exp(-ce_loss) # p_t is the predicted probability of the true label
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha = self.alpha[targets] # select the corresponding balance weight of every sample in targets
        else:
            alpha = self.alpha

        focal_loss = alpha * ((1 - p_t) ** self.gamma) * ce_loss # ce_loss = -ln(p_t)

        # loss computed for every sample, we can choose to return the mean, the sum or the array itself
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss