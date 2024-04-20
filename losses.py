import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
    
BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"

class ModifiedBCELoss(torch.nn.Module):
    def __init__(self, beta = 0.5, eps = 1e-7, from_logits = True):
        torch.nn.Module.__init__(self)
        self.beta = beta
        self.eps = eps
        self.from_logits = from_logits
        
    def forward(self, preds, labels):
        if self.from_logits:
            preds = F.logsigmoid(preds).exp()
        loss = self.beta * (labels * torch.log(preds + self.eps)) + (1 - self.beta) * ((1 - labels) * torch.log(1 - preds + self.eps))
        loss = -torch.mean(loss)
        return loss
    
class ComboLoss(ModifiedBCELoss, smp.losses.DiceLoss):
    def __init__(self, mode, from_logits = True, eps = 1e-7, beta = 0.5, alpha = 0.5):
        ModifiedBCELoss.__init__(self, beta=beta, eps=eps, from_logits=from_logits)
        smp.losses.DiceLoss.__init__(self, mode=mode, from_logits=from_logits, eps=eps)
        self.beta = beta
        self.alpha = alpha

    def forward(self, preds, labels):
        bce = ModifiedBCELoss.forward(self, preds = preds, labels = labels)
        dice = smp.losses.DiceLoss.forward(self, y_pred = preds, y_true=labels)
        return self.alpha * bce - (1- self.alpha) * (-dice + 1)

class ModifiedFocalLoss(torch.nn.Module):
    def __init__(self, delta = 1.0, gamma = 1.0, from_logits = True):
        torch.nn.Module.__init__(self)
        self.delta = delta
        self.gamma = gamma
        self.from_logits = from_logits
    
    def forward(self, preds, labels):
        og_preds = preds
        if self.from_logits:
            preds = F.logsigmoid(preds).exp()
        p_t = (labels * preds + (1-labels) * (1-preds))
        bce = F.binary_cross_entropy_with_logits(og_preds, labels.float(), reduction="none")
        loss = self.delta * (1.0 - p_t)**(1-self.gamma) * bce
        loss = loss.mean()
        return loss
        
class ModifiedFocalTverskyLoss(torch.nn.Module):
    def __init__(self, delta = 0.5, gamma = 1.0, from_logits = True, threshold = 0.5, eps = 1e-7):
        torch.nn.Module.__init__(self)
        self.delta = delta
        self.gamma = gamma
        self.from_logits = from_logits
        self.threshold = threshold
        self.eps = eps
    
    def forward(self, preds, labels):
        if self.from_logits:
            preds = F.logsigmoid(preds).exp()
        preds = (preds > self.threshold).float()
        tp = torch.sum(preds * labels)
        fp = torch.sum(preds * (1 - labels))
        fn = torch.sum((1 - preds) * labels)
        tversky_score = tp / (tp + self.delta * fp + (1-self.delta) * fn + self.eps)
        loss = (1 - tversky_score)**self.gamma
        return loss

class SymmetricUnifiedFocalLoss(ModifiedFocalLoss, ModifiedFocalTverskyLoss):
    def __init__(self, delta = 0.6, gamma = 0.5, lambda_ = 0.5, from_logits = True, threshold = 0.5, eps = 1e-7):
        ModifiedFocalLoss.__init__(self, delta = delta, gamma = gamma, from_logits = from_logits)
        ModifiedFocalTverskyLoss.__init__(self, delta = delta, gamma = gamma, from_logits = from_logits, threshold = threshold, eps = eps)
        self.lambda_ = lambda_
        
    def forward(self, preds, labels):
        cross_entropy = ModifiedFocalLoss.forward(self, preds = preds, labels = labels)
        tversky = ModifiedFocalTverskyLoss.forward(self, preds = preds, labels = labels)
        loss = self.lambda_ * cross_entropy + (1 - self.lambda_) * tversky
        return loss