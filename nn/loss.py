from core.tensor import Tensor
from nn.module import BaseModule

class MSELoss(BaseModule):
    def forward(self, y_pred, y_true):
        from losses.regression import MSE
        return MSE(y_true, y_pred)

class CrossEntropyLoss(BaseModule):
    def forward(self, logits, targets):
        from losses.classification import ce_loss
        return ce_loss(targets, logits)