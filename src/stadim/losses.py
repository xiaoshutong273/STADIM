import torch
import torch.nn as nn

# ===================== Loss Functions =======================
class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()
        self.eps = 1e-10

    def forward(self, x_raw, disp, mean, scale_factor):
        scale_factor = scale_factor.unsqueeze(1)
        mean = mean * scale_factor

        t1 = torch.special.gammaln(disp + self.eps) + torch.special.gammaln(x_raw + 1.0) - torch.special.gammaln(x_raw + disp + self.eps)
        t2 = (disp + x_raw) * torch.log1p(mean / (disp + self.eps)) + \
             (x_raw * (torch.log(disp + self.eps) - torch.log(mean + self.eps)))

        result = t1 + t2
        return torch.mean(result)

class ZINBLoss(nn.Module):
    def __init__(self, ridge_lambda=0.0):
        super(ZINBLoss, self).__init__()
        self.eps = 1e-10
        self.ridge_lambda = ridge_lambda

    def forward(self, x_raw, pi, disp, mean, scale_factor):
        # mean: unscaled output, scale_factor: cell-wise
        scale_factor = scale_factor.unsqueeze(1)  # shape (n_cells, 1)
        mean = mean * scale_factor  # broadcasting to (n_cells, n_genes)

        t1 = torch.special.gammaln(disp + self.eps) + torch.special.gammaln(x_raw + 1.0) - torch.special.gammaln(x_raw + disp + self.eps)
        t2 = (disp + x_raw) * torch.log1p(mean / (disp + self.eps)) + \
             (x_raw * (torch.log(disp + self.eps) - torch.log(mean + self.eps)))
        nb_case = t1 + t2 - torch.log(1.0 - pi + self.eps)

        zero_nb = torch.pow(disp / (disp + mean + self.eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + self.eps)

        result = torch.where(x_raw < 1e-8, zero_case, nb_case)
        ridge = self.ridge_lambda * torch.mean(torch.square(pi))

        return torch.mean(result) + ridge

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)
    
