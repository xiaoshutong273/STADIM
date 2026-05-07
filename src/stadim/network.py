import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================== Model =============================
class MeanAct(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super(MeanAct, self).__init__()
        self.eps = eps
    def forward(self, x):
        x = torch.clamp(x, max=20)
        return torch.exp(x) + self.eps

class DispAct(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super(DispAct, self).__init__()
        self.eps = eps
    def forward(self, x):
        return F.softplus(x) + self.eps

class PiAct(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super(PiAct, self).__init__()
        self.eps = eps
    def forward(self, x):
        return torch.sigmoid(x) + self.eps

def build_mlp(layers):
    modules = []
    n_layers = len(layers) - 1

    for i in range(n_layers):
        modules.append(nn.Linear(layers[i], layers[i+1]))
        modules.append(nn.ReLU())
        modules.append(nn.LayerNorm(layers[i+1]))
    return nn.Sequential(*modules)

class STADIM(nn.Module):
    def __init__(self,
                 input_dim: int,
                 n_batches: int,
                 encoder_layers=[512, 256, 64],
                 decoder_layers=[1000],
                 distribution="nb",
                 seed=2025):
        """
        Args:
            input_dim: number of genes
            n_batches: number of batches
            encoder_layers: list of hidden dims (last element = latent dim)
            decoder_layers: list of hidden dims
            distribution: "nb" | "zinb"
            seed: random seed for initialization
        """
        super(STADIM, self).__init__()
            
        self.input_dim = input_dim
        self.distribution = distribution
        self.n_batches = n_batches
        if n_batches > 1:
            self.batch_dim = 8
            self.batch_emb = nn.Embedding(n_batches, self.batch_dim)
        else:
            self.batch_dim = 0
            self.batch_emb = None

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        decoder_input_dim = encoder_layers[-1] + self.batch_dim
        adjusted_decoder_layers = [decoder_input_dim] + decoder_layers

        self.encoder = build_mlp([input_dim] + encoder_layers)
        self.decoder = build_mlp(adjusted_decoder_layers)

        self.dec_mean = nn.Sequential(nn.Linear(decoder_layers[-1], input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(decoder_layers[-1], input_dim), DispAct())
        if distribution == "zinb":
            self.dec_pi = nn.Sequential(nn.Linear(decoder_layers[-1], input_dim), PiAct())

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

        for m in self.dec_mean.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                nn.init.constant_(m.bias, 0.1)

        for m in self.dec_disp.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                nn.init.constant_(m.bias, 0.1)

        if self.distribution == "zinb":
            for m in self.dec_pi.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x, batch_index=None):#, mode='train'):
        z = self.encoder(x)
        if batch_index is None:
            return z
        if self.batch_dim == 0:
            z_combined = z
        else:
            # if mode == 'train':
            #     b_emb = self.batch_emb(batch_index)
            # else:
            #     ref_id = torch.zeros_like(batch_index) # + 1
            #     b_emb = self.batch_emb(ref_id)
            b_emb = self.batch_emb(batch_index)
            z_combined = torch.cat([z, b_emb], dim=1)
        output = self.decoder(z_combined)

        if self.distribution == "nb":
            return z, self.dec_mean(output), self.dec_disp(output)
        elif self.distribution == "zinb":
            return z, self.dec_mean(output), self.dec_disp(output), self.dec_pi(output)
 