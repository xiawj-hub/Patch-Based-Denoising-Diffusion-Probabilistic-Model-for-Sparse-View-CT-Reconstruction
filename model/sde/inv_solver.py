import torch
from tqdm import tqdm
from .diffusion import SDEDiffusion

class InvSolver(SDEDiffusion):
    def __init__(
        self,
        model,
        sde,
        snr,
        Afun,
        Ainv,
        mask,
        coeff,
        continuous=True,
        reduce_mean=True,
        conditional=False,
        probability_flow = False,
        time_eps = 1e-5,
        likelihood_weighting = False,
        predictor = "euler",
        corrector = None,
        corrector_steps = 1
    ):
        super().__init__(model, sde, snr, continuous, reduce_mean, 
            conditional, probability_flow, time_eps, likelihood_weighting, 
            predictor, corrector, corrector_steps)
        self.Afun = Afun
        self.Ainv = Ainv
        self.mask = mask
        self.coeff = coeff
        
    def merged_known(self, x, y, t, coeff):
        known_sino = self.get_known_sino(x, y, t)
        x_sino = self.Afun(x)
        merged_sino = x_sino * (1 - coeff) + known_sino * coeff
        merged_sino = merged_sino * self.mask + x_sino * (1-self.mask)
        known_x = self.Ainv(merged_sino, x)
        return known_x

    def get_known_sino(self, x, y, t):
        mean, std = self.sde.marginal_prob(y, t[:y.shape[0]])
        z = torch.randn_like(x)
        Az = self.Afun(z)
        known_sino = mean + std[:, None, None, None] * Az
        return known_sino

    def pc_sampling(self, y, denoise=True):
        self.rsde = self.sde.reverse(self.score_fn, self.probability_flow)
        predictor = self.get_predictor()
        corrector = self.get_corrector()
        shape = self.Ainv(y, None).shape
        device = y.device
        x = self.sde.prior_sampling(shape).to(device)
        timesteps = torch.linspace(self.sde.T, self.time_eps, self.sde.N, device=device)

        for i in tqdm(range(self.sde.N), desc='sampling loop time step', total=self.sde.N):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device) * t
            x = self.merged_known(x, y, vec_t, self.coeff)
            x, x_mean = corrector(x, vec_t)
            x = self.merged_known(x, y, vec_t, self.coeff)
            x, x_mean = predictor(x, vec_t)            

        if denoise == True:
            vec_eps = torch.ones(x.shape[0], device=x.device) * self.time_eps
            alpha, std = self.sde.marginal_prob(torch.ones_like(x), vec_eps)
            score = self.score_fn(x, vec_eps, None)
            x = (x + std[:, None, None, None] ** 2 * score) / alpha
            x = self.merged_known(x, y, vec_eps, 1.0)

        return x, x_mean
