
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from . import sde_lib
from scipy import integrate

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

class SDEDiffusion(nn.Module):
    def __init__(
        self,
        model,
        sde,
        snr,
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
        super().__init__()
        self.model = model
        self.sde = sde
        self.snr = snr
        self.continuous = continuous
        self.conditional = conditional
        self.reduce_mean = reduce_mean
        self.time_eps = time_eps
        self.probability_flow = probability_flow
        self.likelihood_weighting = likelihood_weighting
        self.model_fn = self.get_model_fn()
        self.score_fn = self.get_score_fn()
        self.loss_fn = self.get_loss_fn()
        self.predictor = predictor
        self.corrector = corrector
        self.n_steps = corrector_steps   

    def get_model_fn(self):
        if self.conditional:
            def model_fn(x, t, condition):
                return self.model(torch.cat((condition, x), dim=1), t)
        else:
            def model_fn(x, t, condition):
                return self.model(x, t)        
        return model_fn
    
    def get_score_fn(self):
        """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

        Args:
            sde: An `sde_lib.SDE` object that represents the forward SDE.
            model: A score model.
            train: `True` for training and `False` for evaluation.
            continuous: If `True`, the score-based model is expected to directly take continuous time steps.

        Returns:
            A score function.
        """

        if isinstance(self.sde, sde_lib.VPSDE) or isinstance(self.sde, sde_lib.subVPSDE):
            def score_fn(x, t, condition):
            # Scale neural network output by standard deviation and flip sign
                if self.continuous or isinstance(self.sde, sde_lib.subVPSDE):
                    # For VP-trained models, t=0 corresponds to the lowest noise level
                    # The maximum value of time embedding is assumed to 999 for
                    # continuously-trained models.
                    labels = t * 999
                    score = self.model_fn(x, labels, condition)
                    std = self.sde.marginal_prob(torch.zeros_like(x), t)[1]
                else:
                    # For VP-trained models, t=0 corresponds to the lowest noise level
                    labels = t * (self.sde.N - 1)
                    score = self.model_fn(x, labels, condition)
                    std = self.sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

                score = -score / std[:, None, None, None]
                return score

        elif isinstance(self.sde, sde_lib.VESDE):
            def score_fn(x, t, condition):
                if self.continuous:
                    labels = self.sde.marginal_prob(torch.zeros_like(x), t)[1]
                else:
                    # For VE-trained models, t=0 corresponds to the highest noise level
                    labels = self.sde.T - t
                    labels *= self.sde.N - 1
                    labels = torch.round(labels).long()

                score = self.model_fn(x, labels, condition)
                return score

        else:
            raise NotImplementedError(f"SDE class {self.sde.__class__.__name__} not yet supported.")

        return score_fn
    
    def sde_loss_fn(self, batch, condition=None):
        """Compute the loss function.

        Args:
        batch: A mini-batch of training data.

        Returns:
        loss: A scalar that represents the average loss value across the mini-batch.
        """
        t = torch.rand(batch.shape[0], device=batch.device) * (self.sde.T - self.time_eps) + self.time_eps
        z = torch.randn_like(batch)
        mean, std = self.sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = self.score_fn(perturbed_data, t, condition)

        if not self.likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = self.reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = self.sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = self.reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss
    
    def smld_loss_fn(self, batch, condition=None):
        """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
        assert isinstance(self.sde, sde_lib.VESDE), "SMLD training only works for VESDEs."

        # Previous SMLD models assume descending sigmas        

        labels = torch.randint(0, self.sde.N, (batch.shape[0],), device=batch.device)
        sigmas = self.smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None, None, None]
        perturbed_data = noise + batch
        score = self.model_fn(perturbed_data, labels, condition)
        target = -noise / (sigmas ** 2)[:, None, None, None]
        losses = torch.square(score - target)
        losses = self.reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
        loss = torch.mean(losses)
        return loss
    
    def ddpm_loss_fn(self, batch, condition=None):
        """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
        assert isinstance(self.sde, sde_lib.VPSDE), "DDPM training only works for VPSDEs."     
        labels = torch.randint(0, self.sde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = self.sde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = self.sde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                        sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        score = self.model_fn(perturbed_data, labels, condition)
        losses = torch.square(score - noise)
        losses = self.reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss
    
    def get_loss_fn(self):
        self.reduce_op = torch.mean if self.reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        if self.continuous:              
            return self.sde_loss_fn
        else:
            assert not self.likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(self.sde, sde_lib.VESDE):
                self.smld_sigma_array = torch.flip(self.sde.discrete_sigmas, dims=(0,))
                return self.smld_loss_fn
            elif isinstance(self.sde, sde_lib.VPSDE):                               
                return self.ddpm_loss_fn
            else:
                raise ValueError(f"Discrete training for {self.sde.__class__.__name__} is not recommended.")

    def forward(self, x, condition=None):
        loss = self.loss_fn(x, condition)
        return loss

    def EulerMaruyamaPredict(self, x, t, condition=None):
        dt = -1. / self.rsde.N        
        drift, diffusion = self.rsde.sde(x, t, condition)
        x_mean = x + drift * dt
        z = torch.randn_like(drift)
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean
    
    def ReverseDiffusionPredict(self, x, t, condition=None):
        f, G = self.rsde.discretize(x, t, condition)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean

    def AncestralPredict(self, x, t, condition=None):
        """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

        if not isinstance(self.sde, sde_lib.VPSDE) and not isinstance(self.sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {self.sde.__class__.__name__} not yet supported.")
        assert not self.probability_flow, "Probability flow not supported by ancestral sampling"

        def vesde_update_fn(self, x, t, condition):
            sde = self.sde
            timestep = (t * (sde.N - 1) / sde.T).long()
            sigma = sde.discrete_sigmas[timestep]
            adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
            score = self.score_fn(x, t, condition)
            x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
            std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
            noise = torch.randn_like(x)
            x = x_mean + std[:, None, None, None] * noise
            return x, x_mean

        def vpsde_update_fn(self, x, t, condition):
            sde = self.sde
            timestep = (t * (sde.N - 1) / sde.T).long()
            beta = sde.discrete_betas.to(t.device)[timestep]
            score = self.score_fn(x, t, condition)
            x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
            noise = torch.randn_like(x)
            x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
            return x, x_mean

        if isinstance(self.sde, sde_lib.VESDE):
            return vesde_update_fn(x, t, condition)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return vpsde_update_fn(x, t, condition)

    def LangevinCorrector(self, x, t, condition=None):
        if not isinstance(self.sde, sde_lib.VPSDE) \
            and not isinstance(self.sde, sde_lib.VESDE) \
            and not isinstance(self.sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t, condition)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean

    
    def AnnealedLangevinDynamics(self, x, t, condition=None):
        """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

        We include this corrector only for completeness. It was not directly used in our paper.
        """
        if not isinstance(sde, sde_lib.VPSDE) \
            and not isinstance(sde, sde_lib.VESDE) \
            and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {self.sde.__class__.__name__} not yet supported.")
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t, condition)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean

    
    def get_predictor(self):
        if self.predictor is None:
            def NonePredict(x, t, condition):
                return x, x
            return NonePredict
        elif self.predictor.lower() == "euler":
            return self.EulerMaruyamaPredict
        elif self.predictor.lower() == "diffusion":
            return self.ReverseDiffusionPredict
        elif self.predictor.lower() == "ancestral":
            return self.AncestralPredict        
        else:
            raise NotImplementedError(f"Predictior {self.predictor} not yet supported.")

    def get_corrector(self):
        if self.corrector is None:
            def NoneCorrect(x, t, condition):
                return x, x
            return NoneCorrect
        elif self.corrector.lower() == "langevin":
            return self.LangevinCorrector
        elif self.corrector.lower() == "annealedlangevin":
            return self.AnnealedLangevinDynamics
        else:
            raise NotImplementedError(f"Corrector {self.corrector} not yet supported.")
    
    def pc_sampling(self, x_in):
        self.rsde = self.sde.reverse(self.score_fn, self.probability_flow)
        predictor = self.get_predictor()
        corrector = self.get_corrector()
        if self.conditional:
            device = x_in.device
            shape = x_in.shape
            condition = x_in
        else:
            device = next(self.model.parameters()).device
            shape = x_in
            condition = None
        
        x = self.sde.prior_sampling(shape).to(device)
        timesteps = torch.linspace(self.sde.T, self.time_eps, self.sde.N, device=device)

        for i in tqdm(range(self.sde.N), desc='sampling loop time step', total=self.sde.N):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device) * t
            x, x_mean = corrector(x, vec_t, condition)
            x, x_mean = predictor(x, vec_t, condition)            

        return x, x_mean

    def ode_sampling(self, x_in, rtol=1e-5, atol=1e-5, method='RK45', denoise=True):
        self.rsde = self.sde.reverse(self.score_fn, probability_flow=True)
        if self.conditional:
            device = x_in.device
            shape = x_in.shape
            condition = x_in
        else:
            device = next(self.model_fn.parameters()).device
            shape = x_in
            condition = None
        
        x = self.sde.prior_sampling(shape).to(device)

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t
            drift = self.rsde.sde(x, vec_t, condition)[0]
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.sde.T, self.time_eps), to_flattened_numpy(x),
                                        rtol=rtol, atol=atol, method=method)
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

        # Denoising is equivalent to running one predictor step without adding noise
        if denoise == True:
            vec_eps = torch.ones(x.shape[0], device=x.device) * self.time_eps
            x = self.ReverseDiffusionPredict(x, vec_eps, condition)
        
        return x, x
