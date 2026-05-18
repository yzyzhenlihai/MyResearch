from functools import partial
import torch
import torch.nn.functional as F
from .utils import *
from torch import nn
class DiffusionModel(nn.Module):
    def __init__(
            self,
            sigma_data: float,
            sigma_min: float,
            sigma_max: float,
            device: str,
            sigma_sample_density_type: str = 'loglogistic',
            clip_denoised=False,
            max_action=1.0,
    ) -> None:
        super().__init__()

        self.device = device
        # use the score wrapper
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        self.epochs = 0
        self.clip_denoised = clip_denoised
        self.max_action = max_action

    def get_diffusion_scalings(self, sigma):
        """
        Computes the scaling factors for diffusion training at a given time step sigma.

        Args:
        - self: the object instance of the model
        - sigma (float or torch.Tensor): the time step at which to compute the scaling factors

        , where self.sigma_data: the data noise level of the diffusion process, set during initialization of the model

        Returns:
        - c_skip (torch.Tensor): the scaling factor for skipping the diffusion model for the given time step sigma
        - c_out (torch.Tensor): the scaling factor for the output of the diffusion model for the given time step sigma
        - c_in (torch.Tensor): the scaling factor for the input of the diffusion model for the given time step sigma

        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def diffusion_train_step(self, model, x, cond, noise=None, t_chosen=None, return_denoised=False):
        """
        Computes the training loss and performs a single update step for the score-based model.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, dim)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)

        Returns:
        - loss.item() (float): the scalar value of the training loss for this batch

        """
        model.train()
        x = x.to(self.device)
        if cond is not None:
            cond = cond.to(self.device)
        if t_chosen is None:
            t_chosen = self.make_sample_density()(shape=(len(x),), device=self.device)

        if return_denoised:
            denoised_x, diffusion_loss = self.diffusion_loss(model, x, cond, t_chosen, noise, return_denoised)
            return denoised_x, diffusion_loss, t_chosen
        else:
            diffusion_loss = self.diffusion_loss(model, x, cond, t_chosen, noise, return_denoised)
            return diffusion_loss

    def diffusion_loss(self, model, x, cond, t, noise, return_denoised):
        """
        Computes the diffusion training loss for the given model, input, condition, and time.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, channels, height, width)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)
        - t (torch.Tensor): the time step tensor of shape (batch_size,)

        Returns:
        - loss (torch.Tensor): the diffusion training loss tensor of shape ()

        The diffusion training loss is computed based on the following equation from Karras et al. 2022:
        loss = (model_output - target)^2.mean()
        where,
        - noise: a tensor of the same shape as x, containing randomly sampled noise
        - x_1: a tensor of the same shape as x, obtained by adding the noise tensor to x
        - c_skip, c_out, c_in: scaling tensors obtained from the diffusion scalings for the given time step
        - t: a tensor of the same shape as t, obtained by taking the natural logarithm of t and dividing it by 4
        - model_output: the output tensor of the model for the input x_1, condition cond, and time t
        - target: the target tensor for the given input x, scaling tensors c_skip, c_out, c_in, and time t
        """
        if noise is None:
            noise = torch.randn_like(x)
        x_1 = x + noise * append_dims(t, x.ndim)
        c_skip, c_out, c_in = [append_dims(x, 2) for x in self.get_diffusion_scalings(t)]
        t = torch.log(t) / 4
        model_output = model(x_1 * c_in, cond, t)

        if self.clip_denoised:
            denoised_x = c_out * model_output + c_skip * x_1
            denoised_x = denoised_x.clamp(-self.max_action,self.max_action)
            loss = ((denoised_x - x)/c_out).pow(2).mean()
        else:
            denoised_x = c_out * model_output + c_skip * x_1
            target = (x - c_skip * x_1) / c_out
            loss = (model_output - target).pow(2).mean()

        if return_denoised:
            return denoised_x, loss
        else:
            return loss
        

    def make_sample_density(self):
        """
        Returns a function that generates random timesteps based on the chosen sample density.

        Args:
        - self: the object instance of the model

        Returns:
        - sample_density_fn (callable): a function that generates random timesteps

        The method returns a callable function that generates random timesteps based on the chosen sample density.
        The available sample densities are:
        - 'lognormal': generates random timesteps from a log-normal distribution with mean and standard deviation set
                    during initialization of the model also used in Karras et al. (2022)
        - 'loglogistic': generates random timesteps from a log-logistic distribution with location parameter set to the
                        natural logarithm of the sigma_data parameter and scale and range parameters set during initialization
                        of the model
        - 'loguniform': generates random timesteps from a log-uniform distribution with range parameters set during
                        initialization of the model
        - 'uniform': generates random timesteps from a uniform distribution with range parameters set during initialization
                    of the model
        - 'v-diffusion': generates random timesteps using the Variational Diffusion sampler with range parameters set during
                        initialization of the model
        - 'discrete': generates random timesteps from the noise schedule using the exponential density
        - 'split-lognormal': generates random timesteps from a split log-normal distribution with mean and standard deviation
                            set during initialization of the model
        """
        sd_config = []

        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_uniform, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'uniform':
            return partial(rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.n_sampling_steps, 'exponential')
            return partial(rand_discrete, values=sigmas)
        else:
            raise ValueError('Unknown sample density type')
        

    def sample(self, model, cond, action_samples, n_steps=20):
        """
        Sample actions from the pre-trained diffusion-based behavior policy.

        Args:
            model (ScoreNetwork): The score model.
            cond (torch.Tensor): Conditioning input (e.g., state), shape (B, cond_dim)
            action_samples (int): Number of samples per condition/state
            n_steps (int): Number of diffusion steps

        Returns:
            actions (torch.Tensor): Sampled actions, shape (B, action_samples, action_dim)
        """
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")

        model.eval() 

        B, cond_dim = cond.shape
        # Initialize the action samples with random noise
        action_dim = model.layers.output_dim
        actions = torch.randn(B * action_samples, action_dim).to(self.device)

        # Initialize the noise schedule using the same method as in training
        sigmas = self.make_sample_density()(shape=(n_steps,), device=self.device).sort(descending=True)[0]

        # Repeat the conditioning input for each sample
        cond = cond.unsqueeze(1).repeat(1, action_samples, 1).view(B * action_samples, cond_dim)

        with torch.no_grad():
            for i in range(n_steps):
                sigma = sigmas[i]
                sigma = sigma.expand(B * action_samples).unsqueeze(1)  # Expand sigma to match the batch size

                c_skip, c_out, c_in = [append_dims(x, 2) for x in self.get_diffusion_scalings(sigma)]
                t = torch.log(sigma) / 4

                # Add small noise to prevent convergence to a single point
                if i < n_steps - 1:
                    noise_scale = (sigmas[i] ** 2 - sigmas[i + 1] ** 2).sqrt()
                    noise = torch.randn_like(actions) * noise_scale.view(-1, 1)
                else:
                    noise = 0

                model_output = model(actions * c_in, cond, t)
                actions = c_out * model_output + c_skip * actions + noise

                if self.clip_denoised:
                    actions = actions.clamp(-self.max_action, self.max_action)

        actions = actions.view(B, action_samples, action_dim)
        return actions
        
