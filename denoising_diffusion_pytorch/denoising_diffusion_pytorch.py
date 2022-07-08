import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count

from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from PIL import Image

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

##################################################################################
# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def normalize_to_neg_one_to_one(img):
    """
    normalize the image from [0, 1] to [-1, 1]
    """
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    """
    this function converts the image in range [-1, 1] to range [0, 1]
    """
    return (t + 1) * 0.5

##############################################################################
# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )


def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

###########################################################################################
# gaussian diffusion trainer class


def extract(a, t, x_shape):
    """
    For a given minibatch, this function is used to extract the value at corresponding time step index
    For 1D input a, it is equivalent to a[t] where t is an array with size of minibatch
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels=3,
        timesteps=1000,
        loss_type='l1',
        objective='pred_noise',
        beta_schedule='cosine',
        p2_loss_weight_gamma=0.,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and denoise_fn.channels != denoise_fn.out_dim)

        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.objective = objective

        # get beta for each time step, where beta is the volatility of noise at each time step
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # calculate alpha = 1 - beta, and alpha_cumprod as cumulative product of alpha
        # alphas_cumprod is equivalent to \bar{\alpha} in the DDPM paper
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)  # this just insert 1 to the beginning and discard the last value
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32
        # here the self.register_buffer(name, tensor, persistent=True) is provided from nn.Module base class
        # suppose we want to put a variable as self.some_var. If self.some_var = some_tensor is directly used,
        # then self.some_var will be considered as a Parameter and will be optimized. On the other hand, if
        # we use self.register_buffer("some_var", some_tensor) then we can also access self.some_var but here
        # self.some_var will not be Module Parameters, instead they will be just like a normal python instance
        # variables
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # pre-compute parameters for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))  # \sqrt{alphas_cumprod}}
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))  # sqrt{1 - alphas_cumprod}}
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))  # log(1 - \alphas_cumprod)
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))  # sqrt{1 / alphas_cumprod}
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))  # sqrt{1 / alphas_cumprod - 1}

        # pre-compute parameters for posterior distribution q(x_{t-1} | x_t, x_0)
        # the variance of q(x_{t-1} | x_t, x_0) is computed based on the following formula:
        # posterior_variance = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
        # log variance calculation is clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate weight for loss terms
        # theoretically those with larger noise should be emphasized more since those are
        # more difficult tasks
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        """
        this function recovers x_0 from x_t and known noise z_t ~ N(0, 1):
        since x_t = sqrt(alphas_cumprod) * x_0 + sqrt(1 - alphas_cumprod) * z_t, we can get
        x_0 = 1/sqrt(alpha_cumprod) * x_t - sqrt(1/alpha_cumprod - 1) * z_t
        Note everything here is from Bayes rule and has nothing to do with the model yet.
        This function is not used for training. It is used to generate samples backwards from noise.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        This function computes the mean and variance of q(x_{t-1} | x_t, x_0)
        This corresponds to the following formula:
        posterior_mean = sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod) * x0 +
                         (1 - alpha_cumprod_prev) * sqrt(alpha) / (1 - alpha_cumprod) * x_t
                       = posterior_mean_coef1 * x0 + posterior_mean_coef2 * x_t
        Note everything here is from Bayes rule and has nothing to do with the model yet.
        This function is not used for training. It is used to generate samples backwards from noise.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        """
        this function predicts the mean and variance of distribution p(x_{t-1} | x_t).
        Note this result is model dependent since it is the result of variational inference.
        This function first computes the predicted noise given x_t, then calculate x_0 from
        x_t and noise (this step is model independent since we are basically subtracting
        the noise), then calculate mean and variance of x_{t-1} from x_t and x_0 (this step
        is model independent since p(x_{t-1} | x_t, x_0) comes from Bayes rule that inverts
        the forward smearing process

        This function is not used for training. It is used to generate samples backwards from noise.
        """

        # predict noise
        model_output = self.denoise_fn(x, t)

        # generate prediction of x_0
        if self.objective == 'pred_noise':
            # the model outputs a noise that is used to construct x_t from x_0,
            # hence the noise needs to be subtracted to reconstruct x_0 from x_t
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == 'pred_x0':
            # the model outputs x_0, hence nothing to do
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # clip denoised image
        if clip_denoised:
            x_start.clamp_(-1., 1.)

        # computes the mean and variance of q(x_{t-1} | x_t, x_0).
        # now this is possible because x_t is given and x_0 is already computed from above
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True):
        """
        sampling backwards for just one time step,
        each time generates a denoised image x_{t-1} from smeared image x_t
        """

        # predict x_{t-1} image mean and variance for distribution p(x_{t-1} | x_t)
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)

        # Additional noise added when denoising, no noise when t == 0
        # the generated image is mean + sqrt(variance) * noise
        # here the noise is needed because we get only the mean and variance of the image, hence
        # we need to add a noise like mean + sqrt(variance) * noise to account for the variance
        # here we do not add noise when t = 0 because the mean and variance is for distribution
        # p(x_{t-1} | x_t, x_0) and should have zero variance for p(x_0 | x_1, x_0)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        generate a minibatch of images, where the batch size is the first dimension of shape argument.
        """
        device = self.betas.device

        b = shape[0]  # batch size to generate
        img = torch.randn(shape, device=device)  # a minibatch of random noise as value of x_T to start image generation

        # generate image from x_T to x_0
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        # convert image from range [-1, 1] to range [0, 1]
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        """
        generate a minibatch of images
        """
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        """
        the key function for adding noise to a given image, this function creates x_t from x_0 by adding noise
        it performs the following operation:
        x_0 = x_0 * sqrt{alpha_cumprod} + noise * sqrt{1 - alpha_cumprod}
        this function is used to generate x_t which will be then used to predict added noise that
        constructed x_t from x_0 in loss function
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        """
        create loss function, the paper uses loss function of L2 for true noise and predicted noise
        """
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise=None):
        """
        generate loss function with given x0 and time steps t
        the loss function is very simple:
            1. with x0 and t, generate x_t
            2. let the neural network predict the noise used to generate x_t from x_0
            3. get the MSE error between true noise and predicted noise as the loss function
        """
        b, c, h, w = x_start.shape

        # generate noise
        noise = default(noise, lambda: torch.randn_like(x_start))

        # get a smeared image, then use the smeared image to predict the added noise
        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # this is x_t
        model_out = self.denoise_fn(x, t)  # predict the noise used to generate x_t from x_0

        # this corresponds to different ways of parameterizing,
        # we can let the model predict the noise or the image
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # calculate the loss function between each target and output
        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        # the weight used in the loss function as a function of noise level is added
        # loss is simply unweighted_loss * p2_loss_weight
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        return loss.mean()

    def forward(self, img, *args, **kwargs):
        """
        create loss from given batches of images
        """

        # check image shape
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # random sample a minibatch of time steps t
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # normalize the image from [0, 1] to [-1, 1]
        img = normalize_to_neg_one_to_one(img)

        # generate loss function
        return self.p_losses(img, t, *args, **kwargs)

##################################################################################


class Dataset(Dataset):
    """
    Dataset classes, used to indicate how to load data with a given index
    """
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # create transform for the dataset
        self.transform = T.Compose([
            T.Lambda(partial(convert_image_to, 'RGB')),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        # return the number of data, used to calculating whether we have reached the end of a epoch
        return len(self.paths)

    def __getitem__(self, index):
        # get an item at index, needs to be transformed.
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

##################################################################################

class Trainer(object):
    """
    trainer class, used to train a given diffusion_model passed as argument
    """
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./results',
        amp=False,
        fp16=False,
        split_batches=True
    ):
        """
        gradient_accumulate_every:
            accumulate gradient for n steps from different minibatches, then do gradient descent

        """

        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # create dataset, which defines where to find data with given index
        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip=augment_horizontal_flip)

        # create dataloader, which is the actual workhorse to load the data into memory and generate batches
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())
        self.dl = cycle(dl)  # now dataloader has infinite amount of data by looping and yield

        # create optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        # for logging results in a folder periodically
        # this is only done in main process
        # here an EMA model that takes exponential moving average of parameters is also saved
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        # now model, dataloader and optimizer should all run on GPU (suppose exists)
        # there should be no need to call data.to(device) since the dataloader is generated
        # by accelerator
        self.model, self.dl, self.opt = self.accelerator.prepare(self.model, self.dl, self.opt)

    def save(self, milestone):
        """
        save relevant data and models
        """

        # only save from main process
        if not self.accelerator.is_main_process:
            return

        data = {
            'step': self.step,
            # use accelerator.get_state_dict to save model that was wrappeed in accelerator
            'model': self.accelerator.get_state_dict(self.model),
            'ema': self.ema.state_dict(),  # this is fine because it is not wrapped by accelerator
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        # need to unwrap_model before load since it was wrapped by accelerator
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']

        # load EMA model
        self.ema.load_state_dict(data['ema'])

        # load scaler
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        # the progress bar is only enabled for main_process
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                # accumulate gradient from many minibatch before doing gradient descent
                # equivalent to larger batch size
                for _ in range(self.gradient_accumulate_every):

                    # the call to(device) should not be necessary, since data loader is generated by accelerator
                    # this might be safe since we used the device as to(accelerator.device)
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        # the gradient needs to be scaled because each step is just a fraction of all the batches
                        # needed before gradient descent
                        # note here we call self.accelerator.backward instead of self.opt.backward
                        self.accelerator.backward(loss / self.gradient_accumulate_every)

                # show progress bar on main process
                pbar.set_description(f'loss: {loss.item():.4f}')

                # wait for gradient descent is computed on all processes
                accelerator.wait_for_everyone()

                # update the parameters
                self.opt.step()
                self.opt.zero_grad()

                # need to wait for all process to reach this point
                accelerator.wait_for_everyone()

                # save model in the main process
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()  # update the EMA model params since diffusion model parameters changed

                    # time to save model
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:

                        # below use self.ema.ema_model to perform evaluation
                        # go to evaluation mode
                        self.ema.ema_model.eval()

                        # sample images
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        # save all generated images
                        all_images = torch.cat(all_images_list, dim=0)
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        self.save(milestone)

                self.step += 1
                pbar.update(1)

        accelerator.print('training complete')
