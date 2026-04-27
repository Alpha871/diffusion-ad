import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T


def get_beta_schedule(num_diffusion_steps, name="cosine"):
    betas = []
    if name == "cosine":
        max_beta = 0.999
        def f(t): return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)
    elif name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end,
                            num_diffusion_steps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas


def extract(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape).to(device)


def mean_flat(tensor):
    return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL Divergence between two gaussians

    :param mean1:
    :param logvar1:
    :param mean2:
    :param logvar2:
    :return: KL Divergence between N(mean1,logvar1^2) & N(mean2,logvar2^2))
    """
    return 0.5 * (-1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretised_gaussian_log_likelihood(x, means, log_scales):
    """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.
        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))

    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,
                    torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusionModel:
    def __init__(
            self,
            img_size,
            betas,
            img_channels=1,
            loss_type="l2",  # l2,l1 hybrid
            loss_weight='none',  # prop t / uniform / None
            noise="gauss",  # gauss / perlin / simplex
    ):
        super().__init__()

        if noise == "gauss":
            self.noise_fn = lambda x, t: torch.randn_like(x)

        self.img_size = img_size
        self.img_channels = img_channels
        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps, 0, -1)
        elif loss_weight == "uniform":
            self.weights = np.ones(self.num_timesteps)

        self.loss_weight = loss_weight
        alphas = 1 - betas
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:],0.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(
            1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.gauss_blur = T.GaussianBlur(kernel_size=31, sigma=3)

    def sample_t_with_weights(self, b_size, device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p), size=b_size, p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / len(p) * p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

    def predict_x_0_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps)

    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - pred_x_0) \
            / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)

    def q_mean_variance(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape, x_0.device)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_0.shape, x_0.device
        )
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """

        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t)

        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_var = extract(self.posterior_variance,
                                t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(self, model, x_t, t, estimate_noise=None):
        """
        Finds the mean & variance from N(x_{t-1}; mu_theta(x_t,t), sigma_theta (x_t,t))

        :param model:
        :param x_t:
        :param t:
        :return:
        """
        if estimate_noise == None:
            estimate_noise = model(x_t, t)

        # fixed model variance defined as \hat{\beta}_t - could add learned parameter
        model_var = np.append(self.posterior_variance[1], self.betas[1:])
        model_logvar = np.log(model_var)
        model_var = extract(model_var, t, x_t.shape, x_t.device)
        model_logvar = extract(model_logvar, t, x_t.shape, x_t.device)

        pred_x_0 = self.predict_x_0_from_eps(
            x_t, t, estimate_noise).clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(
            pred_x_0, x_t, t
        )
        return {
            "mean":         model_mean,
            "variance":     model_var,
            "log_variance": model_logvar,
            "pred_x_0":     pred_x_0,
        }

    def sample_p(self, model, x_t, t, denoise_fn="gauss"):
        out = self.p_mean_variance(model, x_t, t)
        # noise = torch.randn_like(x_t)
        if denoise_fn == "gauss":
            noise = torch.randn_like(x_t)
        else:
            noise = denoise_fn(x_t, t)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )

        sample = out["mean"] + nonzero_mask * \
            torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}

    def forward_backward(
            self, model, x, see_whole_sequence="half", t_distance=None, denoise_fn="gauss",
    ):
        assert see_whole_sequence == "whole" or see_whole_sequence == "half" or see_whole_sequence == None

        if t_distance == 0:
            return x.detach()

        if t_distance is None:
            t_distance = self.num_timesteps
        seq = [x.cpu().detach()]
        if see_whole_sequence == "whole":

            for t in range(int(t_distance)):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                # noise = torch.randn_like(x)
                noise = self.noise_fn(x, t_batch).float()
                with torch.no_grad():
                    x = self.sample_q_gradual(x, t_batch, noise)

                seq.append(x.cpu().detach())
        else:
            t_tensor = torch.tensor(
                [t_distance - 1], device=x.device).repeat(x.shape[0])
            x = self.sample_q(
                x, t_tensor,
                self.noise_fn(x, t_tensor).float()
            )
            if see_whole_sequence == "half":
                seq.append(x.cpu().detach())

        for t in range(int(t_distance) - 1, -1, -1):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                out = self.sample_p(model, x, t_batch, denoise_fn)
                x = out["sample"]
            if see_whole_sequence:
                seq.append(x.cpu().detach())

        return x.detach() if not see_whole_sequence else seq

    def sample_q(self, x_0, t, noise):
        """
            q (x_t | x_0 )

            :param x_0:
            :param t:
            :param noise:
            :return:
        """
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise)

    def sample_q_gradual(self, x_t, t, noise):
        """
        q (x_t | x_{t-1})
        :param x_t:
        :param t:
        :param noise:
        :return:
        """
        return (extract(self.sqrt_alphas, t, x_t.shape, x_t.device) * x_t +
                extract(self.sqrt_betas, t, x_t.shape, x_t.device) * noise)

    def calc_vlb_xt(self, model, x_0, x_t, t, estimate_noise=None):
        # find KL divergence at t
        true_mean, _, true_log_var = self.q_posterior_mean_variance(
            x_0, x_t, t)
        output = self.p_mean_variance(model, x_t, t, estimate_noise)
        kl = normal_kl(true_mean, true_log_var,
                       output["mean"], output["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretised_gaussian_log_likelihood(
            x_0, output["mean"], log_scales=0.5 * output["log_variance"]
        )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        nll = torch.where((t == 0), decoder_nll, kl)
        return {"output": nll, "pred_x_0": output["pred_x_0"]}


    
    def calc_loss(self, model, x_0, t):

        noise = self.noise_fn(x_0, t).float()
        x_t = self.sample_q(x_0, t, noise)
        estimate_noise = model(x_t, t)
        loss = {}
        if self.loss_type == "l1":
            loss["loss"] = mean_flat((estimate_noise - noise).abs())
        elif self.loss_type == "l2":
            loss["loss"] = mean_flat((estimate_noise - noise).square())
        elif self.loss_type == "hybrid":
            # add vlb term
            loss["vlb"] = self.calc_vlb_xt(
                model, x_0, x_t, t, estimate_noise)["output"]
            loss["loss"] = loss["vlb"] + \
                mean_flat((estimate_noise - noise).square())
        else:
            loss["loss"] = mean_flat((estimate_noise - noise).square())
        return loss, x_t, estimate_noise

    def debug_anomaly_mask_source(
        self,
        anomaly_mask,
        anomaly_label=None,
        prefix: str = "[mask-debug]",
        show_unique: bool = True,
        max_unique: int = 10,
    ):
        """
        Affiche des infos pour deviner d'où vient anomaly_mask:
        - GT (ground_truth) -> souvent 0/255 et masque vide quand label=0
        - masque binaire -> 0/1
        - masque float -> 0..1
        """
    
        # ---- 0) Validation / anti-crash ----
        if not (torch.is_tensor(anomaly_mask) or isinstance(anomaly_mask, (list, tuple))):
            # (np.ndarray est géré si numpy est importé; sinon on le traite via torch.as_tensor plus bas)
            print(f"{prefix} ❌ anomaly_mask type inattendu: {type(anomaly_mask)}")
            return
    
        # ---- 1) Convert to tensor safely ----
        try:
            m = anomaly_mask if torch.is_tensor(anomaly_mask) else torch.as_tensor(anomaly_mask)
        except Exception as e:
            print(f"{prefix} ❌ impossible de convertir anomaly_mask en tensor. type={type(anomaly_mask)} err={repr(e)}")
            return
    
        # ---- 2) CPU inspect (évite l'impact GPU) ----
        m_cpu = m.detach().cpu()
    
        # ---- 3) Mettre sous forme [B,H,W] si possible ----
        if m_cpu.dim() == 2:          # [H,W]
            m_view = m_cpu.unsqueeze(0)  # -> [1,H,W]
        elif m_cpu.dim() == 3:        # [B,H,W]
            m_view = m_cpu
        elif m_cpu.dim() == 4:        # [B,C,H,W]
            if m_cpu.shape[1] == 1:
                m_view = m_cpu[:, 0]     # -> [B,H,W]
            else:
                # si plusieurs canaux, on réduit (max) pour avoir un masque global
                m_view = m_cpu.max(dim=1).values  # -> [B,H,W]
        else:
            print(f"{prefix} ❌ dim inattendue: {m_cpu.dim()} shape={tuple(m_cpu.shape)}")
            return
    
        # ---- 4) Stats ----
        shp = tuple(m_view.shape)
        mn = float(m_view.min().item())
        mx = float(m_view.max().item())
    
        # ratio de pixels > 0
        ratio_per_img = (m_view > 0).float().flatten(1).mean(dim=1)  # [B]
        ratio_mean = float(ratio_per_img.mean().item())
    
        # ---- 5) Valeurs uniques (optionnel) ----
        uniq_sample = None
        if show_unique:
            try:
                uniq = torch.unique(m_view)
                uniq_sample = (uniq[:max_unique].tolist() if uniq.numel() > max_unique else uniq.tolist())
            except Exception as e:
                uniq_sample = f"unique_failed: {repr(e)}"
    
        # ---- 6) Heuristique ----
        if mx == 0.0:
            guess = "Masque vide (aucune anomalie). Souvent images 'good' ou label=0."
        elif mx == 1.0 and mn == 0.0:
            guess = "Masque binaire 0/1 (souvent déjà normalisé ou généré en tensor)."
        elif mx == 255.0 and mn == 0.0:
            guess = "Masque image 0/255 (souvent ground_truth PNG de MVTec/VisA)."
        elif 0.0 <= mn and mx <= 1.0:
            guess = "Masque float dans [0,1] (peut venir d'un preprocess/normalisation)."
        else:
            guess = "Valeurs atypiques (peut venir d'un traitement intermédiaire)."
    
        # ---- 7) Print ----
        orig_device = m.device if torch.is_tensor(anomaly_mask) else "n/a"
        print(f"{prefix} shape={shp} dtype={m_cpu.dtype} device(original)={orig_device}")
    
        if anomaly_label is not None:
            try:
                lab = anomaly_label.detach().cpu().tolist() if torch.is_tensor(anomaly_label) else anomaly_label
            except Exception:
                lab = str(anomaly_label)
            print(f"{prefix} label={lab}")
    
        print(f"{prefix} min={mn:.4f} max={mx:.4f} anomaly_pixels_ratio≈{ratio_mean*100:.2f}%")
    
        if show_unique:
            print(f"{prefix} unique(sample)={uniq_sample}")
    
        print(f"{prefix} guess: {guess}")


    # def mask_based_ratio(
    #     self,
    #     anomaly_mask: torch.Tensor,
    #     args: dict,
    #     delta_ratio: float = 0.05,
    #     verbose: bool = True,
    #     mask_threshold: float = 0.0,     # 0.0 => tout pixel >0 est anomalie ; 0.5 => binaire "dur"
    #     min_high_normal: int = 20        # évite high_normal=1 (normal_t=0 tout le temps)
    # ):
    #     """
    #     Calcule des bornes de timesteps adaptatives à partir du masque d'anomalie.
    
    #     anomaly_mask: Tensor [B,1,H,W] ou [B,H,W] ou [H,W]
    #                  (peut être 0/255, 0/1, ou floats [0..1] avec valeurs intermédiaires)
    #     args: doit contenir "T" et "less_t_range"
    #     delta_ratio: ex 0.05 => delta = 5% de T
    #     mask_threshold:
    #         - 0.0 : considère anomalie si valeur > 0 (inclut valeurs intermédiaires 0.5 etc.)
    #         - 0.5 : considère anomalie si valeur >= 0.5 (seuil "dur")
    #     min_high_normal: borne minimale pour high_normal_raw (>=1), pour éviter normal_t presque toujours 0.
    
    #     Retour (IMPORTANT): EXACTEMENT 3 valeurs:
    #       high_normal: Tensor [B] (borne haute exclusive pour normal_t, >=1)
    #       low_noisy:   Tensor [B] (borne basse inclusive pour noisier_t, >=1)
    #       ratio:       Tensor [B] (ratio anomalie par image, 0..1)
    #     """
    #     T = int(args["T"])
    #     less = int(args["less_t_range"])
    
    #     m = anomaly_mask if torch.is_tensor(anomaly_mask) else torch.as_tensor(anomaly_mask)
    #     device = m.device
    
    #     # --- Mise en forme -> [B,H,W]
    #     if m.dim() == 2:
    #         m = m.unsqueeze(0)      # [1,H,W]
    #     elif m.dim() == 4 and m.shape[1] == 1:
    #         m = m.squeeze(1)        # [B,H,W]
    
    #     # --- Normalisation si masque en 0..255 (uint8 ou float 0..255)
    #     # (tes logs montrent déjà float 0..1, mais on sécurise)
    #     if m.max() > 1.5:
    #         m = m / 255.0
    
    #     # --- Binarisation contrôlée (prend en compte tes valeurs intermédiaires)
    #     m_bin = (m > mask_threshold) if mask_threshold == 0.0 else (m >= mask_threshold)
    
    #     # --- Ratio anomalie par image: [B]
    #     ratio = m_bin.flatten(1).float().mean(dim=1).clamp(0.0, 1.0)
    
    #     # --- Mapping ratio -> t_center (par image)
    #     t_center = (ratio.sqrt() * (T - 1)).round().long()  # [B]
    
    #     # --- Delta fixe (5% de T) borné
    #     delta = max(1, int(delta_ratio * T))
    #     delta = min(delta, T // 3)
    
    #     # --- Bornes brutes (par image)
    #     high_normal_raw = (t_center - delta).clamp(1, T - 1)
    #     low_noisy_raw   = (t_center + delta).clamp(1, T - 1)
    
    #     # ✅ éviter high_normal=1 trop souvent (donc normal_t=0)
    #     if min_high_normal is not None and min_high_normal > 1:
    #         min_high_normal = min(min_high_normal, less)  # ne dépasse pas la zone "normal"
    #         high_normal_raw = torch.maximum(high_normal_raw, torch.full_like(high_normal_raw, min_high_normal))
    
    #     # --- Garde-fou DiffusionAD (respecter less_t_range)
    #     high_normal = torch.minimum(high_normal_raw, torch.full_like(high_normal_raw, less))
    #     low_noisy   = torch.maximum(low_noisy_raw,   torch.full_like(low_noisy_raw,   less))
    
    #     # --- Infos debug
    #     if verbose:
    #         ratio_pct = (ratio * 100).detach().cpu()
    #         tc = t_center.detach().cpu()
    #         hn = high_normal.detach().cpu()
    #         ln = low_noisy.detach().cpu()
    
    #         # infos sur le masque (utile car tu as des valeurs 0.5019 etc.)
    #         m_min = float(m.detach().min().cpu().item())
    #         m_max = float(m.detach().max().cpu().item())
    
    #         print(f"[mask_based_ratio] T={T}, less_t_range={less}, delta={delta} (~{delta_ratio*100:.1f}%), "
    #               f"mask_thr={mask_threshold}, min_high={min_high_normal}, mask_min={m_min:.3f}, mask_max={m_max:.3f}")
    
    #         for i in range(ratio.shape[0]):
    #             normal_span = int(hn[i].item())      # normal_t in [0, high_normal)
    #             noisy_span  = int(T - ln[i].item())  # noisier_t in [low_noisy, T)
    #             print(
    #                 f"  img[{i}] anomaly={ratio_pct[i]:.2f}% | "
    #                 f"t_center={int(tc[i])} | "
    #                 f"high_normal={int(hn[i])} (span={normal_span}) | "
    #                 f"low_noisy={int(ln[i])} (span={noisy_span})"
    #             )
    
    #     return high_normal.to(device), low_noisy.to(device), ratio.to(device)

    def mask_based_ratio(self, anomaly_mask, args,
                     mask_threshold=0.0,
                     gap_ratio=0.05,
                     min_high=20,
                        
                        ):

        T = int(args["T"])
        less = int(args["less_t_range"])
    
        m = anomaly_mask if torch.is_tensor(anomaly_mask) else torch.as_tensor(anomaly_mask)
    
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.dim() == 4:
            m = m.squeeze(1)
    
        ratio = (m > mask_threshold).float().flatten(1).mean(dim=1).clamp(0, 1)
    
        gap = max(1, int(gap_ratio * T))
    
        # ---- NORMAL (toujours < less - gap)
        high_normal = torch.full_like(
            ratio.long(),
            max(min_high, less - gap)
        ).clamp(1, T - 1)
    
        # ---- NOISY (toujours > less + gap)
        span = (ratio.sqrt() * (T - (less + gap) - 1)).round().long()
        low_noisy = (less + gap + span).clamp(less + gap, T - 1)

        return high_normal.to(m.device), low_noisy.to(m.device), ratio.to(m.device)

    # def mask_based_ratio(
    #     self,
    #     anomaly_mask: torch.Tensor,
    #     args: dict,
    #     delta_ratio: float = 0.05,
    #     min_high: int = 20,
    #     mask_threshold: float = 0.0,
    #     verbose: bool = False
    # ):
    #     T = int(args["T"])
    #     less = int(args["less_t_range"])
    
    #     m = anomaly_mask if torch.is_tensor(anomaly_mask) else torch.as_tensor(anomaly_mask)
    #     device = m.device
    
    #     if m.dim() == 2:
    #         m = m.unsqueeze(0)
    #     elif m.dim() == 4:
    #         m = m.squeeze(1)
    
    #     # --- Binarisation robuste
    #     m_bin = (m > mask_threshold)
    
    #     ratio = m_bin.flatten(1).float().mean(dim=1).clamp(0.0, 1.0)
    
    #     # --- Mapping ratio → centre bruit
    #     t_center = (ratio.sqrt() * (T - 1)).round().long()
    
    #     # --- Delta
    #     delta = max(1, int(delta_ratio * T))
    #     delta = min(delta, T // 3)
    
    #     # --- Bornes
    #     high_normal_raw = (t_center - delta).clamp(1, T - 1)
    #     low_noisy_raw   = (t_center + delta).clamp(1, T - 1)
    
    #     high_normal = torch.minimum(high_normal_raw, torch.full_like(high_normal_raw, less))
    #     high_normal = torch.maximum(high_normal, torch.full_like(high_normal, min_high))
    
    #     low_noisy = torch.maximum(low_noisy_raw, torch.full_like(low_noisy_raw, less))
    
        # if verbose:
        #     ratio_mean = ratio.mean().item() * 100
        #     normal_span = high_normal.float().mean().item()
        #     noisy_span = (T - low_noisy).float().mean().item()
        #     avg_noise_level = t_center.float().mean().item()
    
        #     print(
        #         f"[mask_based_ratio] "
        #         f"T={T} | delta={delta} ({delta_ratio*100:.1f}%) | "
        #         f"mean_anomaly={ratio_mean:.2f}% | "
        #         f"mean_t_center={avg_noise_level:.1f} | "
        #         f"normal_levels≈{normal_span:.1f} | "
        #         f"noisy_levels≈{noisy_span:.1f}"
        #     )
    
        return high_normal.to(device), low_noisy.to(device), ratio.to(device)
                
    def norm_guided_one_step_denoising(self, model, x_0, anomaly_label, anomaly_mask, args):
        # self.debug_anomaly_mask_source(anomaly_mask, anomaly_label)
        high_normal, low_noisy, ratio = self.mask_based_ratio(anomaly_mask, args)

        B = x_0.shape[0]
        T = self.num_timesteps  # doit être == args["T"]
        
        # sécurité device/dtype
        high_normal = high_normal.to(x_0.device)
        low_noisy = low_noisy.to(x_0.device)
        
        # normal_t par image : [0, high_normal[i])
        u = torch.rand(B, device=x_0.device)
        normal_t = (u * high_normal.float()).floor().long().clamp(0, T - 1)
        
        # noisier_t par image : [low_noisy[i], T)
        span = (T - low_noisy).clamp(min=1)
        v = torch.rand(B, device=x_0.device)
        noisier_t = (low_noisy + (v * span.float()).floor().long()).clamp(0, T - 1)
        
        # (optionnel mais conseillé) garantir noisier_t > normal_t
        noisier_t = torch.maximum(noisier_t, normal_t + 1).clamp(0, T - 1)
       
        normal_loss, x_normal_t, estimate_noise_normal = self.calc_loss(
            model, x_0, normal_t)
        noisier_loss, x_noiser_t, estimate_noise_noisier = self.calc_loss(
            model, x_0, noisier_t)

        pred_x_0_noisier = self.predict_x_0_from_eps(
            x_noiser_t, noisier_t, estimate_noise_noisier).clamp(-1, 1)
        pred_x_t_noisier = self.sample_q(
            pred_x_0_noisier, normal_t, estimate_noise_normal)

        # Only calculate the noise loss of normal samples according to formula 9.
        loss = (normal_loss["loss"]+noisier_loss["loss"]
                )[anomaly_label == 0].mean()
        # When the batch size is small, it may lead to an entire batch consisting solely of abnormal samples
        # If they are all abnormal samples, set loss to 0.
        if torch.isnan(loss):
            loss.fill_(0.0)

        estimate_noise_hat = estimate_noise_normal - \
            extract(self.sqrt_one_minus_alphas_cumprod, normal_t, x_normal_t.shape,
                    x_0.device) * args["condition_w"] * (pred_x_t_noisier-x_normal_t)
        pred_x_0_norm_guided = self.predict_x_0_from_eps(
            x_normal_t, normal_t, estimate_noise_hat).clamp(-1, 1)

        return loss, pred_x_0_norm_guided, normal_t, x_normal_t, x_noiser_t
    

    def norm_guided_one_step_denoising_eval(self, model, x_0, normal_t, noisier_t, args):

        normal_loss, x_normal_t, estimate_noise_normal = self.calc_loss(
            model, x_0, normal_t)
        noisier_loss, x_noisier_t, estimate_noise_noisier = self.calc_loss(
            model, x_0, noisier_t)

        pred_x_0_noisier = self.predict_x_0_from_eps(
            x_noisier_t, noisier_t, estimate_noise_noisier).clamp(-1, 1)
        pred_x_t_noisier = self.sample_q(
            pred_x_0_noisier, normal_t, estimate_noise_normal)

        loss = (normal_loss["loss"]+noisier_loss["loss"]).mean()
        pred_x_0_normal = self.predict_x_0_from_eps(
            x_normal_t, normal_t, estimate_noise_normal).clamp(-1, 1)
        estimate_noise_hat = estimate_noise_normal - \
            extract(self.sqrt_one_minus_alphas_cumprod, normal_t, x_0.shape,
                    x_0.device) * args["condition_w"] * (pred_x_t_noisier-x_normal_t)
        pred_x_0_norm_guided = self.predict_x_0_from_eps(
            x_normal_t, normal_t, estimate_noise_hat).clamp(-1, 1)

        return loss, pred_x_0_norm_guided, pred_x_0_normal, pred_x_0_noisier, x_normal_t, x_noisier_t, pred_x_t_noisier

    def noise_t(self, model, x_0, t, args):
        loss, x_t, estimate_noise = self.calc_loss(model, x_0, t)
        loss = (loss["loss"]).mean()
        pred_x_0 = self.predict_x_0_from_eps(
            x_t, t, estimate_noise).clamp(-1, 1)
        return loss, pred_x_0, x_t
