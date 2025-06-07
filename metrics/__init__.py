from metrics.metrics_long_range import compute_all_metrics, setup_optimizer
import numpy as np
import torch


def evaluate_model_uncond(real_sig, gen_sig, args):
    """Evaluate an unconditional model on the given dataset.

    For short-term datasets (e.g. sine, stock, mujoco, energy) discriminative
    and predictive scores are computed.
    For long-term datasets (e.g. ``fred_md``) marginal scores are returned.

    Returns a dictionary mapping metric names to values.
    """

    if args.dataset in ['stock', 'sine', 'mujoco', 'energy']:
        # proceed with short term evaluation
        metric_iteration = 10
        from metrics.discriminative_torch import discriminative_score_metrics
        # For deterministic results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        disc_res = []
        for _ in range(metric_iteration):
            dsc = discriminative_score_metrics(real_sig, gen_sig, args)
            disc_res.append(dsc)
        disc_mean = np.round(np.mean(disc_res), 4)
        disc_std = np.round(np.std(disc_res), 4)
        from metrics.predictive_metrics import predictive_score_metrics
        predictive_score = list()
        for _ in range(metric_iteration):
            temp_pred = predictive_score_metrics(real_sig, gen_sig)
            predictive_score.append(temp_pred)
        pred_mean = np.round(np.mean(predictive_score), 4)
        pred_std = np.round(np.std(predictive_score), 4)
        return {
            'disc_mean': disc_mean,
            'disc_std': disc_std,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
        }

    else:
        # proceed with long term evaluation
        # conversion to meet benchmark requirements:
        real_sig, gen_sig = (
            torch.Tensor(real_sig).float(),
            torch.Tensor(gen_sig).float(),
        )
        scores = compute_all_metrics(
            real_sig,
            gen_sig,
            setup_optimizer,
            torch.nn.Sigmoid()
            if args.dataset == 'temperature_rain'
            else torch.nn.Identity(),
            args.device,
        )
        return scores
