import numpy as np
import scipy.stats

MIN_POSITIVE_MU = 1e-8
MIN_PRED_VALUE = 1e-4


def variates_from_mean_variance(
    mu: np.ndarray,
    sigma2: np.ndarray,
    distribution: str,
    state: np.random.Generator,
) -> np.ndarray:
    zero_means = mu < MIN_POSITIVE_MU

    if distribution.lower() == "normal":
        distr = scipy.stats.norm(loc=mu, scale=np.sqrt(sigma2))
    elif distribution.lower() == "lognormal":
        scale = (mu ** 2) / np.sqrt((mu ** 2) + sigma2)
        s2 = np.log(1 + sigma2 / (mu ** 2))
        distr = scipy.stats.lognorm(scale=scale, s=np.sqrt(s2))
    elif distribution.lower() == "gamma":
        shape_param = mu ** 2 / sigma2
        rate_param = mu / sigma2
        rate_param[zero_means] = 1
        distr = scipy.stats.gamma(a=shape_param, scale=1 / rate_param)
    else:
        raise Exception(f"Unimplemented distribution {distribution.lower()}")
    return np.maximum(distr.rvs(random_state=state), MIN_PRED_VALUE)
