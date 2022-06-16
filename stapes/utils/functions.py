# Utility functions used by the Stan program
UTIL_FUNCTIONS = """
functions {
    array[] real mean_variance_log_lik(array[] real obs, array[] real obs_mean, array[] real obs_variance, int family) {
        int N = size(obs);
        real loc_param;
        real scale_param;
        real shape_param;
        real rate_param;
        array[N] real log_lik;

        if (family == 1) {
            /* Family 1 is normal. */
            for (n in 1:N) {
                loc_param = obs_mean[n];
                scale_param = sqrt(obs_variance[n]);
                log_lik[n] = normal_lpdf(obs[n] | loc_param, scale_param);
            }
        } else if (family == 2) {
            /* Family 2 is log-normal. */
            for (n in 1:N) {
                loc_param = log(obs_mean[n] ^ 2 / sqrt(obs_mean[n]^2 + obs_variance[n]));
                scale_param = sqrt(log(1 + obs_variance[n] / obs_mean[n]^2));
                log_lik[n] = lognormal_lpdf(obs[n] | loc_param, scale_param);
            }
        } else if (family == 3) {
            /* Family 3 is gamma. */
            for (n in 1:N) {
                shape_param = obs_mean[n] ^ 2 / obs_variance[n];
                rate_param = obs_mean[n] / obs_variance[n];
                log_lik[n] = gamma_lpdf(obs[n] | shape_param, rate_param);
            }
        } else {
            reject("Positive variables modeled with mean and variance are not compatible with family=", family);
        }

        return log_lik;
    }
}
"""
