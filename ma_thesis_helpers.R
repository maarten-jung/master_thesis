
## Author: Maarten L. Jung
## 
## Helper functions for analyses_and_plots.R

library(fitdistrplus)


# computes the sample skewness
skewness <- function(x) {
  mean((x - mean(x))^3) / sd(x)^3
}

# computes the KL divergernce from p to q (for discrete p and q)
kl_div <- function(p, q) {
  sum(ifelse(p != 0, p * log(p / q), 0))
}

# Freedman-Diaconis rule to calculate optimal histogram bin widths
freedman_diaconis <- function(x) {
  2 * IQR(x) / length(x)^(1/3)
}


# pdf of the 3-parameter lognormal distribution
#
# standard parameterization:
# 1/((x-theta)*simga*sqrt(2*pi)) * exp(-0.5*(ln(x-theta)-mu)^2/sigma^2)
#
# parameterization used in the functions below (delta = -theta):
# 1/((x+delta)*simga*sqrt(2*pi)) * exp(-0.5*(ln(x+delta)-mu)^2/sigma^2)


# pdf of the 3-parameter lognormal distribution
dshiftlnorm <- function(x, mu, sigma, delta, log = FALSE) {
  dlnorm(x + delta, mu, sigma, log = log)
}

# cdf of the 3-parameter lognormal distribution
pshiftlnorm <- function(q, mu, sigma, delta, log.p = FALSE) {
  plnorm(q + delta, mu, sigma, log.p = log.p)
}

# fits a 3-parameter lognormal distribution 
fit_shiftlnrom <- function(x) {
  x <- x[!is.na(x)]
  delta_init <- 1 - min(x)
  lnorm_fit <- fitdist(x + delta_init, "lnorm")
  start <- list(mu = as.numeric(lnorm_fit$estimate["meanlog"]),  
                sigma = as.numeric(lnorm_fit$estimate["sdlog"]), 
                delta = delta_init)
  shiftlnorm_fit <- fitdist(x, "shiftlnorm", start = start, 
                            lower = c(-Inf, 0, -min(x)))
  shiftlnorm_fit$estimate
}

# fits a 3-parameter lognormal distribution, calculates the densities at 
# n evenly spaced values within the range of x, and scales them to match the 
# histogram bin counts 
interpolate_shiftlnorm <- function(x, n = 101) {
  x <- x[!is.na(x)]
  shiftlnorm_param <- fit_shiftlnrom(x)
  shiftlnorm_support <- seq(min(x), max(x), length.out = n)
  data.frame(x = shiftlnorm_support,
             y = length(x) * freedman_diaconis(x) # scaling factor 
             * do.call(dshiftlnorm,
                       c(list(shiftlnorm_support),
                         shiftlnorm_param)))
}

# fits a 3-parameter lognormal distribution and returns a data.frame
# with the corresponding parameters of the standard parameterization
get_shiftlornm_standard_param <- function(x) {
  shiftlnorm_param <- fit_shiftlnrom(x)
  data.frame(theta = -as.numeric(shiftlnorm_param["delta"]),
             mu = as.numeric(shiftlnorm_param["mu"]),
             simga = as.numeric(shiftlnorm_param["sigma"]))  
}
