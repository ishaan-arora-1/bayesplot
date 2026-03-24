# =============================================================================
# Deep Testing: Prior vs Posterior Prototypes
# =============================================================================
# Tests: edge cases, input formats, transformations, bounds, color schemes,
# stress tests, scale mismatch strategies, real rstanarm models, and
# side-by-side comparisons of all 5 prototypes.
# =============================================================================

library(bayesplot)
library(ggplot2)
library(dplyr)
library(patchwork)
library(posterior)

source("proposal/prototypes/prototype_prior_posterior.R")

outdir <- "proposal/prototypes/deep_tests"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

save_plot <- function(name, plot, w = 10, h = 6) {
  path <- file.path(outdir, paste0(name, ".png"))
  ggsave(path, plot, width = w, height = h, dpi = 150, bg = "white")
  cat("  Saved:", path, "\n")
}


# =============================================================================
# TEST 1: Input format compatibility
# =============================================================================
cat("\n========== TEST 1: Input Formats ==========\n")

set.seed(42)
n <- 1000

# Ground truth draws
post_mat <- cbind(alpha = rnorm(n, 3, 0.5), beta = rnorm(n, -1, 0.3))
prior_mat <- cbind(alpha = rnorm(n, 0, 5), beta = rnorm(n, 0, 3))

# Format A: plain matrix (the default)
cat("  A) Matrix...\n")
d1 <- mcmc_prior_posterior_data(post_mat, prior_mat)
stopifnot(nrow(d1) == 4000, ncol(d1) == 4)
cat("     OK: ", nrow(d1), "rows\n")

# Format B: 3D array (iter x chain x param)
cat("  B) 3D array...\n")
post_arr <- array(post_mat, dim = c(250, 4, 2),
                  dimnames = list(NULL, paste0("chain:", 1:4), c("alpha", "beta")))
prior_arr <- array(prior_mat, dim = c(500, 2, 2),
                   dimnames = list(NULL, paste0("chain:", 1:2), c("alpha", "beta")))
d2 <- mcmc_prior_posterior_data(post_arr, prior_arr)
stopifnot(nrow(d2) == 4000)
cat("     OK: ", nrow(d2), "rows\n")

# Format C: posterior::draws_array
cat("  C) draws_array...\n")
post_draws <- posterior::as_draws_array(post_arr)
d3 <- mcmc_prior_posterior_data(post_draws, prior_mat)
stopifnot(nrow(d3) == 4000)
cat("     OK: ", nrow(d3), "rows\n")

# Format D: posterior::draws_df
cat("  D) draws_df...\n")
post_draws_df <- posterior::as_draws_df(post_draws)
d4 <- mcmc_prior_posterior_data(post_draws_df, prior_mat)
cat("     OK: ", nrow(d4), "rows\n")

# Format E: data.frame with Chain column
cat("  E) data.frame with Chain column...\n")
post_df_chain <- data.frame(post_mat, Chain = rep(1:4, each = 250))
d5 <- mcmc_prior_posterior_data(post_df_chain, prior_mat)
cat("     OK: ", nrow(d5), "rows\n")

# Format F: mismatched draw counts (prior has fewer draws than posterior)
cat("  F) Mismatched draw counts...\n")
prior_small <- cbind(alpha = rnorm(200, 0, 5), beta = rnorm(200, 0, 3))
d6 <- mcmc_prior_posterior_data(post_mat, prior_small)
prior_count <- sum(d6$Source == "Prior")
post_count <- sum(d6$Source == "Posterior")
cat("     OK: prior=", prior_count, " posterior=", post_count, "\n")

cat("  All input format tests passed!\n")


# =============================================================================
# TEST 2: Edge cases
# =============================================================================
cat("\n========== TEST 2: Edge Cases ==========\n")

# Single parameter
cat("  A) Single parameter...\n")
post_1 <- cbind(mu = rnorm(n, 5, 1))
prior_1 <- cbind(mu = rnorm(n, 0, 10))

p_single <- mcmc_prior_posterior_dens(post_1, prior_1) +
  ggtitle("Edge: Single parameter")
save_plot("edge_single_param", p_single, w = 6, h = 4)

# Many parameters (10+)
cat("  B) Many parameters (12)...\n")
param_names <- paste0("beta[", 1:12, "]")
post_many <- matrix(rnorm(n * 12, mean = rep(1:12 / 3, each = n), sd = 0.3),
                    ncol = 12, dimnames = list(NULL, param_names))
prior_many <- matrix(rnorm(n * 12, 0, 5),
                     ncol = 12, dimnames = list(NULL, param_names))

p_many_dens <- mcmc_prior_posterior_dens(post_many, prior_many) +
  ggtitle("Edge: 12 parameters — density overlay")
save_plot("edge_many_params_dens", p_many_dens, w = 14, h = 10)

p_many_int <- mcmc_prior_posterior_intervals(post_many, prior_many) +
  ggtitle("Edge: 12 parameters — intervals")
save_plot("edge_many_params_intervals", p_many_int, w = 8, h = 8)

p_many_areas <- mcmc_prior_posterior_areas(post_many, prior_many) +
  ggtitle("Edge: 12 parameters — areas")
save_plot("edge_many_params_areas", p_many_areas, w = 10, h = 8)

# Multimodal posterior
cat("  C) Multimodal posterior...\n")
post_bimodal <- cbind(
  theta = c(rnorm(n/2, -2, 0.5), rnorm(n/2, 2, 0.5))
)
prior_bimodal <- cbind(theta = rnorm(n, 0, 5))

p_bimodal <- mcmc_prior_posterior_dens(post_bimodal, prior_bimodal) +
  ggtitle("Edge: Bimodal posterior")
save_plot("edge_bimodal", p_bimodal, w = 6, h = 4)

p_bimodal_hist <- mcmc_prior_posterior_hist(post_bimodal, prior_bimodal) +
  ggtitle("Edge: Bimodal posterior — histogram")
save_plot("edge_bimodal_hist", p_bimodal_hist, w = 6, h = 4)

# Nearly identical prior and posterior (data barely informative)
cat("  D) Uninformative data (prior ≈ posterior)...\n")
post_same <- cbind(mu = rnorm(n, 0.1, 4.9))
prior_same <- cbind(mu = rnorm(n, 0, 5))

p_same <- mcmc_prior_posterior_dens(post_same, prior_same) +
  ggtitle("Edge: Prior ≈ Posterior (uninformative data)")
save_plot("edge_uninformative", p_same, w = 6, h = 4)

# Partial parameter overlap (prior has extra params not in posterior)
cat("  E) Partial parameter overlap...\n")
post_partial <- cbind(alpha = rnorm(n, 2, 0.5), beta = rnorm(n, -1, 0.3))
prior_partial <- cbind(alpha = rnorm(n, 0, 5), beta = rnorm(n, 0, 3),
                       gamma = rnorm(n, 0, 1))  # extra param
d_partial <- mcmc_prior_posterior_data(post_partial, prior_partial)
cat("     Shared params:", paste(levels(d_partial$Parameter), collapse = ", "), "\n")
stopifnot(all(levels(d_partial$Parameter) %in% c("alpha", "beta")))
cat("     OK: extra 'gamma' correctly excluded\n")

cat("  All edge case tests passed!\n")


# =============================================================================
# TEST 3: Parameter selection (pars, regex_pars)
# =============================================================================
cat("\n========== TEST 3: Parameter Selection ==========\n")

post_sel <- cbind(
  `alpha` = rnorm(n, 2, 0.5),
  `beta[1]` = rnorm(n, -1, 0.3),
  `beta[2]` = rnorm(n, 0.5, 0.2),
  `beta[3]` = rnorm(n, 1.2, 0.4),
  `sigma` = rgamma(n, 20, 10)
)
prior_sel <- cbind(
  `alpha` = rnorm(n, 0, 10),
  `beta[1]` = rnorm(n, 0, 5),
  `beta[2]` = rnorm(n, 0, 5),
  `beta[3]` = rnorm(n, 0, 5),
  `sigma` = rexp(n, 1)
)

# Select specific parameters
cat("  A) pars = c('alpha', 'sigma')...\n")
p_sel1 <- mcmc_prior_posterior_dens(post_sel, prior_sel,
                                     pars = c("alpha", "sigma")) +
  ggtitle("Selection: pars = c('alpha', 'sigma')")
save_plot("select_explicit", p_sel1, w = 8, h = 4)

# Regex selection
cat("  B) regex_pars = 'beta'...\n")
p_sel2 <- mcmc_prior_posterior_dens(post_sel, prior_sel,
                                     regex_pars = "beta") +
  ggtitle("Selection: regex_pars = 'beta'")
save_plot("select_regex", p_sel2, w = 10, h = 4)

# Combined
cat("  C) pars + regex_pars...\n")
p_sel3 <- mcmc_prior_posterior_intervals(post_sel, prior_sel,
                                          pars = "alpha",
                                          regex_pars = "beta") +
  ggtitle("Selection: pars='alpha' + regex_pars='beta'")
save_plot("select_combined", p_sel3, w = 8, h = 5)

cat("  Parameter selection tests passed!\n")


# =============================================================================
# TEST 4: Transformations
# =============================================================================
cat("\n========== TEST 4: Transformations ==========\n")

post_trans <- cbind(
  mu = rnorm(n, 5, 1),
  sigma = rgamma(n, 20, 10)
)
prior_trans <- cbind(
  mu = rnorm(n, 0, 10),
  sigma = rexp(n, 1)
)

# Log transformation on sigma
cat("  A) log(sigma) transformation...\n")
p_trans <- mcmc_prior_posterior_dens(
  post_trans, prior_trans,
  transformations = list(sigma = "log")
) + ggtitle("Transformation: log(sigma)")
save_plot("transform_log_sigma", p_trans, w = 8, h = 4)

# Custom transformation function
cat("  B) Custom function transformation...\n")
p_trans2 <- mcmc_prior_posterior_dens(
  post_trans, prior_trans,
  transformations = list(sigma = function(x) x^2)
) + ggtitle("Transformation: sigma^2 (custom function)")
save_plot("transform_custom", p_trans2, w = 8, h = 4)

# Transformation applied to all parameters
cat("  C) Single transform applied to all...\n")
p_trans3 <- mcmc_prior_posterior_dens(
  post_trans, prior_trans,
  transformations = "log"
) + ggtitle("Transformation: log() on all parameters")
save_plot("transform_all", p_trans3, w = 8, h = 4)

cat("  Transformation tests passed!\n")


# =============================================================================
# TEST 5: Scale mismatch scenarios (the core challenge)
# =============================================================================
cat("\n========== TEST 5: Scale Mismatch ==========\n")

# Varying degrees of prior width
widths <- c(1, 5, 20, 100, 500)
post_fixed <- cbind(theta = rnorm(n, 3, 0.5))

plots_mismatch <- lapply(widths, function(w) {
  pr <- cbind(theta = rnorm(n, 0, w))
  mcmc_prior_posterior_dens(post_fixed, pr) +
    ggtitle(sprintf("Prior SD = %d", w)) +
    theme(legend.position = "none")
})

p_mismatch <- wrap_plots(plots_mismatch, nrow = 1) +
  plot_annotation(title = "Scale mismatch: increasing prior width",
                  subtitle = "Posterior is always N(3, 0.5)")
save_plot("mismatch_progression", p_mismatch, w = 16, h = 4)

# Same with the paired approach (handles it better)
plots_paired <- lapply(widths, function(w) {
  pr <- cbind(theta = rnorm(n, 0, w))
  mcmc_prior_posterior_paired(post_fixed, pr) +
    ggtitle(sprintf("Prior SD = %d", w)) +
    theme(legend.position = "none")
})

p_paired_mismatch <- wrap_plots(plots_paired, nrow = 1) +
  plot_annotation(title = "Same mismatch, paired layout",
                  subtitle = "Each density gets its own y-scale")
save_plot("mismatch_progression_paired", p_paired_mismatch, w = 16, h = 5)

# Same with intervals (most robust)
plots_int <- lapply(widths, function(w) {
  pr <- cbind(theta = rnorm(n, 0, w))
  mcmc_prior_posterior_intervals(post_fixed, pr) +
    ggtitle(sprintf("Prior SD = %d", w)) +
    theme(legend.position = "none")
})

p_int_mismatch <- wrap_plots(plots_int, nrow = 1) +
  plot_annotation(title = "Same mismatch, interval approach",
                  subtitle = "Intervals remain readable regardless of scale")
save_plot("mismatch_progression_intervals", p_int_mismatch, w = 16, h = 3.5)

cat("  Scale mismatch tests saved!\n")


# =============================================================================
# TEST 6: Bounded parameters (positive-only, [0,1], etc.)
# =============================================================================
cat("\n========== TEST 6: Bounded Parameters ==========\n")

# Positive-only: sigma ~ Exp(1), posterior concentrated
post_pos <- cbind(sigma = rgamma(n, 50, 25))
prior_pos <- cbind(sigma = rexp(n, 1))

cat("  A) Positive parameter (no bounds arg)...\n")
p_pos1 <- mcmc_prior_posterior_dens(post_pos, prior_pos) +
  ggtitle("Positive parameter: no bounds specified")
save_plot("bounds_positive_nobounds", p_pos1, w = 6, h = 4)

cat("  B) Positive parameter (bounds = c(0, NA))...\n")
p_pos2 <- mcmc_prior_posterior_dens(post_pos, prior_pos, bounds = c(0, NA)) +
  ggtitle("Positive parameter: bounds = c(0, NA)")
save_plot("bounds_positive_bounded", p_pos2, w = 6, h = 4)

# Probability parameter [0, 1]
post_prob <- cbind(p = rbeta(n, 30, 10))
prior_prob <- cbind(p = rbeta(n, 1, 1))

cat("  C) Probability parameter...\n")
p_prob <- mcmc_prior_posterior_dens(post_prob, prior_prob) +
  ggtitle("Probability parameter [0,1]")
save_plot("bounds_probability", p_prob, w = 6, h = 4)

p_prob_hist <- mcmc_prior_posterior_hist(post_prob, prior_prob) +
  ggtitle("Probability parameter — histogram + prior")
save_plot("bounds_probability_hist", p_prob_hist, w = 6, h = 4)

cat("  Bounded parameter tests saved!\n")


# =============================================================================
# TEST 7: Color schemes
# =============================================================================
cat("\n========== TEST 7: Color Schemes ==========\n")

d_color <- generate_test_data()$simple
schemes <- c("blue", "red", "green", "purple", "gray", "brightblue",
             "teal", "mix-blue-red")

plots_colors <- lapply(schemes, function(s) {
  color_scheme_set(s)
  mcmc_prior_posterior_dens(d_color$posterior, d_color$prior,
                            pars = c("alpha", "sigma")) +
    ggtitle(s) +
    theme(legend.position = "none", plot.title = element_text(size = 10))
})
color_scheme_set("brightblue")

p_colors <- wrap_plots(plots_colors, nrow = 2) +
  plot_annotation(title = "All prototypes across color schemes (density overlay)")
save_plot("color_schemes_dens", p_colors, w = 16, h = 8)

# Areas across color schemes
plots_areas_colors <- lapply(schemes[1:4], function(s) {
  color_scheme_set(s)
  mcmc_prior_posterior_areas(d_color$posterior, d_color$prior) +
    ggtitle(s)
})
color_scheme_set("brightblue")

p_areas_colors <- wrap_plots(plots_areas_colors, nrow = 2) +
  plot_annotation(title = "Areas prototype across color schemes")
save_plot("color_schemes_areas", p_areas_colors, w = 14, h = 10)

cat("  Color scheme tests saved!\n")


# =============================================================================
# TEST 8: Side-by-side comparison of ALL 5 prototypes on same data
# =============================================================================
cat("\n========== TEST 8: Side-by-Side Comparison ==========\n")

color_scheme_set("brightblue")
d_compare <- generate_test_data()$simple

p_comp1 <- mcmc_prior_posterior_dens(d_compare$posterior, d_compare$prior) +
  ggtitle("1. Density Overlay") +
  theme(legend.position = "none")

p_comp2 <- mcmc_prior_posterior_areas(d_compare$posterior, d_compare$prior) +
  ggtitle("2. Areas + Prior Outline")

p_comp3 <- mcmc_prior_posterior_intervals(d_compare$posterior, d_compare$prior) +
  ggtitle("3. Interval Comparison")

p_comp4 <- mcmc_prior_posterior_hist(d_compare$posterior, d_compare$prior) +
  ggtitle("4. Histogram + Prior") +
  theme(legend.position = "none")

p_comp5 <- mcmc_prior_posterior_paired(d_compare$posterior, d_compare$prior) +
  ggtitle("5. Paired Density")

p_all5 <- (p_comp1 | p_comp4) / (p_comp2 | p_comp3) / (p_comp5) +
  plot_annotation(
    title = "All 5 Prototypes — Same Data (Simple Regression)",
    subtitle = "alpha ~ N(0,10), beta ~ N(0,5), sigma ~ Exp(1)"
  )
save_plot("comparison_all5_simple", p_all5, w = 16, h = 16)

# Same for hierarchical
d_hier <- generate_test_data()$hierarchical
p_h1 <- mcmc_prior_posterior_dens(d_hier$posterior, d_hier$prior) +
  ggtitle("1. Density Overlay") + theme(legend.position = "none")
p_h3 <- mcmc_prior_posterior_intervals(d_hier$posterior, d_hier$prior) +
  ggtitle("3. Intervals")
p_h2 <- mcmc_prior_posterior_areas(d_hier$posterior, d_hier$prior) +
  ggtitle("2. Areas")
p_h4 <- mcmc_prior_posterior_hist(d_hier$posterior, d_hier$prior) +
  ggtitle("4. Hist + Prior") + theme(legend.position = "none")

p_hier_comp <- (p_h1 + p_h4) / (p_h2 + p_h3) +
  plot_annotation(title = "All Prototypes — Hierarchical Model (5 params)")
save_plot("comparison_all_hierarchical", p_hier_comp, w = 18, h = 14)

cat("  Side-by-side comparisons saved!\n")


# =============================================================================
# TEST 9: Prior-posterior overlap metric
# =============================================================================
cat("\n========== TEST 9: Overlap Metric ==========\n")

# Test overlap across different scenarios
scenarios <- list(
  "identical (overlap~1)" = list(
    post = cbind(x = rnorm(n, 0, 1)),
    prior = cbind(x = rnorm(n, 0, 1))
  ),
  "slight update" = list(
    post = cbind(x = rnorm(n, 0.5, 0.9)),
    prior = cbind(x = rnorm(n, 0, 1))
  ),
  "moderate update" = list(
    post = cbind(x = rnorm(n, 2, 0.5)),
    prior = cbind(x = rnorm(n, 0, 2))
  ),
  "strong update" = list(
    post = cbind(x = rnorm(n, 5, 0.3)),
    prior = cbind(x = rnorm(n, 0, 5))
  ),
  "disjoint (overlap~0)" = list(
    post = cbind(x = rnorm(n, 100, 0.1)),
    prior = cbind(x = rnorm(n, 0, 1))
  )
)

cat("  Overlap metric across scenarios:\n")
for (nm in names(scenarios)) {
  s <- scenarios[[nm]]
  ov <- prior_posterior_overlap(s$post, s$prior)
  cat(sprintf("    %-30s overlap = %.4f\n", nm, ov$Overlap))
}

# Visualize the overlap spectrum
overlap_plots <- lapply(names(scenarios), function(nm) {
  s <- scenarios[[nm]]
  ov <- prior_posterior_overlap(s$post, s$prior)
  mcmc_prior_posterior_dens(s$post, s$prior) +
    ggtitle(sprintf("%s\noverlap = %.3f", nm, ov$Overlap)) +
    theme(legend.position = "none",
          plot.title = element_text(size = 9))
})

p_overlap <- wrap_plots(overlap_plots, nrow = 1) +
  plot_annotation(title = "Prior-Posterior Overlap Spectrum",
                  subtitle = "From identical (≈1) to disjoint (≈0)")
save_plot("overlap_spectrum", p_overlap, w = 18, h = 4)

cat("  Overlap tests saved!\n")


# =============================================================================
# TEST 10: Real model — rstanarm
# =============================================================================
cat("\n========== TEST 10: rstanarm Integration ==========\n")

if (requireNamespace("rstanarm", quietly = TRUE)) {
  library(rstanarm)

  cat("  Fitting rstanarm model...\n")
  fit <- suppressWarnings(stan_glm(
    mpg ~ wt + hp + am,
    data = mtcars,
    prior = normal(0, 5),
    prior_intercept = normal(20, 10),
    prior_aux = exponential(1),
    seed = 123,
    chains = 2,
    iter = 1000,
    refresh = 0
  ))

  # Extract posterior draws
  post_draws <- as.matrix(fit)
  # Remove lp__ column
  post_draws <- post_draws[, !colnames(post_draws) %in% "log-posterior", drop = FALSE]

  # Extract prior draws using rstanarm's update with prior_PD
  cat("  Sampling from prior...\n")
  fit_prior <- suppressWarnings(update(
    fit,
    prior_PD = TRUE,
    chains = 2,
    iter = 1000,
    refresh = 0
  ))
  prior_draws_mat <- as.matrix(fit_prior)
  prior_draws_mat <- prior_draws_mat[, !colnames(prior_draws_mat) %in% "log-posterior", drop = FALSE]

  # Match column names
  shared <- intersect(colnames(post_draws), colnames(prior_draws_mat))
  cat("  Shared parameters:", paste(shared, collapse = ", "), "\n")

  # Density overlay
  cat("  Generating plots...\n")
  p_stan1 <- mcmc_prior_posterior_dens(post_draws[, shared], prior_draws_mat[, shared]) +
    ggtitle("rstanarm: Density overlay — mpg ~ wt + hp + am")
  save_plot("rstanarm_dens", p_stan1, w = 12, h = 6)

  # Areas
  p_stan2 <- mcmc_prior_posterior_areas(post_draws[, shared], prior_draws_mat[, shared]) +
    ggtitle("rstanarm: Areas — mpg ~ wt + hp + am")
  save_plot("rstanarm_areas", p_stan2, w = 10, h = 6)

  # Intervals
  p_stan3 <- mcmc_prior_posterior_intervals(post_draws[, shared], prior_draws_mat[, shared]) +
    ggtitle("rstanarm: Intervals — mpg ~ wt + hp + am")
  save_plot("rstanarm_intervals", p_stan3, w = 8, h = 5)

  # Histogram
  p_stan4 <- mcmc_prior_posterior_hist(post_draws[, shared], prior_draws_mat[, shared]) +
    ggtitle("rstanarm: Histogram + prior — mpg ~ wt + hp + am")
  save_plot("rstanarm_hist", p_stan4, w = 12, h = 6)

  # Paired
  p_stan5 <- mcmc_prior_posterior_paired(post_draws[, shared], prior_draws_mat[, shared]) +
    ggtitle("rstanarm: Paired density — mpg ~ wt + hp + am")
  save_plot("rstanarm_paired", p_stan5, w = 12, h = 6)

  # With parameter selection
  p_stan_sel <- mcmc_prior_posterior_dens(
    post_draws[, shared], prior_draws_mat[, shared],
    pars = c("wt", "hp")
  ) + ggtitle("rstanarm: Selected params (wt, hp)")
  save_plot("rstanarm_selected", p_stan_sel, w = 8, h = 4)

  # With log transform on sigma
  p_stan_log <- mcmc_prior_posterior_dens(
    post_draws[, shared], prior_draws_mat[, shared],
    pars = "sigma",
    transformations = list(sigma = "log")
  ) + ggtitle("rstanarm: log(sigma)")
  save_plot("rstanarm_log_sigma", p_stan_log, w = 6, h = 4)

  # Overlap
  ov <- prior_posterior_overlap(post_draws[, shared], prior_draws_mat[, shared])
  cat("  Overlap:\n")
  print(ov)

  # Compare with rstanarm's built-in posterior_vs_prior
  p_rstanarm_native <- posterior_vs_prior(fit, prob = 0.9, color_by = "vs")
  p_rstanarm_native <- p_rstanarm_native + ggtitle("rstanarm native: posterior_vs_prior()")
  save_plot("rstanarm_native_comparison", p_rstanarm_native, w = 10, h = 6)

  cat("  rstanarm tests saved!\n")
} else {
  cat("  rstanarm not available, skipping.\n")
}


# =============================================================================
# TEST 11: Stress test — very large number of draws
# =============================================================================
cat("\n========== TEST 11: Performance / Stress ==========\n")

for (n_stress in c(1000, 10000, 50000)) {
  post_big <- cbind(alpha = rnorm(n_stress, 3, 0.5), beta = rnorm(n_stress, -1, 0.3))
  prior_big <- cbind(alpha = rnorm(n_stress, 0, 5), beta = rnorm(n_stress, 0, 3))

  t0 <- proc.time()
  p <- mcmc_prior_posterior_dens(post_big, prior_big)
  # Force ggplot to build (otherwise it's lazy)
  invisible(ggplot_build(p))
  dt <- (proc.time() - t0)["elapsed"]
  cat(sprintf("  n_draws = %6d  =>  %.3f sec\n", n_stress, dt))
}


# =============================================================================
# TEST 12: Different density estimation settings
# =============================================================================
cat("\n========== TEST 12: Density Estimation Settings ==========\n")

d_dens <- generate_test_data()$simple

bw_methods <- c("nrd0", "SJ", "ucv")
plots_bw <- lapply(bw_methods, function(b) {
  tryCatch({
    mcmc_prior_posterior_dens(d_dens$posterior, d_dens$prior,
                              pars = "sigma", bw = b) +
      ggtitle(paste("bw =", b)) +
      theme(legend.position = "none")
  }, error = function(e) {
    ggplot() + ggtitle(paste("bw =", b, "- ERROR"))
  })
})

p_bw <- wrap_plots(plots_bw, nrow = 1) +
  plot_annotation(title = "Different bandwidth methods (sigma parameter)")
save_plot("density_bandwidth_methods", p_bw, w = 12, h = 4)

adjust_vals <- c(0.5, 1, 2, 4)
plots_adj <- lapply(adjust_vals, function(a) {
  mcmc_prior_posterior_dens(d_dens$posterior, d_dens$prior,
                            pars = "sigma", adjust = a) +
    ggtitle(paste("adjust =", a)) +
    theme(legend.position = "none")
})

p_adj <- wrap_plots(plots_adj, nrow = 1) +
  plot_annotation(title = "Different bandwidth adjustments (sigma parameter)")
save_plot("density_adjust_values", p_adj, w = 12, h = 4)

cat("  Density estimation tests saved!\n")


# =============================================================================
# TEST 13: Interval prototype options
# =============================================================================
cat("\n========== TEST 13: Interval Options ==========\n")

d_int <- generate_test_data()$simple

# Different prob levels
p_int50 <- mcmc_prior_posterior_intervals(d_int$posterior, d_int$prior,
                                           prob = 0.5, prob_outer = 0.9) +
  ggtitle("prob=0.5, prob_outer=0.9")
p_int80 <- mcmc_prior_posterior_intervals(d_int$posterior, d_int$prior,
                                           prob = 0.8, prob_outer = 0.95) +
  ggtitle("prob=0.8, prob_outer=0.95")
p_int_wide <- mcmc_prior_posterior_intervals(d_int$posterior, d_int$prior,
                                              prob = 0.5, prob_outer = 0.99) +
  ggtitle("prob=0.5, prob_outer=0.99")

p_int_options <- (p_int50 | p_int80 | p_int_wide) +
  plot_annotation(title = "Interval comparison: different probability levels")
save_plot("intervals_prob_levels", p_int_options, w = 18, h = 5)

# color_by options
p_by_vs <- mcmc_prior_posterior_intervals(d_int$posterior, d_int$prior,
                                           color_by = "vs") +
  ggtitle("color_by = 'vs'")
p_by_par <- mcmc_prior_posterior_intervals(d_int$posterior, d_int$prior,
                                            color_by = "parameter") +
  ggtitle("color_by = 'parameter'")

p_color_options <- (p_by_vs | p_by_par) +
  plot_annotation(title = "Interval comparison: color_by options")
save_plot("intervals_color_options", p_color_options, w = 14, h = 5)

cat("  Interval options tests saved!\n")


# =============================================================================
# SUMMARY
# =============================================================================
cat("\n\n========================================\n")
cat("DEEP TESTING COMPLETE\n")
cat("========================================\n")
cat("All outputs saved to:", outdir, "\n")
files <- list.files(outdir, pattern = "\\.png$")
cat(length(files), "PNG files generated:\n")
for (f in files) cat("  ", f, "\n")
cat("\n")
