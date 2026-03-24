# =============================================================================
# Prototype: Prior vs Posterior Distribution Comparison Plots for bayesplot
# =============================================================================
#
# This prototype explores multiple visualization approaches for comparing
# prior and posterior parameter distributions. The mentor's guidance:
# "There are many possible ways of doing that, so it could be something that
# requires making different prototypes and trying with different kinds of
# models to see what we prefer."
#
# Approach: Build a data-preparation back-end first, then layer multiple
# plot types on top, following bayesplot's existing architecture:
#   - prepare_mcmc_array() -> melt -> ggplot
#   - _data() functions as public back-ends
#   - color scheme integration via bp_color()
#
# =============================================================================

library(bayesplot)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggridges)
library(posterior)

# Helper: access bayesplot's internal color getter.
# In the actual package code this would just be bp_color() directly.
bp_color <- function(level) {
  scheme <- color_scheme_get()
  level_map <- c(
    l = "light", lh = "light_highlight", m = "mid",
    mh = "mid_highlight", d = "dark", dh = "dark_highlight",
    light = "light", light_highlight = "light_highlight",
    mid = "mid", mid_highlight = "mid_highlight",
    dark = "dark", dark_highlight = "dark_highlight"
  )
  nms <- level_map[level]
  unlist(scheme[nms], use.names = FALSE)
}

# =============================================================================
# SECTION 1: Example data — three model scenarios
# =============================================================================

generate_test_data <- function() {
  set.seed(1234)
  n_draws <- 2000

  # --- Scenario A: Simple regression (location + scale params) ---
  # Priors: alpha ~ Normal(0, 10), beta ~ Normal(0, 5), sigma ~ Exp(1)
  # Posteriors: concentrated after seeing data
  prior_A <- cbind(
    alpha  = rnorm(n_draws, 0, 10),
    beta   = rnorm(n_draws, 0, 5),
    sigma  = rexp(n_draws, 1)
  )
  posterior_A <- cbind(
    alpha  = rnorm(n_draws, 2.3, 0.5),
    beta   = rnorm(n_draws, -1.1, 0.3),
    sigma  = rgamma(n_draws, shape = 20, rate = 10)
  )

  # --- Scenario B: Hierarchical model (shrinkage visible) ---
  # Group-level SD has a half-Cauchy prior; group means have normal prior
  prior_B <- cbind(
    mu_group   = rnorm(n_draws, 0, 5),
    tau        = abs(rcauchy(n_draws, 0, 2)),
    `theta[1]` = rnorm(n_draws, 0, 5),
    `theta[2]` = rnorm(n_draws, 0, 5),
    `theta[3]` = rnorm(n_draws, 0, 5)
  )
  posterior_B <- cbind(
    mu_group   = rnorm(n_draws, 3.1, 0.8),
    tau        = rgamma(n_draws, shape = 4, rate = 2),
    `theta[1]` = rnorm(n_draws, 2.5, 0.6),
    `theta[2]` = rnorm(n_draws, 3.8, 0.7),
    `theta[3]` = rnorm(n_draws, 3.0, 0.5)
  )

  # --- Scenario C: Weakly-informative vs strong data (dramatic update) ---
  # Wide prior barely constraining; data pins the posterior
  prior_C <- cbind(
    effect = rnorm(n_draws, 0, 100),
    noise  = rexp(n_draws, 0.01)
  )
  posterior_C <- cbind(
    effect = rnorm(n_draws, 42.7, 1.2),
    noise  = rgamma(n_draws, shape = 50, rate = 10)
  )

  list(
    simple       = list(prior = prior_A, posterior = posterior_A),
    hierarchical = list(prior = prior_B, posterior = posterior_B),
    wide_prior   = list(prior = prior_C, posterior = posterior_C)
  )
}


# =============================================================================
# SECTION 2: Data preparation layer
# =============================================================================
# This is the foundation — a tidy long-format data frame with columns:
#   Parameter, Value, Source (prior/posterior)
# Follows the pattern of mcmc_areas_data() / mcmc_intervals_data()

mcmc_prior_posterior_data <- function(
    posterior,
    prior,
    pars = character(),
    regex_pars = character(),
    transformations = list()
) {
  # First, figure out which parameters to select BEFORE transformations rename them.
  # We need to select from the raw prior using original names, then transform both.

  # Prepare posterior draws — bayesplot's prepare_mcmc_array handles all formats,
  # selects pars, and applies+renames transformations
  post_array <- bayesplot:::prepare_mcmc_array(
    posterior, pars, regex_pars, transformations
  )
  post_mat <- bayesplot:::merge_chains(post_array)
  post_pars <- colnames(post_mat)

  # Prepare prior draws — convert to matrix first
  if (is.matrix(prior) || is.data.frame(prior)) {
    prior_mat <- as.matrix(prior)
  } else if (is.array(prior) && length(dim(prior)) == 3) {
    prior_mat <- bayesplot:::merge_chains(prior)
  } else if (posterior::is_draws(prior)) {
    prior_arr <- posterior::as_draws_array(prior)
    dim_p <- dim(prior_arr)
    prior_mat <- array(prior_arr, dim = c(prod(dim_p[1:2]), dim_p[3]))
    colnames(prior_mat) <- dimnames(prior_arr)[[3]]
  } else {
    prior_mat <- as.matrix(prior)
  }
  prior_raw_pars <- colnames(prior_mat)

  # Determine the original (pre-transformation) parameter names that were selected.
  # prepare_mcmc_array renames them, e.g. sigma -> log(sigma). We need the originals
  # to select from the prior, then apply the same transform+rename to the prior.
  #
  # Strategy: also run prior through prepare_mcmc_array with the same args.
  # This handles pars/regex_pars selection AND transformations+renaming consistently.

  # Figure out which pars to ask for from the prior. We need the raw names that
  # exist in both posterior (pre-transform) and prior.
  # Re-prepare posterior WITHOUT transformations to get the raw selected names.
  post_array_raw <- bayesplot:::prepare_mcmc_array(
    posterior, pars, regex_pars, transformations = list()
  )
  raw_selected <- colnames(bayesplot:::merge_chains(post_array_raw))

  # Intersect with prior's available parameters (using raw names)
  shared_raw <- intersect(raw_selected, prior_raw_pars)
  if (length(shared_raw) == 0) {
    stop(
      "No matching parameter names between prior and posterior.\n",
      "  Posterior parameters: ", paste(raw_selected, collapse = ", "), "\n",
      "  Prior parameters: ", paste(prior_raw_pars, collapse = ", ")
    )
  }

  # Now run prior through the same machinery with only the shared raw pars
  prior_array <- bayesplot:::prepare_mcmc_array(
    prior_mat, pars = shared_raw, transformations = transformations
  )
  prior_mat <- bayesplot:::merge_chains(prior_array)

  # Also subset posterior to only the shared (transformed) parameter names
  shared_pars <- intersect(post_pars, colnames(prior_mat))
  post_mat <- post_mat[, shared_pars, drop = FALSE]
  prior_mat <- prior_mat[, shared_pars, drop = FALSE]

  # Melt to long format
  melt_draws <- function(mat, source_label) {
    df <- reshape2::melt(
      mat,
      varnames = c("Draw", "Parameter"),
      value.name = "Value",
      as.is = FALSE
    )
    df$Source <- source_label
    df
  }

  post_long <- melt_draws(post_mat, "Posterior")
  prior_long <- melt_draws(prior_mat, "Prior")

  combined <- rbind(prior_long, post_long)
  combined$Parameter <- factor(combined$Parameter, levels = shared_pars)
  combined$Source <- factor(combined$Source, levels = c("Prior", "Posterior"))

  dplyr::as_tibble(combined)
}


# =============================================================================
# SECTION 3: PROTOTYPE 1 — Density overlay
# =============================================================================
# Prior and posterior KDE curves overlaid in same panel, faceted by parameter.
# This is the most intuitive and commonly requested approach.

mcmc_prior_posterior_dens <- function(
    posterior,
    prior,
    pars = character(),
    regex_pars = character(),
    transformations = list(),
    ...,
    facet_args = list(),
    trim = FALSE,
    bw = NULL,
    adjust = NULL,
    kernel = NULL,
    n_dens = NULL,
    bounds = NULL,
    alpha = 0.4
) {
  data <- mcmc_prior_posterior_data(
    posterior, prior, pars, regex_pars, transformations
  )
  n_param <- length(unique(data$Parameter))

  bw <- bw %||% "nrd0"
  adjust <- adjust %||% 1
  kernel <- kernel %||% "gaussian"
  n_dens <- n_dens %||% 1024

  geom_args <- list(
    linewidth = 0.7,
    na.rm = TRUE,
    trim = trim,
    bw = bw,
    adjust = adjust,
    kernel = kernel,
    n = n_dens
  )
  if (!is.null(bounds)) geom_args[["bounds"]] <- bounds

  graph <- ggplot(data, aes(
    x = .data$Value,
    fill = .data$Source,
    color = .data$Source
  )) +
    do.call(stat_density, c(
      list(mapping = aes(y = after_stat(density))),
      geom_args,
      list(geom = "area", position = "identity", alpha = alpha)
    )) +
    scale_fill_manual(
      name = "",
      values = c(
        "Prior" = bp_color("l"),
        "Posterior" = bp_color("m")
      )
    ) +
    scale_color_manual(
      name = "",
      values = c(
        "Prior" = bp_color("lh"),
        "Posterior" = bp_color("mh")
      )
    )

  facet_args[["scales"]] <- facet_args[["scales"]] %||% "free"
  if (n_param > 1) {
    facet_args[["facets"]] <- vars(.data$Parameter)
    graph <- graph + do.call("facet_wrap", facet_args)
  } else {
    graph <- graph + xlab(levels(data$Parameter)[1])
  }

  graph +
    bayesplot_theme_get() +
    yaxis_text(FALSE) +
    yaxis_ticks(FALSE) +
    yaxis_title(FALSE) +
    xaxis_title(on = n_param == 1) +
    legend_move("top")
}


# =============================================================================
# SECTION 4: PROTOTYPE 2 — Areas (ridgeline) with prior as outline
# =============================================================================
# Extends the mcmc_areas() style: posterior as filled area, prior as
# dashed outline curve. Compact for many parameters.

mcmc_prior_posterior_areas <- function(
    posterior,
    prior,
    pars = character(),
    regex_pars = character(),
    transformations = list(),
    ...,
    prob = 0.5,
    prob_outer = 1,
    point_est = c("median", "mean", "none"),
    bw = NULL,
    adjust = NULL,
    kernel = NULL,
    n_dens = NULL,
    bounds = NULL
) {
  point_est <- match.arg(point_est)
  data <- mcmc_prior_posterior_data(
    posterior, prior, pars, regex_pars, transformations
  )

  params <- levels(data$Parameter)
  n_param <- length(params)

  bw_val <- bw %||% "nrd0"
  adj_val <- adjust %||% 1
  kern_val <- kernel %||% "gaussian"
  n_dens_val <- n_dens %||% 1024

  compute_dens <- function(values, interval_width = 1) {
    tail_w <- (1 - interval_width) / 2
    qs <- quantile(values, probs = c(tail_w, 1 - tail_w))
    support <- range(qs)
    args <- list(x = values, from = support[1], to = support[2], n = n_dens_val)
    if (is.character(bw_val)) args$bw <- bw_val
    if (!is.null(adj_val)) args$adjust <- adj_val
    if (!is.null(kern_val)) args$kernel <- kern_val
    d <- do.call(stats::density, args)
    data.frame(x = d$x, density = d$y)
  }

  # Build density data for each parameter x source
  build_density_data <- function(src, iw) {
    sub <- data[data$Source == src, ]
    do.call(rbind, lapply(params, function(p) {
      vals <- sub$Value[sub$Parameter == p]
      d <- compute_dens(vals, interval_width = iw)
      d$parameter <- p
      d$source <- src
      d
    }))
  }

  post_outer <- build_density_data("Posterior", prob_outer)
  post_inner <- build_density_data("Posterior", prob)
  prior_dens <- build_density_data("Prior", 1)

  post_outer$parameter <- factor(post_outer$parameter, levels = params)
  post_inner$parameter <- factor(post_inner$parameter, levels = params)
  prior_dens$parameter <- factor(prior_dens$parameter, levels = params)

  # Point estimates for posterior
  if (point_est != "none") {
    est_fn <- if (point_est == "median") stats::median else base::mean
    post_sub <- data[data$Source == "Posterior", ]
    point_data <- do.call(rbind, lapply(params, function(p) {
      vals <- post_sub$Value[post_sub$Parameter == p]
      pt <- est_fn(vals)
      # Get density at point estimate
      d <- compute_dens(vals, interval_width = 1)
      dens_at_pt <- approx(d$x, d$density, xout = pt)$y
      data.frame(parameter = p, x = pt, density = dens_at_pt %||% 0)
    }))
    point_data$parameter <- factor(point_data$parameter, levels = params)
  }

  # Normalize densities per parameter for consistent ridgeline heights
  for (p in params) {
    max_d <- max(c(
      post_outer$density[post_outer$parameter == p],
      prior_dens$density[prior_dens$parameter == p]
    ), na.rm = TRUE)
    post_outer$density[post_outer$parameter == p] <-
      post_outer$density[post_outer$parameter == p] / max_d
    post_inner$density[post_inner$parameter == p] <-
      post_inner$density[post_inner$parameter == p] / max_d
    prior_dens$density[prior_dens$parameter == p] <-
      prior_dens$density[prior_dens$parameter == p] / max_d
    if (point_est != "none") {
      point_data$density[point_data$parameter == p] <-
        point_data$density[point_data$parameter == p] / max_d
    }
  }

  # Build plot
  graph <- ggplot(post_outer, aes(x = .data$x, y = .data$parameter)) +
    # Posterior filled area (outer)
    ggridges::geom_ridgeline(
      aes(height = .data$density, scale = 0.9),
      fill = NA,
      color = bp_color("dark"),
      linewidth = 0.4
    ) +
    # Posterior filled area (inner)
    ggridges::geom_ridgeline(
      data = post_inner,
      aes(height = .data$density, scale = 0.9),
      fill = bp_color("light"),
      color = bp_color("dark"),
      linewidth = 0.4
    ) +
    # Prior as dashed outline
    ggridges::geom_ridgeline(
      data = prior_dens,
      aes(height = .data$density, scale = 0.9),
      fill = NA,
      color = bp_color("dark_highlight"),
      linewidth = 0.7,
      linetype = "dashed"
    )

  # Add point estimate
  if (point_est != "none") {
    graph <- graph +
      ggridges::geom_ridgeline(
        data = point_data,
        aes(height = .data$density, scale = 0.9),
        color = NA,
        fill = bp_color("mid_highlight"),
        linewidth = 0.4
      )
  }

  graph +
    scale_y_discrete(
      limits = rev(params),
      expand = expansion(add = c(0.3, 0.8))
    ) +
    bayesplot_theme_get() +
    yaxis_text(face = "bold") +
    yaxis_title(FALSE) +
    yaxis_ticks(linewidth = 1) +
    xaxis_title(FALSE) +
    labs(subtitle = "Solid = posterior, dashed = prior")
}


# =============================================================================
# SECTION 5: PROTOTYPE 3 — Interval comparison
# =============================================================================
# Point + interval for both prior and posterior, side by side.
# Inspired by rstanarm::posterior_vs_prior() but as a bayesplot function.

mcmc_prior_posterior_intervals <- function(
    posterior,
    prior,
    pars = character(),
    regex_pars = character(),
    transformations = list(),
    ...,
    prob = 0.5,
    prob_outer = 0.9,
    point_est = c("median", "mean", "none"),
    color_by = c("vs", "parameter")
) {
  point_est <- match.arg(point_est)
  color_by <- match.arg(color_by)
  data <- mcmc_prior_posterior_data(
    posterior, prior, pars, regex_pars, transformations
  )

  params <- levels(data$Parameter)
  n_param <- length(params)

  # Compute intervals per parameter per source
  alpha_inner <- (1 - prob) / 2
  alpha_outer <- (1 - prob_outer) / 2
  probs <- sort(c(alpha_outer, alpha_inner, 0.5, 1 - alpha_inner, 1 - alpha_outer))

  compute_intervals <- function(vals, est) {
    qs <- quantile(vals, probs = probs)
    pt <- switch(est,
      median = median(vals),
      mean = mean(vals),
      none = NA_real_
    )
    data.frame(
      ll = qs[1], l = qs[2], m = qs[3], u = qs[4], uu = qs[5],
      point = pt
    )
  }

  intervals <- do.call(rbind, lapply(c("Prior", "Posterior"), function(src) {
    sub <- data[data$Source == src, ]
    do.call(rbind, lapply(params, function(p) {
      vals <- sub$Value[sub$Parameter == p]
      df <- compute_intervals(vals, point_est)
      df$Parameter <- p
      df$Source <- src
      df
    }))
  }))

  intervals$Parameter <- factor(intervals$Parameter, levels = params)
  intervals$Source <- factor(intervals$Source, levels = c("Prior", "Posterior"))

  # Dodge positioning
  pd <- position_dodge(width = 0.6)

  if (color_by == "vs") {
    color_aes <- aes(color = .data$Source, group = .data$Source)
    color_scale <- scale_color_manual(
      name = "",
      values = c(
        "Prior" = bp_color("light_highlight"),
        "Posterior" = bp_color("dark")
      )
    )
  } else {
    color_aes <- aes(color = .data$Parameter, group = .data$Source)
    color_scale <- NULL
  }

  graph <- ggplot(intervals, aes(y = .data$Parameter)) +
    color_aes +
    # Outer interval
    geom_segment(
      aes(x = .data$ll, xend = .data$uu, yend = .data$Parameter),
      position = pd,
      linewidth = 0.5,
      lineend = "round"
    ) +
    # Inner interval
    geom_segment(
      aes(x = .data$l, xend = .data$u, yend = .data$Parameter),
      position = pd,
      linewidth = 2,
      lineend = "round"
    )

  if (point_est != "none") {
    graph <- graph +
      geom_point(
        aes(x = .data$point, shape = .data$Source),
        position = pd,
        size = 3,
        fill = "white"
      ) +
      scale_shape_manual(
        name = "",
        values = c("Prior" = 22, "Posterior" = 21)
      )
  }

  if (!is.null(color_scale)) graph <- graph + color_scale

  x_range <- range(c(intervals$ll, intervals$uu))
  if (x_range[1] < 0 && x_range[2] > 0) {
    graph <- graph + geom_vline(xintercept = 0, color = "gray80", linewidth = 0.3)
  }

  caption <- sprintf("Showing %g%% and %g%% intervals",
                     round(prob * 100, 1), round(prob_outer * 100, 1))

  graph +
    scale_y_discrete(limits = rev(params)) +
    bayesplot_theme_get() +
    yaxis_text(face = "bold") +
    yaxis_title(FALSE) +
    yaxis_ticks(linewidth = 1) +
    xaxis_title(FALSE) +
    legend_move("top") +
    labs(subtitle = caption)
}


# =============================================================================
# SECTION 6: PROTOTYPE 4 — Faceted density with histogram (posterior)
#                           + line (prior)
# =============================================================================
# Posterior shown as histogram, prior shown as line. This visually
# distinguishes the two distributions clearly.

mcmc_prior_posterior_hist <- function(
    posterior,
    prior,
    pars = character(),
    regex_pars = character(),
    transformations = list(),
    ...,
    facet_args = list(),
    binwidth = NULL,
    bins = NULL,
    freq = FALSE,
    bw = NULL,
    adjust = NULL,
    kernel = NULL,
    n_dens = NULL,
    bounds = NULL
) {
  data <- mcmc_prior_posterior_data(
    posterior, prior, pars, regex_pars, transformations
  )
  n_param <- length(unique(data$Parameter))

  post_data <- data[data$Source == "Posterior", ]
  prior_data <- data[data$Source == "Prior", ]

  graph <- ggplot(post_data, aes(x = .data$Value)) +
    geom_histogram(
      aes(y = after_stat(density)),
      fill = bp_color("mid"),
      color = bp_color("mid_highlight"),
      linewidth = 0.25,
      na.rm = TRUE,
      binwidth = binwidth,
      bins = bins,
      alpha = 0.8
    ) +
    stat_density(
      data = prior_data,
      aes(x = .data$Value),
      geom = "line",
      color = bp_color("dark_highlight"),
      linewidth = 1,
      linetype = "dashed",
      trim = FALSE,
      bw = bw %||% "nrd0",
      adjust = adjust %||% 1,
      kernel = kernel %||% "gaussian",
      n = n_dens %||% 1024,
      na.rm = TRUE
    )

  facet_args[["scales"]] <- facet_args[["scales"]] %||% "free"
  if (n_param > 1) {
    facet_args[["facets"]] <- vars(.data$Parameter)
    graph <- graph + do.call("facet_wrap", facet_args)
  } else {
    graph <- graph + xlab(levels(data$Parameter)[1])
  }

  graph +
    bayesplot_theme_get() +
    yaxis_text(FALSE) +
    yaxis_ticks(FALSE) +
    yaxis_title(FALSE) +
    xaxis_title(on = n_param == 1) +
    labs(subtitle = "Bars = posterior, dashed line = prior")
}


# =============================================================================
# SECTION 7: PROTOTYPE 5 — Paired density (triptych-inspired, simplified)
# =============================================================================
# Two-row layout: prior density on top, posterior density on bottom,
# with shared x-axis per parameter. Handles scale mismatch well.

mcmc_prior_posterior_paired <- function(
    posterior,
    prior,
    pars = character(),
    regex_pars = character(),
    transformations = list(),
    ...,
    bw = NULL,
    adjust = NULL,
    kernel = NULL,
    n_dens = NULL,
    bounds = NULL
) {
  data <- mcmc_prior_posterior_data(
    posterior, prior, pars, regex_pars, transformations
  )

  bw <- bw %||% "nrd0"
  adjust <- adjust %||% 1
  kernel <- kernel %||% "gaussian"
  n_dens <- n_dens %||% 1024

  graph <- ggplot(data, aes(x = .data$Value, fill = .data$Source)) +
    stat_density(
      aes(y = after_stat(density)),
      geom = "area",
      position = "identity",
      alpha = 0.7,
      linewidth = 0.5,
      color = bp_color("dark"),
      trim = FALSE,
      bw = bw,
      adjust = adjust,
      kernel = kernel,
      n = n_dens,
      na.rm = TRUE
    ) +
    scale_fill_manual(
      name = "",
      values = c(
        "Prior" = bp_color("light"),
        "Posterior" = bp_color("mid")
      )
    ) +
    facet_grid(
      rows = vars(.data$Source),
      cols = vars(.data$Parameter),
      scales = "free"
    )

  graph +
    bayesplot_theme_get() +
    yaxis_text(FALSE) +
    yaxis_ticks(FALSE) +
    yaxis_title(FALSE) +
    xaxis_title(FALSE) +
    legend_none()
}


# =============================================================================
# SECTION 8: Run all prototypes
# =============================================================================

run_all_prototypes <- function() {
  cat("Generating test data...\n")
  test_data <- generate_test_data()

  # ---- Scenario A: Simple regression ----
  cat("\n=== Scenario A: Simple Regression ===\n")
  d <- test_data$simple

  cat("  Prototype 1: Density overlay\n")
  p1 <- mcmc_prior_posterior_dens(d$posterior, d$prior)
  print(p1 + ggtitle("P1: Density overlay — Simple regression"))

  cat("  Prototype 2: Areas with prior outline\n")
  p2 <- mcmc_prior_posterior_areas(d$posterior, d$prior)
  print(p2 + ggtitle("P2: Areas + prior outline — Simple regression"))

  cat("  Prototype 3: Interval comparison\n")
  p3 <- mcmc_prior_posterior_intervals(d$posterior, d$prior)
  print(p3 + ggtitle("P3: Interval comparison — Simple regression"))

  cat("  Prototype 4: Histogram + prior line\n")
  p4 <- mcmc_prior_posterior_hist(d$posterior, d$prior)
  print(p4 + ggtitle("P4: Histogram + prior line — Simple regression"))

  cat("  Prototype 5: Paired density\n")
  p5 <- mcmc_prior_posterior_paired(d$posterior, d$prior)
  print(p5 + ggtitle("P5: Paired density — Simple regression"))

  # ---- Scenario B: Hierarchical model ----
  cat("\n=== Scenario B: Hierarchical Model ===\n")
  d <- test_data$hierarchical

  p1b <- mcmc_prior_posterior_dens(d$posterior, d$prior)
  print(p1b + ggtitle("P1: Density overlay — Hierarchical"))

  p2b <- mcmc_prior_posterior_areas(d$posterior, d$prior)
  print(p2b + ggtitle("P2: Areas — Hierarchical"))

  p3b <- mcmc_prior_posterior_intervals(d$posterior, d$prior)
  print(p3b + ggtitle("P3: Intervals — Hierarchical"))

  # ---- Scenario C: Wide prior (dramatic update) ----
  cat("\n=== Scenario C: Wide Prior ===\n")
  d <- test_data$wide_prior

  p1c <- mcmc_prior_posterior_dens(d$posterior, d$prior)
  print(p1c + ggtitle("P1: Density overlay — Wide prior (scale mismatch!)"))

  # This is where the paired approach shines
  p5c <- mcmc_prior_posterior_paired(d$posterior, d$prior)
  print(p5c + ggtitle("P5: Paired density — Wide prior (handles mismatch)"))

  # ---- Parameter selection demo ----
  cat("\n=== Parameter Selection ===\n")
  d <- test_data$hierarchical

  p_sel <- mcmc_prior_posterior_dens(
    d$posterior, d$prior,
    regex_pars = "theta"
  )
  print(p_sel + ggtitle("Parameter selection: regex_pars = 'theta'"))

  cat("\nAll prototypes rendered.\n")
  invisible(list(
    simple = list(dens = p1, areas = p2, intervals = p3, hist = p4, paired = p5),
    hierarchical = list(dens = p1b, areas = p2b, intervals = p3b),
    wide_prior = list(dens = p1c, paired = p5c)
  ))
}


# =============================================================================
# SECTION 9: Utility — summary of prior-posterior overlap
# =============================================================================
# Compute overlap coefficient between prior and posterior densities.
# This is a numeric summary that complements the plots.

prior_posterior_overlap <- function(
    posterior,
    prior,
    pars = character(),
    regex_pars = character(),
    transformations = list(),
    n_dens = 1024
) {
  data <- mcmc_prior_posterior_data(
    posterior, prior, pars, regex_pars, transformations
  )

  params <- levels(data$Parameter)

  overlaps <- vapply(params, function(p) {
    post_vals <- data$Value[data$Parameter == p & data$Source == "Posterior"]
    prior_vals <- data$Value[data$Parameter == p & data$Source == "Prior"]

    # Common support
    rng <- range(c(post_vals, prior_vals))
    xs <- seq(rng[1], rng[2], length.out = n_dens)

    d_post <- density(post_vals, from = rng[1], to = rng[2], n = n_dens)
    d_prior <- density(prior_vals, from = rng[1], to = rng[2], n = n_dens)

    # Overlap = integral of min(f_prior, f_post)
    mins <- pmin(d_post$y, d_prior$y)
    dx <- diff(d_post$x[1:2])
    sum(mins) * dx
  }, numeric(1))

  data.frame(
    Parameter = params,
    Overlap = round(overlaps, 4),
    stringsAsFactors = FALSE
  )
}


# =============================================================================
# RUN
# =============================================================================

if (interactive()) {
  cat("Running prior vs posterior prototypes...\n\n")
  results <- run_all_prototypes()

  cat("\n--- Prior-Posterior Overlap (Simple Regression) ---\n")
  test_data <- generate_test_data()
  d <- test_data$simple
  print(prior_posterior_overlap(d$posterior, d$prior))

  cat("\n--- Prior-Posterior Overlap (Hierarchical) ---\n")
  d <- test_data$hierarchical
  print(prior_posterior_overlap(d$posterior, d$prior))
}
