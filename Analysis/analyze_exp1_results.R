#!/usr/bin/env Rscript
#
# Grace Project - Experiment 1 Analysis
#
# Analyzes raw text responses from Study 1 and Study 2 Experiment 1
# across 4 models: Gemma-2B, Gemma-9B, Llama-3B, Llama-8B

library(tidyverse)
library(here)

# ============================================================================
# Load Data
# ============================================================================

cat("Loading Experiment 1 results...\n")

study1 <- read_csv(here("outputs/analysis/study1_exp1_all_models.csv"),
                   show_col_types = FALSE)

study2 <- read_csv(here("outputs/analysis/study2_exp1_all_models.csv"),
                   show_col_types = FALSE)

cat(sprintf("Study 1: %d trials from %d models\n",
            nrow(study1),
            length(unique(study1$model_name))))

cat(sprintf("Study 2: %d trials from %d models\n",
            nrow(study2),
            length(unique(study2$model_name))))

# ============================================================================
# Study 1: Knowledge Attribution Analysis
# ============================================================================

cat("\n=== Study 1: Knowledge Attribution ===\n")

# Response distribution by model
study1_summary <- study1 %>%
  group_by(model_name, response) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(model_name) %>%
  mutate(
    total = sum(count),
    percent = count / total * 100
  ) %>%
  arrange(model_name, response)

cat("\nResponse distribution by model:\n")
print(study1_summary)

# Compare access/observe conditions
study1_by_condition <- study1 %>%
  mutate(
    response_numeric = as.numeric(response),
    access_observe = paste0("access=", access, ", observe=", observe)
  ) %>%
  group_by(model_name, access_observe) %>%
  summarise(
    mean_response = mean(response_numeric, na.rm = TRUE),
    sd_response = sd(response_numeric, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

cat("\nMean responses by access/observe condition:\n")
print(study1_by_condition)

# Plot: Response distribution by model
p1 <- ggplot(study1, aes(x = response, fill = model_name)) +
  geom_bar(position = "dodge") +
  facet_wrap(~model_name, ncol = 2) +
  labs(
    title = "Study 1: Response Distribution by Model",
    subtitle = sprintf("%d trials per model", nrow(study1) / length(unique(study1$model_name))),
    x = "Response (# of passing exams)",
    y = "Count"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(here("outputs/analysis/study1_response_distribution.png"),
       p1, width = 10, height = 8)

cat("✅ Saved: outputs/analysis/study1_response_distribution.png\n")

# ============================================================================
# Study 2: Politeness Judgments Analysis
# ============================================================================

cat("\n=== Study 2: Politeness Judgments ===\n")

# Response length distribution
study2 <- study2 %>%
  mutate(response_length = nchar(response))

study2_length_summary <- study2 %>%
  group_by(model_name) %>%
  summarise(
    mean_length = mean(response_length),
    sd_length = sd(response_length),
    min_length = min(response_length),
    max_length = max(response_length),
    .groups = "drop"
  )

cat("\nResponse length by model:\n")
print(study2_length_summary)

# Responses by goal condition
study2_by_goal <- study2 %>%
  group_by(model_name, Goal) %>%
  summarise(
    n = n(),
    mean_length = mean(response_length),
    .groups = "drop"
  )

cat("\nResponse count by goal condition:\n")
print(study2_by_goal)

# Plot: Response length distribution
p2 <- ggplot(study2, aes(x = response_length, fill = model_name)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~model_name, ncol = 2) +
  labs(
    title = "Study 2: Response Length Distribution by Model",
    subtitle = sprintf("%d trials per model", nrow(study2) / length(unique(study2$model_name))),
    x = "Response Length (characters)",
    y = "Density"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(here("outputs/analysis/study2_response_length.png"),
       p2, width = 10, height = 8)

cat("✅ Saved: outputs/analysis/study2_response_length.png\n")

# ============================================================================
# Cross-Model Comparison
# ============================================================================

cat("\n=== Cross-Model Comparison ===\n")

# Extract model family and size
add_model_info <- function(df) {
  df %>%
    mutate(
      model_family = case_when(
        str_detect(model_name, "gemma") ~ "Gemma",
        str_detect(model_name, "llama") ~ "Llama",
        TRUE ~ "Other"
      ),
      model_size = case_when(
        str_detect(model_name, "2b") ~ "2B",
        str_detect(model_name, "3b") ~ "3B",
        str_detect(model_name, "8b") ~ "8B",
        str_detect(model_name, "9b") ~ "9B",
        TRUE ~ "Unknown"
      ),
      model_size_numeric = case_when(
        model_size == "2B" ~ 2,
        model_size == "3B" ~ 3,
        model_size == "8B" ~ 8,
        model_size == "9B" ~ 9,
        TRUE ~ NA_real_
      )
    )
}

study1_with_info <- add_model_info(study1)
study2_with_info <- add_model_info(study2)

# Compare Gemma vs Llama (matched sizes)
cat("\nGemma vs Llama comparison (matched sizes):\n")
cat("  Gemma-2B vs Llama-3B (small)\n")
cat("  Gemma-9B vs Llama-8B (medium)\n")

# ============================================================================
# Summary
# ============================================================================

cat("\n" , rep("=", 70), "\n", sep = "")
cat("✅ Analysis Complete!\n")
cat(rep("=", 70), "\n", sep = "")
cat("\nGenerated files:\n")
cat("  - outputs/analysis/study1_response_distribution.png\n")
cat("  - outputs/analysis/study2_response_length.png\n")
cat("\nNext steps:\n")
cat("  1. Review visualizations\n")
cat("  2. Run more detailed statistical tests\n")
cat("  3. Compare with Experiment 2 (probability distributions) when available\n")
cat("\n")
