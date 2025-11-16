library(dplyr)
library(tidyr)
library(papaja)
library(rstatix)
library(ggplot2)

# load study1_annotated.csv
df.data <- read.csv("study1_annotated.csv")

# drop trial if drop_incomplete is not NA or not ""
df.data <- df.data %>% filter(is.na(drop_incomplete) | drop_incomplete == "")

# drop trial if drop_wrongPrior is not NA or not ""
df.data <- df.data %>% filter(is.na(drop_wrongPrior) | drop_wrongPrior == "")

# drop trial if drop_wrongKnowledge is not NA or not ""
df.data <- df.data %>% filter(is.na(drop_wrongKnowledge) | drop_wrongKnowledge == "")

# integer encode X0, X1, X2, and X3
df.data$X0 <- as.integer(df.data$X0)
df.data$X1 <- as.integer(df.data$X1)
df.data$X2 <- as.integer(df.data$X2)
df.data$X3 <- as.integer(df.data$X3)

# remove rows with NA in X0, X1, X2, and X3
df.data <- df.data %>% filter(!is.na(X0) & !is.na(X1) & !is.na(X2) & !is.na(X3))

# summarize X0, X1, X2, and X3 as 0, 1, 2, 3 mean by access and observe including se
reports.X.means <- df.data %>% group_by(access, observe) %>%
  summarize(X0.mean = mean(X0, na.rm = TRUE), X0.se = sd(X0, na.rm = TRUE) / sqrt(n()),
            X1.mean = mean(X1, na.rm = TRUE), X1.se = sd(X1, na.rm = TRUE) / sqrt(n()),
            X2.mean = mean(X2, na.rm = TRUE), X2.se = sd(X2, na.rm = TRUE) / sqrt(n()),
            X3.mean = mean(X3, na.rm = TRUE), X3.se = sd(X3, na.rm = TRUE) / sqrt(n()),
            n = n())

# make df.data long by X0, X1, X2, and X3
df.data.long <- df.data %>% gather(key = "X", value = "value", X0, X1, X2, X3)

# First, your data should be in long format with columns:
# X (0-3), value (y-axis values), se (standard error), observe ("One"/"Two"/"Three"), and access (1/2/3)

# Remove 'X' from column 'X' values
df.data.long$X <- as.integer(gsub("X", "", df.data.long$X))


# First calculate summary statistics
df_summary <- df.data.long %>%
  group_by(observe, access, X) %>%
  summarise(
    mean_value = mean(value, na.rm = TRUE),
    se = sd(value, na.rm = TRUE) / sqrt(n()),
    sd = sd(value, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  # Ensure proper factor ordering
  mutate(
    observe = factor(observe, levels = c("3", "2", "1")),
    access = factor(access),
    X = factor(X)
  )

ggplot(df_summary, aes(x = X, y = mean_value)) +
  # Add horizontal lines first (so they appear behind the bars)
  geom_hline(yintercept = seq(0, 100, 25),
             linetype = "dashed",
             color = "grey90",
             size = 0.3) +
  geom_hline(yintercept = 25,
             linetype = "dashed",
             color = "grey60",  # Darker line at 25
             size = 0.3) +
  geom_bar(stat = "identity",
           fill = "grey90",
           color = "black",
           width = 0.7,
           size = 0.1) +
  geom_errorbar(aes(ymin = mean_value - sd, ymax = mean_value + sd),
                width = 0.2) +
  facet_grid(observe ~ access,
             labeller = labeller(
               observe = function(x) paste0('"', x, '"'),
               access = function(x) paste("access", x)),
             drop = TRUE) +
  labs(x = "Word",
       y = "% chosen") +
  scale_y_continuous(sec.axis = dup_axis()) +
  coord_cartesian(ylim = c(0, 100)) +
  theme_bw() +
  theme(
    panel.grid = element_blank(),
    strip.background = element_rect(fill = "white", color = "black", size = 0.1),
    strip.text = element_text(size = 10),
    axis.text.y = element_text(),
    axis.title = element_text(),
    axis.title.x = element_text(size = 10),
    axis.title.y = element_text(size = 10),
    axis.text.y.right = element_blank(),
    axis.title.y.right = element_blank(),
    axis.ticks.y.right = element_line(),
    panel.border = element_rect(color = "black", fill = NA, size = 0.1)
  )

# Check data structure
str(df.data.long)
# View unique combinations
table(df.data.long$access, df.data.long$observe)

# Helper function for paired t-tests
run_paired_test <- function(data, access_level, observe_level, x1, x2) {
  data %>%
    filter(access == access_level, observe == observe_level) %>%
    select(participant_id, X, value) %>%
    spread(X, value) %>%
    with(t.test(get(x1) - get(x2), alternative = "greater"))
}

# 1. ANOVA on bets for state 3
df_anova <- df.data.long %>%
  filter(X == 3)
anova_result <- aov(value ~ access * observe, data = df_anova)
summary(anova_result)

# 2. Planned Comparisons

# Complete Access (access == 3) comparisons
# When speaker says "two": bets on state 3 vs state 2
complete_two <- run_paired_test(df.data.long, "3", "2", "2", "3")

# When speaker says "one":
# Bets on state 1 vs state 3
complete_one_vs_three <- run_paired_test(df.data.long, "3", "1", "1", "3")
# Bets on state 1 vs state 2
complete_one_vs_two <- run_paired_test(df.data.long, "3", "1", "1", "2")

# Access 1 comparisons when speaker says "one"
# Note: Access 1 only has observations for observe="1"
access1_one_vs_two <- run_paired_test(df.data.long, "1", "1", "1", "2")
access1_one_vs_three <- run_paired_test(df.data.long, "1", "1", "1", "3")

# Access 2 comparisons
# When speaker says "two" (only exists for observe="1" and "2")
access2_two_vs_three <- run_paired_test(df.data.long, "2", "2", "2", "3")

# When speaker says "one"
access2_one_vs_three <- run_paired_test(df.data.long, "2", "1", "1", "3")
access2_one_vs_two <- run_paired_test(df.data.long, "2", "1", "1", "2")

library(kableExtra)
library(broom)  # For tidy model extraction

# Function to format F-test results
format_f_test <- function(df1, df2, f_value, p_value) {
  sprintf("F(%d,%d) = %.2f, p %s",
          df1, df2, f_value,
          ifelse(p_value < .001, "< .001",
                 sprintf("= %.3f", p_value)))
}

# Function to format t-test results
format_t_test <- function(df, t_value, p_value) {
  sprintf("t(%d) = %.2f, p %s",
          df, t_value,
          ifelse(p_value < .001, "< .001",
                 sprintf("= %.3f", p_value)))
}

# Extract ANOVA results
anova_results <- tidy(anova_result)
anova_table <- data.frame(
  Analysis = c("Main effect of access", "Main effect of observe", "Access × Observe interaction"),
  Paper = c("F(2,205) = 6.57, p < .01",
            "F(2,205) = 269.8, p < .001",
            "F(1,205) = 34.7, p < .001"),
  Our_Results = with(anova_results, c(
    format_f_test(1, 233, statistic[1], p.value[1]),
    format_f_test(1, 233, statistic[2], p.value[2]),
    format_f_test(1, 233, statistic[3], p.value[3])
  ))
)

# Extract t-test results
ttest_table <- data.frame(
  Condition = c("Complete Access (speaker: 'two')",
                "Complete Access (speaker: 'one' vs state 3)",
                "Complete Access (speaker: 'one' vs state 2)",
                "Access 1 (speaker: 'one' vs state 2)",
                "Access 1 (speaker: 'one' vs state 3)",
                "Access 2 (speaker: 'two' vs state 3)",
                "Access 2 (speaker: 'one' vs state 3)",
                "Access 2 (speaker: 'one' vs state 2)"),
  Paper = c("t(43) = 10.2, p < .001",
            "t(42) = 13.1, p < .001",
            "t(42) = 17.1, p < .001",
            "t(24) = 1.9, p = .96",
            "t(24) = 3.2, p = 1.0",
            "t(24) = 1.1, p = .87",
            "t(25) = 3.9, p < .001",
            "t(25) = 1.5, p = .92"),
  Our_Results = c(
    format_t_test(complete_two$parameter, complete_two$statistic, complete_two$p.value),
    format_t_test(complete_one_vs_three$parameter, complete_one_vs_three$statistic, complete_one_vs_three$p.value),
    format_t_test(complete_one_vs_two$parameter, complete_one_vs_two$statistic, complete_one_vs_two$p.value),
    format_t_test(access1_one_vs_two$parameter, access1_one_vs_two$statistic, access1_one_vs_two$p.value),
    format_t_test(access1_one_vs_three$parameter, access1_one_vs_three$statistic, access1_one_vs_three$p.value),
    format_t_test(access2_two_vs_three$parameter, access2_two_vs_three$statistic, access2_two_vs_three$p.value),
    format_t_test(access2_one_vs_three$parameter, access2_one_vs_three$statistic, access2_one_vs_three$p.value),
    format_t_test(access2_one_vs_two$parameter, access2_one_vs_two$statistic, access2_one_vs_two$p.value)
  )
)

# Print tables
cat("ANOVA Results:\n")
kable(anova_table, format = "pipe", align = c('l', 'l', 'l'))

cat("\nPlanned Comparisons:\n")
kable(ttest_table, format = "pipe", align = c('l', 'l', 'l'))