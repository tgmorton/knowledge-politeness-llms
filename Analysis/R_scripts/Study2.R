library(dplyr)
library(tidyr)
library(ggplot2)
library(stats)


# load updated_file.csv
df.data <- read.csv("updated_file.csv")

# Process the data with complete combinations and SD
processed_data <- df.data %>%
  # First create all possible combinations
  ungroup() %>%
  complete(
    State = unique(State),
    Goal_simplified = unique(Goal_simplified),
    Assessment = c("terrible", "bad", "good", "amazing"),
    Was.Wasn.t = c("was", "wasn't"),
    fill = list(n = 0)
  ) %>%
  # Get total counts for each condition
  group_by(State, Goal_simplified) %>%
  mutate(total_responses = n()) %>%
  # Now calculate proportions and SD within each group
  group_by(State, Goal_simplified, Assessment, Was.Wasn.t) %>%
  summarise(
    count = n(),
    proportion = count / first(total_responses),
    # Calculate SD for binary data (bernoulli distribution)
    sd = sqrt(proportion * (1 - proportion)),
    .groups = 'drop'
  ) %>%
  # Ensure Assessment is properly ordered
  mutate(
    Assessment = factor(Assessment,
                        levels = c("terrible", "bad", "good", "amazing")),
    # Make sure Was.Wasn.t matches your graph labels
    Was.Wasn.t = case_when(
      Was.Wasn.t == "was" ~ "It was ~",
      Was.Wasn.t == "wasn't" ~ "It wasn't ~",
      TRUE ~ Was.Wasn.t
    )
  )

# order Goal_simplified, informative, social, both
processed_data$Goal_simplified <- factor(processed_data$Goal_simplified, levels = c("informational", "social", "both"))

# Create the visualization
ggplot(processed_data,
       aes(x = Assessment, y = proportion,
           color = Was.Wasn.t, group = Was.Wasn.t)) +
  # Add lines and points
  geom_line() +
  geom_point() +
  # Add error bars using SD
  geom_errorbar(
    aes(ymin = pmax(0, proportion - sd),
        ymax = pmin(1, proportion + sd)),
    width = 0.2
  ) +
  # Add a reference line at y = 0.1
  geom_hline(yintercept = 0.1, linetype = "dashed", color = "black") +
  # Facet by State and Goal_simplified
  facet_grid(Goal_simplified ~ State) +
  # Customize colors
  scale_color_manual(values = c("It was ~" = "blue", "It wasn't ~" = "red")) +
  # Set axis limits and labels
  scale_y_continuous(limits = c(0, 1),
                     breaks = seq(0, 1, 0.25)) +
  # Customize theme elements
  theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.background = element_rect(fill = "white"),
    legend.position = "bottom",
    legend.title = element_blank()
  ) +
  # Labels
  labs(x = "utterance",
       y = "proportion chosen")

# If you want to save the plot
ggsave("response_visualization.pdf", width = 12, height = 8)

# For 0 hearts, both goal, "wasn't amazing"
data_0hearts <- df.data %>%
  filter(State == "0 hearts",
         Goal_simplified == "both")

successes_0hearts <- sum(data_0hearts$Was.Wasn.t == "wasn't" &
                           data_0hearts$Assessment == "amazing")
total_0hearts <- nrow(data_0hearts)

binom_test_0hearts <- binom.test(successes_0hearts, total_0hearts,
                                 p = 0.125,
                                 alternative = "greater")

# For 1 heart, both goal, "amazing"
data_1heart <- df.data %>%
  filter(State == "1 heart",
         Goal_simplified == "both")

successes_1heart <- sum(data_1heart$Assessment == "amazing")
total_1heart <- nrow(data_1heart)

binom_test_1heart <- binom.test(successes_1heart, total_1heart,
                                p = 0.125,
                                alternative = "greater")

# Print results
cat("Test 1: 0 hearts, both goal, 'wasn't amazing' vs chance (0.125)\n")
cat("Observed proportion:", successes_0hearts/total_0hearts, "\n")
cat("Count:", successes_0hearts, "out of", total_0hearts, "\n")
cat("p-value:", binom_test_0hearts$p.value, "\n\n")

cat("Test 2: 1 heart, both goal, 'amazing' vs chance (0.125)\n")
cat("Observed proportion:", successes_1heart/total_1heart, "\n")
cat("Count:", successes_1heart, "out of", total_1heart, "\n")
cat("p-value:", binom_test_1heart$p.value, "\n")