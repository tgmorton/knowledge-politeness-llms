# Analyses

## Study 1

### Data Preparation

Prior to analysis, we excluded trials based on three criteria: (1) incomplete responses (marked in `drop_incomplete`), (2) incorrect prior understanding (marked in `drop_wrongPrior`), and (3) incorrect knowledge assessment (marked in `drop_wrongKnowledge`). Additionally, we removed any trials with missing values in the probability distribution variables (X0, X1, X2, X3), which represent participants' probability judgments for states 0, 1, 2, and 3, respectively.

### Statistical Analyses

To examine how participants' probability judgments for state 3 varied as a function of access level and observation condition, we conducted a 3 (access: 1, 2, 3) × 3 (observe: "One", "Two", "Three") analysis of variance (ANOVA) on the probability values assigned to state 3. We then conducted planned paired-samples t-tests to compare probability distributions across specific conditions. All t-tests were one-tailed with an alternative hypothesis that the first state would receive higher probability than the second state.

For complete access conditions (access = 3), we compared: (1) state 2 vs. state 3 when the speaker said "two", (2) state 1 vs. state 3 when the speaker said "one", and (3) state 1 vs. state 2 when the speaker said "one". For access level 1 (which only had observations for observe = "one"), we compared: (1) state 1 vs. state 2, and (2) state 1 vs. state 3. For access level 2, we compared: (1) state 2 vs. state 3 when the speaker said "two" (for observe = "1" and "2"), (2) state 1 vs. state 3 when the speaker said "one", and (3) state 1 vs. state 2 when the speaker said "one".

## Study 2

### Statistical Analyses

We calculated the proportion of each utterance choice (terrible, bad, good, amazing) across all combinations of State (0, 1, 2, or 3 hearts), Goal (informational, social, or both), and positivity frame (was vs. wasn't). To test whether specific utterance choices exceeded chance levels, we conducted binomial tests comparing observed proportions to the null hypothesis of 0.125 (1/8, representing chance selection among 8 possible utterance options: 4 assessments × 2 positivity frames). We conducted two specific tests: (1) whether "wasn't amazing" was chosen more than chance for 0 hearts with both goals, and (2) whether "amazing" was chosen more than chance for 1 heart with both goals. Both tests used one-tailed alternatives with the hypothesis that observed proportions would exceed chance.

## Polite Speaker Analysis

### Speaker Production

We analyzed speaker production data to examine how utterance choices varied as a function of true state (0, 1, 2, or 3 hearts), communicative goal (informative, social, or both), and positivity frame (negation vs. no negation). For each combination of these factors, we calculated the proportion of each utterance choice and computed Bayesian confidence intervals using the `binom.bayes` function.

To model the relationship between positivity frame (negation vs. no negation) and our predictors, we fit a generalized linear mixed-effects model using the `glmer` function from the `lme4` package (Bates et al., 2015). The model included true state (numeric, 0-3), goal (informative, social, both), and their interaction as fixed effects, with random intercepts for participants. The outcome was binary (positivity frame: 0 = negation, 1 = no negation), so we used a binomial family. A fuller random effects structure (including random slopes) caused model convergence failures, so we retained the simpler structure with random intercepts only.

Additionally, we fit a Bayesian regression model using the `brms` package (Bürkner, 2017) with a Bernoulli family. This model included true state (scaled) and goal as predictors, with goal reference-coded to "both". The model included random effects for both participants and items, with random slopes for true state and goal.

### Literal Semantics

We analyzed participants' acceptability judgments for literal semantic interpretations across different states (0-3) and utterances (terrible, bad, good, amazing), separately for positive ("It was ~") and negative ("It wasn't ~") frames. For each combination of positivity frame, state, and utterance, we calculated mean acceptability judgments and bootstrapped confidence intervals using the `multi_boot_standard` function from the `langcog` package.

