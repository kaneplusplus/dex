context("Survival Model Testing")

library(survival)
data(lung)

lung$status <- lung$status - 1
lung$sex <- as.factor(lung$sex)
lung$ph.ecog <- as.factor(lung$ph.ecog)

lung <- na.omit(lung[,c("time", "status", "sex", "age", "meal.cal", "wt.loss",
                        "ph.ecog")])

#tf$config$run_functions_eagerly(TRUE)
dc_fit <- dcoxph(
  lung, 
  Surv(time, status) ~ sex + age + meal.cal + wt.loss + ph.ecog)

expect_true(inherits(dc_fit, "survival_deepglm"))

cph <- coxph(Surv(time, status) ~ sex + age + meal.cal + wt.loss + ph.ecog, 
             lung)

expect_true(
  sd(predict(dc_fit, lung) - predict(cph, lung, type = "expected") < 3)

