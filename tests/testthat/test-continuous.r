context("Continuous dependent variables")

data(penguins)

penguins <- na.omit(penguins)

mc1 <- dglm(penguins, bill_length_mm ~ .)
expect_true(inherits(mc1, "continuous_deepglm"))

mcl <- lm(bill_length_mm ~ ., penguins)
expect_true(
  (abs(sd(predict(mc1, penguins) - penguins$bill_length_mm) - 
       sd(predict(mcl, penguins) - penguins$bill_length_mm))) < 3)

mc2 <- dglm(penguins, bill_length_mm ~ ., hidden_layers = 10)
expect_true(
  abs(sd(predict(mc2, penguins) - penguins$bill_length_mm) -
      sd(predict(mc1, penguins) - penguins$bill_length_mm)) < 3)

#expect_equal(predict(fit_dl1, iris), predict(fit_linear, iris))
