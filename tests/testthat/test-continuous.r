library(palmerpenguins)

data(penguins)

penguins <- na.omit(penguins)

mc1 <- penguins %>%
  formula_dataset(bill_length_mm ~ .) %>%
  dglm()

test_that("A continuous deepglm model has been created with dglm().", {
  expect_true(inherits(mc1, "continuous_deepglm"))
})

fmc1 <- fdglm(penguins, bill_length_mm ~ .)

test_that("A continuous deepglm model has been created with tdglm().", {
  expect_true(inherits(fmc1, "continuous_deepglm"))
})

mcl <- lm(bill_length_mm ~ ., penguins)

test_that(paste("Predictions for the continuous deepglm with no hidden", 
                "layers is close to lm()."), {
  expect_true(
    (abs(sd(predict(mc1, penguins) - penguins$bill_length_mm) - 
         sd(predict(mcl, penguins) - penguins$bill_length_mm))) < 3)
})

mc2 <- penguins %>%
  formula_dataset(bill_length_mm ~ .) %>%
  dglm(hidden_layers = 10)

test_that(paste("Predictions for the continuous deepglm with one hidden", 
                "layers is close to lm()."), {
  expect_true(
    abs(sd(predict(mc2, penguins) - penguins$bill_length_mm) -
        sd(predict(mc1, penguins) - penguins$bill_length_mm)) < 3)
})
#expect_equal(predict(fit_dl1, iris), predict(fit_linear, iris))
