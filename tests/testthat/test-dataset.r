data(iris)

mm <- model.matrix(Sepal.Length ~ . - 1, iris)

mmd <- model_matrix_dataset(
  torch_tensor(mm),
  torch_tensor(matrix(data = iris$Sepal.Length, ncol = 1)),
  y_name = "Sepal.Length",
  x_mm_name = colnames(mm))

test_that("Get item is a list.", {
  expect_type(mmd$.getitem(1), "list")
})

dss <- mmd$.getitem(1:10)

test_that("Get item retrieves data correctly.", {
  expect_equal(dss$x_train$shape, c(10, 6))
})

test_that("The size of the data is correct.", {
  expect_equal(mmd$.length(), 150)
})

test_that(
  paste("The formula_data set function errors when no intercept is", 
        "explicitly provided."), {
  expect_error(formula_dataset(iris, Sepal.Length ~ . - 1))
})

fd <- formula_dataset(iris, Sepal.Length ~ .)

fd <- formula_dataset(iris, Species ~ .)
