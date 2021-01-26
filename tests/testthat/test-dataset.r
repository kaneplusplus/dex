context("Test Generating Datasets from data.frame Objects")

data(iris)

mm <- model.matrix(Sepal.Length ~ . - 1, iris)

mmd <- model_matrix_dataset(
  torch_tensor(matrix(data = iris$Sepal.Length, ncol = 1)),
  torch_tensor(mm),
  "Sepal.Length",
  colnames(mm))

expect_type(mmd$.getitem(1), "list")

dss <- mmd$.getitem(1:10)

expect_equal(dss$x$shape, c(10, 6))

expect_equal(mmd$.length(), 150)

expect_error(formula_dataset(iris, Sepal.Length ~ . - 1))

fd <- formula_dataset(iris, Sepal.Length ~ ., iris)
