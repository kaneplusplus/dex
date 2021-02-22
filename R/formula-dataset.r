# Note that unused levels need to be removed before this function is
# called. Also, no Na's

#' @title Create a Torch Dataset from a Model Matrix
#' @export
model_matrix_dataset <- dataset(
  "model_matrix_dataset",

  initialize = function(
    x_train, 
    y_train, 
    x_test = NULL, 
    y_test = NULL, 
    x_name = NULL,
    x_mm_name = NULL,
    y_name = NULL,
    x_0 = NULL,
    y_0 = NULL,
    formula = NULL,
    na_action = NULL,
    mm_attr = NULL) {

    self$x_train <- x_train
    self$y_train <- y_train
    self$x_test <- x_test
    self$y_test <- y_test
    self$x_name <- x_name
    self$x_mm_name <- x_mm_name
    self$y_name <- y_name
    self$x_0 <- x_0
    self$y_0 <- y_0
    self$formula <- formula
    self$na_action <- na_action
    self$mm_attr <- mm_attr
  },

  .getitem = function(i) {
    list(x_train = self$x_train[i,,drop = FALSE],
         y_train = self$y_train[i,,drop = FALSE])
  },
  
  .length = function() {
    self$x_train$shape[1]
  },

  train = function() {
    list(
      x_train = self$x_train,
      y_train = self$y_train)
  },

  test = function() {
    list(
      x_test = self$x_test,
      y_test = self$y_test) 
  }
)

remove_intercept <- function(mm) {
  if (!("(Intercept)" %in% colnames(mm))) {
    stop("The intercept should be removed with the bias argument.")
  } else {
    i_col <- which("(Intercept)" == colnames(mm))
    mm_attr <- attributes(mm)
    mm <- mm[, -i_col, drop = FALSE]
    mm_attr$dimnames[[2]] <- mm_attr$dimnames[[2]][-i_col]
    mm_attr$assign <- mm_attr$assign[-1]
    mm_attr$dim[2] <- mm_attr$dim[2] - 1
    attributes(mm) <- mm_attr
    mm
  }
}

#' @title Create a Torch Dataset from a Data Frame and Formula
#' @export
formula_dataset <- function(x, formula, ...) {
  UseMethod("formula_dataset", x)
}

#' @export
formula_dataset.default <- function(x, formula, ...) {
  stop("Don't know how to make a formula_dataset from an object of type ",
       paste(class(x), collapse = " "), ".")
}

#' @importFrom Formula Formula model.part
#' @importFrom rsample validation_split analysis assessment
#' @importFrom tibble as_tibble
#' @importFrom stats na.fail
#' @importFrom future future value
#' @export
formula_dataset.data.frame <- function(x, formula, prop = 0.8, 
  strata = NULL, breaks = 4, contrasts_arg = NULL, na_action = na.fail, 
  xlev = NULL, ...) {

  mf <- model.frame(formula = formula, data = x, na.action = na_action, 
                    xlev = xlev)

  if (0 < prop && prop < 1) {
    vs <- validation_split(data = mf, prop = prop, strata = strata, 
                           breaks = breaks)
  } else if (prop == 1) {
    browser()
  } else {
    stop("prop must be (0,1]")
  }

  mm_trainf <- 
    future(
      remove_intercept(
        model.matrix(
          formula, 
          analysis(vs$splits[[1]]), 
          contrasts.arg = contrasts_arg, 
          xlev = xlev)))

  mm_testf <- 
    future(
      remove_intercept(
        model.matrix(
          formula, 
          assessment(vs$splits[[1]]), 
          contrasts.arg = contrasts_arg, 
          xlev = xlev)))

  form <- Formula(formula)
  dep_var <- names(model.part(form, mf, lhs = 1))
  if (length(dep_var) > 1) {
    stop("Only one dependent variable should be specified.")
  }

  if (is.factor(mf[[dep_var]]) && !is.ordered(mf[[dep_var]])) {
    num_levels <- length(levels(mf[[dep_var]]))

    to_one_hot <- . %>%
      as.integer() %>%
      torch_tensor() %>%
      nnf_one_hot(num_classes = num_levels)
      
    y_train <- analysis(vs$splits[[1]])[[dep_var]] %>% to_one_hot()
    # To cast to float.
    y_train <- y_train + torch_zeros(y_train$shape)

    y_test <- assessment(vs$splits[[1]])[[dep_var]] %>% to_one_hot()
    y_test <- y_test + torch_zeros(y_test$shape)

  } else if (is.factor(mf[[dep_var]]) && !is.ordered(mf[[dep_var]])) {
    stop("Ordered factors are not yet supported.")
  } else if (is.numeric(mf[[dep_var]])) {

    to_dependent <- . %>%
      matrix(ncol = 1) %>%
      torch_tensor()

    y_train <- analysis(vs$splits[[1]])[[dep_var]] %>% to_dependent()
      matrix(ncol = 1) %>%
      torch_tensor()
    y_test <- assessment(vs$splits[[1]])[[dep_var]] %>% to_dependent()
      
  } else {
    stop("Don't know how to handle dependent variable of type ", 
         paste(class(x[[dep_var]]), collapse = " "), ".")
  }

  mm_train <- value(mm_trainf)
  x_train <- torch_tensor(mm_train)

  mm_test<- value(mm_testf)
  x_test <- torch_tensor(mm_test)

  x_name <- names(model.part(form, mf, rhs = NULL))
  model_matrix_dataset(
    x_train, 
    y_train, 
    x_test, 
    y_test, 
    x_name = x_name,
    x_mm_name = colnames(mm_train),
    y_name = dep_var, 
    x_0 = as_tibble(mf[c(), x_name]),
    y_0 = as_tibble(mf[[dep_var]][c()]), 
    na_action = na_action,
    mm_attr = attributes(mm_train))
}

