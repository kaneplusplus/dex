#' @title Create a Torch Dataset from a Data Frame and Formula
#' @export
tt_formula_dataset <- function(x, formula, ...) {
  UseMethod("formula_dataset", x)
}

tt_formula_dataset.default <- function(x, formula, ...) {
  stop("Don't know how to make a formula_dataset from an object of type ",
       paste(class(x), collapse = " "), ".")
}

# Note that unused levels need to be removed before this function is
# called. Also, no Na's

#' @title Create a Torch Dataset from a Model Matrix
#' @export
tt_model_matrix_dataset <- dataset(
  "tt_model_matrix_dataset",

  initialize = function(y_train, x_train, y_test, x_test, y_name, 
                        x_names, y_0 = typeof(a)) {
    self$x_train <- x_train
    self$y_train <- y_train
    self$x_test <- x_test
    self$y_test <- y_test
    self$x_names <- x_names
    self$y_name <- y_name
    self$y_0
  },

  .getitem = function(i) {
    list(y_train = self$y_train[i,,drop = FALSE], 
         x_train = self$x_train[i,,drop = FALSE])
  },
  
  .length = function() {
    self$x_train$shape[1]
  },

  var_names = function() {
    list(y = self$y_name, x = self$x_names)
  },

  y_0 = function() {
    self$y_0
  },

  train = function() {
    list(y_train = self$y_train, x_train = self$x_train)
  },

  test = function() {
    list(y_test = self$y_test, x_test = self$x_test)
  }
)

remove_intercept <- function(mm) {
  if (!("(Intercept)" %in% colnames(mm))) {
    stop("The intercept should be removed with the bias argument.")
  } else {
    i_col <- which("(Intercept)" == colnames(mm))
    mm[, -i_col, drop = FALSE]
  }
}

#' @importFrom Formula Formula model.part
#' @importFrom rsample validation_split
tt_formula_dataset.data.frame <- function(x, formula, prop = 3/4, 
  strata = NULL, breaks = 4, 
  contrasts_arg = NULL, na_action = na.fail, xlev = NULL, ...) {

  mf <- model.frame(formula = formula, data = x, na.action = na_action, 
                    xlev = xlev)

  vs <- validation_split(data = mf, prop = prop, strata = strata, 
                         breaks = breaks)

  mm_train <- 
    remove_intercept(
      model.matrix(
        formula, 
        analysis(vs$splits[[1]]), 
        contrasts.arg = contrasts_arg, 
        xlev = xlev))

  mm_test <- 
    remove_intercept(
      model.matrix(
        formula, 
        assessment(vs$splits[[1]]), 
        contrasts.arg = contrasts_arg, 
        xlev = xlev))

  x_train <- torch_tensor(mm_train)

  x_test <- torch_tensor(mm_test)

  form <- Formula(formula)
  dep_var <- model.part(form, mf, lhs = 1)
  if (length(dep_var) > 1) {
    stop("Only one dependent variable should be specified.")
  }

  if (is.factor(mf[[dep_var]]) && !is.ordered(mf[[dep_var]])) {
    stop("Factors are not yet supported.")
  } else if (is.factor(mf[[dep_var]]) && !is.ordered(mf[[dep_var]])) {
    stop("Ordered factors are not yet supported.")
  } else if (is.numeric(mf[[dep_var]])) {
    y_train <- 
      torch_tensor(matrix(analysis(mf$splits[[1]])[[dep_var]], ncol = 1))
    y_test <- 
      torch_tensor(matrix(assessment(mf$splits[[1]])[[dep_var]], ncol = 1))
  } else {
    stop("Don't know how to handle dependent variable of type ", 
         paste(class(x[[dep_var]]), collapse = " "), ".")
  }

  tt_model_matrix_dataset(y_train, x_train, y_test, y_train, 
                          dep_var, colnames(mm_test), mf[[dep_var]][c()])
}

