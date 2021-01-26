#' @title Create a Torch Dataset from a Data Frame and Formula
#' @export
formula_dataset <- function(x, formula, ...) {
  UseMethod("formula_dataset", x)
}

formula_dataset.default <- function(x, formula, ...) {
  stop("Don't know how to make a formula_dataset from an object of type ",
       paste(class(x), collapse = " "), ".")
}

numeric_dep_var <- function(x, dep_var_name) {
}


# Note that unused levels need to be removed before this function is
# called. Also, no Na's

#' @title Create a Torch Dataset from a Model Matrix
#' @export
model_matrix_dataset <- dataset(
  "data_frame_dataset",

  initialize = function(y, x, y_name, x_names) {
    self$x <- x
    self$y <- y
    self$x_names <- x_names
    self$y_name <- y_name
  },

  .getitem = function(i) {
    list(y = self$y[i,,drop = FALSE], x = self$x[i,,drop = FALSE])
  },
  
  .length = function() {
    self$x$shape[1]
  },

  var_names = function() {
    list(y = self$y_name, x = self$x_names)
  }
)

#' @importFrom Formula Formula model.part
formula_dataset.data.frame <- function(x, formula, contrasts_arg = NULL,
  na_action = na.fail, xlev = NULL, ...) {

  mf <- model.frame(formula = formula, data = x, na.action = na_action, 
                    xlev = xlev)

  mm <- model.matrix(formula, mf)

  if (!("(Intercept)" %in% colnames(mm))) {
    stop("The intercept should be removed with the bias argument.")
  } else {
    i_col <- which("(Intercept)" == colnames(mm))
    mm <- mm[, -i_col, drop = FALSE]
  }

  x <- torch_tensor(mm)

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
    y <- torch_tensor(matrix(mf[[dep_var]], ncol = 1))
  } else {
    stop("Don't know how to handle dependent variable of type ", 
         paste(class(x[[dep_var]]), collapse = " "), ".")
  }

  model_matrix_dataset(y, x, dep_var, colnames(mm))
}

