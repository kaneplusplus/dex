
#' @importFrom stats terms
#' @importFrom crayon yellow
make_model_matrix <- function(form, x) {
  ret <- model.matrix(form, x)
  ts <- terms(form, data = x)
  if (attributes(ts)$intercept == 0) {
    stop(yellow("No intercept specified. This should be done using the", 
                "use_bias parameter."))
  } else {
    ret <- ret[,-1]
  }
  ret
}

