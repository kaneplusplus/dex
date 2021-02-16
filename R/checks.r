
#' @importFrom crayon red
#' @export
error_on_bad_hidden_layer_desc <- 
  function(hidden_layers, hidden_layers_activation) {

  if (length(hidden_layers) != length(hidden_layers_activation)) {
    stop(red("hidden_layers and their activations must have the same length."))
  }
  invisible(TRUE)
}

#' @importFrom crayon red
error_if_not_model_matrix <- function(mm) {
  if (!all(c("dim", "dimnames", "assign", "contrasts") %in%
           names(attributes(mm)))) {
    stop(red("Argument mm must be a model matrix."))
  }
  invisible(TRUE)
}

#' @importFrom crayon red
error_if_not_data_frame <- function(df) {
  if (!inherits(df, "data.frame")) {
    stop(red("Argument df must be inherited from a data.frame."))
  }
  invisible(TRUE)
}
