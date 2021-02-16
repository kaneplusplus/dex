
nn_list_module_gen <- nn_module(
  "nn_list_module",
  
  initialize = function(module_list) {
    self$module_list <- nn_module_list(module_list)
  },

  forward = function(x) {
    for (i in seq_along(self$module_list)) { 
      x <- self$module_list[[i]](x)
    }
    x
  }
)

#' @importFrom checkmate assert check_logical
#' @importFrom foreach foreach %do%
#' @export
create_nn_from_module_list <- 
  function(
    input_size, 
    hidden_layers, 
    hidden_activations,
    hidden_layer_bias) {

  assert(
    length(input_size) == 1,
    input_size == as.integer(input_size),
    combine = "and"
  )

  assert(check_logical(hidden_layer_bias))

  assert(
    all(map_lgl(hidden_activations, ~ inherits(.x, "nn_module_generator"))))

  if (length(hidden_layers) != length(hidden_layer_bias)) {
    stop("hidden_layers and hidden_layer_bias must have the same length")
  }

  is <- input_size
  layers <- foreach(i = seq_along(hidden_layers)) %do% {
    ret <- hidden_activations[[i]](
      in_features = is, 
      out_features = hidden_layers[i], 
      bias = hidden_layer_bias[i])

    is <- hidden_layers[i]
    ret
  }

  nn_list_module_gen(layers)
}
