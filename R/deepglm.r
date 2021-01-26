
#' @title Create a Supervised Deep Learning Model
#' 
#' @description A deep glm model genralizes a generalized linear model (GLM)from
#' a linear combination of independent variables estimating the expected
#' value of a dependent variable through an inverse link function to a 
#' non-linear combination of independent variables. That is, if 
#' Y is an n x 1 matrix generated from a distribution in the exponential 
#' family, X is a n x p matrix, the expected value of Y is mu,
#' b is a p x 1 vector of slope coefficients,
#' g is a link function and ginv is the inverse link function, then the GLM
#' can be expressed as: 
#'
#' mu = ginv(X \%*\% b)
#'
#' or, for a given row i
#'
#'   mu[i] = ginv(x[i, 1] * b[1] + x[2,1] * b[2] + ... x[i, p] * b[p]).
#' 
#' A deepglm extends this model to a non-linear combination of variables, f,
#' which can be expressed as combinations those variables in a specified
#' sequence of hidden layers, each with it's own activation
#'
#'   mu[i] = ginv( f(x[i, 1], x[i, 2], ..., x[i, p]) ).
#' 
#' The `dglm` function allows the user to create a variety of
#' deep generalized linear model, similar to `glm`. The interface is modeleled
#' after the `glm` function, taking a data.frame and formula. By default,
#' the dglm function tries to infer the type of model you would like to create
#' based on the dependent variable. However, this can easily overridden 
#' depending on the use case.
#' 
#' The network of independent variable relationships is specified through
#' the `hidden_layers` parameters. This is expected to be a vector where
#' the length of the vector specifies the number of hidden nodes and 
#' the element values specify how many nodes should be included at each hidden
#' layer.
#' @param data a `data.frame` (or derived) object whose columns correpsond to 
#' variables referenced in the `formula` argument.
#' @param formula a `formula` giving a symbolic description of the model to
#' be fitted.
#' @param hidden_layers a vector whose length corresponds to the number of 
#' hidden layers in the resulting model and whose values tell how many nodes
#' should added at the respective hidden layer. The default `integer()`
#' corresponds to a model with no hidden layers - a generalized linear model.
#' @param hidden_layer_activation the activations for each of the hidden 
#' layers (k_softmax for example). See the `keras` documentation for a 
#' complete list. By default linear activation is chosen.
#' @param hidden_layer_names the names of each of hidden layers. This 
#' option allows the user to reference and examine outputs of hidden layers.
#' @param hidden_layer_bias a vector of boolean values indicating if bias 
#' correction should be performed at each hidden layer.
#' @param loss the loss function used to fit the model.
#' @param optimizer the optimizer to use during the gradient descent procedure.
#' @param metrics the metrics to show while training the model and evaluating
#' the training history.
#' @param output_activation the activation for the output layer. By default,
#' keras::loss_mean_squared_error is chosen if the output is continuous or
#' count_model is `TRUE` and keras::loss_categorical_crossentropy is chosen if 
#' the output is categorical,
#' @param batch_size the number of contiguous samples to use when calculating
#' the gradient in the gradient descent procedure.
#' @param epochs the number of times to iterate through the complete data set.
#' @param verbose should extra information, like the metrics per epoch, 
#' be written to the output. The default is TRUE.
#' @param valdation_split the percent the proporation of data to use to
#' evaluate (rather than train) a model. The default is 0.2 (20\%).
#' @param count_model is the dependent variable count data? By default,
#' if the dependent variable is non-negative and integer or an 
#' ordered factor, this will be set to true.
#' @param name the name of the model.
#' @importFrom fu make_variable_desc
#' @importFrom keras keras_model_sequential layer_dense %>% compile fit
#' optimizer_adadelta loss_mean_squared_error loss_categorical_crossentropy
#' @examples
#' library(palmerpenguins)
#' 
#' data(penguins)
#'
#' # A model to estimate penguin species with 1 hidden layer  with 16 nodes and
#' # sigmoid activation.
#' species_fit <- dglm(penguins, island ~ ., 
#'                     hidden_layers = c(16),
#'                     hidden_layers_activation = c("sigmoid"))
#' predict(species_fit, penguins) 
#' # What is the total prediction accuracy?
#' sum(predict(species_fit, penguins) == penguins$species) / nrow(penguins)
#' @export
dglm <- function(dataset, 
                 formula, 
                 hidden_layers = integer(), 
                 hidden_layers_activation = 
                   rep("linear", length(hidden_layers)),
                 hidden_layer_names = 
                   paste("hidden_layer", seq_along(hidden_layers), sep = "_"),
                 hidden_layer_bias = rep(TRUE, length(hidden_layers)),
                 loss = NULL,
                 optimizer = optimizer_adadelta(),
                 metrics = NULL, 
                 output_activation = NULL,
                 output_activation_bias = TRUE,
                 batch_size = nrow(data),
                 epochs = 1000,
                 verbose = FALSE,
                 validation_split = 0.2,
                 callbacks = 
                  list(
                    callback_early_stopping(min_delta = 0.1, 
                                            patience = 500,
                                            restore_best_weights = TRUE)),
                 name = NULL) {

  xf <- model.frame(formula, data)

  var_desc <- make_variable_desc(xf, formula)

  error_on_more_than_one_dep_var(var_desc)
  error_on_no_indep_var(var_desc)
  error_on_conditional_var(var_desc)
  error_on_unsupported_dependent_var(var_desc, c("numeric", "factor"))
  error_on_bad_hidden_layer_desc(hidden_layers, hidden_layers_activation)

  x_train <- make_model_matrix(formula, xf)

  # make sure there is an intercept, if not warning
  mm_column_var_assign <- attributes(x_train)$assign
  column_var_name <- names(xf)
  mm_column_var_name <- colnames(x_train)

  model <- 
    create_input_and_hidden_layers(
      x_train, 
      hidden_layers,
      hidden_layers_activation,
      hidden_layer_names,
      hidden_layer_bias,
      name)

  if (var_desc$class[var_desc$role == "dependent"] == "factor") {

   #              categorical_loss = loss_categorical_crossentropy,
   #              continuous_loss = loss_mean_squared_error,
   # output_activation <- "softmax"

    if (is.null(loss)) {
      loss <- loss_categorical_crossentropy
    }
    if (is.null(metrics)) {
      metrics <- "categorical_accuracy"
    }

    if (is.null(output_activation)) {
      output_activation <- "softmax"
    }

    # Are the encodings other than one-hot to consider?  
    oh <- make_one_hot(xf[[var_desc$name[var_desc$role == "dependent"]]])

    input_shape <- NULL
    if (length(hidden_layers) == 0) {
      input_shape <- ncol(x_train)
    }

    model %>% 
      layer_dense(
          units=length(var_desc$levels[var_desc$role == "dependent"][[1]]),
          input_shape = input_shape,
          name = paste(output_activation, "output", sep = "_"),
          activation = output_activation,
          use_bias = output_activation_bias,
          kernel_initializer = "random_normal") 
    
    type <- "categorical_deepglm"
    loss <- loss_categorical_crossentropy
    y_train <- 
      to_one_hot(xf[[var_desc$name[var_desc$role == "dependent"]]],
                   oh)
  } else if (var_desc$class[var_desc$role == "dependent"] == "numeric") {
    input_shape <- NULL
    if (length(hidden_layers) == 0) {
      input_shape <- ncol(x_train)
    }

    if (is.null(loss)) {
      loss <- loss_mean_squared_error
    }

    if (is.null(metrics)) {
      metrics = "mse"
    }
    model %>% 
      layer_dense(units = 1, input_shape = input_shape, 
                  name = "continuous_output", use_bias = TRUE,
                  activation = output_activation,
                  kernel_initializer = "random_normal")

    type <- "continuous_deepglm"
    y_train <- as.matrix(xf[[var_desc$name[var_desc$role == "dependent"]]])
  } else {
    stop("Unsupported dependent variable type.")
  }

  if (is.null(metrics)) {
    metrics <- loss
  }

  mm_col_names <- colnames(x_train)
  
  model %>% 
    compile(loss = loss, optimizer = optimizer, metrics = metrics) 

  history <- model %>%
    fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
        validation_split = validation_split, verbose = verbose,
        callbacks = callbacks)

  ret <- list(formula = formula, 
              hidden_layers = hidden_layers, 
              hidden_layers_activation = hidden_layers_activation,
              column_var_name = column_var_name,
              mm_column_var_name = mm_column_var_name,
              mm_column_var_assign = mm_column_var_assign,
              loss = loss,
              var_desc = var_desc, 
              model = model, 
              history = history,
              mm_col_names = mm_col_names)
  class(ret) <- c(type, "deepglm")
  ret
}

#' @export
plot_training_history <- function(x, metrics, smooth, theme_bw) {
  UseMethod("plot_training_history", x)
}

#' @importFrom crayon red
#' @export
plot_training_history.default <- function(x, metrics, smooth, theme_bw) {
  print(red("Don't know how to plot training history for an object of type:",
            paste(class(x), collapse = " ")))
}

#' @importFrom ggplot2 ggplot aes aes_ geom_point scale_shape theme_bw 
#' theme_minimal geom_smooth facet_grid element_text scale_x_continuous theme
#' element_blank element_rect
#' @importFrom tools toTitleCase
#' @export
plot_training_history.deepglm <- function(x, metrics = NULL, 
    smooth = getOption("keras.plot.history.smooth", TRUE), 
    theme_bw = FALSE) {

  df <- as.data.frame(x$history)
  if (is.null(metrics))
      metrics <- Filter(function(name) !grepl("^val_", name),
          names(x$history$metrics))
  df <- df[df$metric %in% metrics, ]
  do_validation <- any(grepl("^val_", names(x$history$metrics)))
  
  names(df) <- toTitleCase(names(df))
  df$Metric <- toTitleCase(as.character(df$Metric))
  df$Data <- toTitleCase(as.character(df$Data))

  int_breaks <- function(x) pretty(x)[pretty(x)%%1 == 0]
  if (do_validation) {
    if (theme_bw) {
      p <- ggplot(df, 
                  aes_(~Epoch, ~Value, color = ~Data, fill = ~Data, 
                       linetype = ~Data, shape = ~Data))
    } else {
      p <- ggplot(df, aes_(~Epoch, ~Value, color = ~Data, fill = ~Data))
    }
  } else {
    p <- ggplot(df, ggplot2::aes_(~Epoch, ~Value))
  }
  smooth_args <- list(se = FALSE, method = "loess", na.rm = TRUE)
  if (theme_bw) {
    smooth_args$size <- 0.5
    smooth_args$color <- "gray47"
    p <- p + 
      theme_bw() + 
      geom_point(col = 1, na.rm = TRUE, size = 2) + 
      scale_shape(solid = FALSE)
  }
  else {
    p <- p + geom_point(shape = 21, col = 1, na.rm = TRUE)
  }
  if (smooth && x$history$params$epochs >= 10) {
    p <- p + do.call(geom_smooth, smooth_args)
  }
  p + facet_grid(Metric ~ ., switch = "y", scales = "free_y") + 
    scale_x_continuous(breaks = int_breaks) +
    theme_minimal() +
    theme(axis.title.y = element_blank(),
          strip.placement = "outside", 
          strip.text = element_text(colour = "black", size = 11), 
          strip.background = element_rect(fill = NA, color = NA))
}
