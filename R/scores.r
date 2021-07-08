overall_precision <- function(input, target) {
  oh_target <- nnf_one_hot(target)
  num <- input$multiply(oh_target)$nansum(dim = 1)
  denom <- input$nansum(dim = 1)
  denom[denom == 0] <- 1
  num$div(denom)$mean(dim = 2)
}

overall_recall <- function(input, target) {
  oh_target <- nnf_one_hot(target)
  num <- input$multiply(oh_target)$nansum(dim = 1)
  denom <- oh_target$nansum(dim = 1)
  denom[denom == 0] <- 1
  num$div(denom)$mean(dim = 2)
}

# This needs to be tested.
dice <- function(input, target) {
  oh_target <- nnf_one_hot(target)
  num <- input$multiply(oh_target)$sum(dim = 2)
  denom <- (oh_target + input)$sum(dim = 2)
  num$div(denom)$sum(2)$mean()
}

neg_dice <- function(input, target) {
  -dice(input, target)
}

overall_f <- function(input, target, beta = 1) {
  prec <- overall_precision(input, target)
  rec <- overall_recall(input, target)
  if (!any(is.finite(as.numeric(prec))) || !any(is.finite(as.numeric(rec)))) {
    browser()
  }
  num <- (1 + beta^2) * prec * rec
  denom <- (beta^2 * prec + rec)
  denom[denom == 0] <- 1
  num$div(denom)$mean()
}

neg_overall_f <- function(input, target, beta = 1) {
  -overall_f(input, target, beta)
}


