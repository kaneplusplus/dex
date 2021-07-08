
#' @export
make_test_train <- function(nsamples, prop = 0.9) {
  train_inds <- sample(seq_len(nsamples), size = floor(nsamples * prop))
  test_inds <- setdiff(seq_len(nsamples), train_inds)
  ret <- rep("train", nsamples)
  ret[test_inds] <- "test"
  ret
}
