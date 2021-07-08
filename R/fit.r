
fit_params <- function(ds, model, epochs = 1000, batch_size = 10,
                  loss_fun = nnf_mse_loss, shuffle = TRUE,
                  optimizer = optim_adam(model$parameters),
                  progress = interactive()) {

  ilh <- olh <- rep(NA_real_, epochs)
  min_olh_loss <- Inf
  test_data <- ds$get_test()
  batch_loss <- c()
  if (progress) {
    pb <- progress_bar$new(total = epochs)
  }
  for (epoch in seq_len(epochs)) {
    if (progress) {
      pb$tick()
    }
    dl <- dataloader(ds, batch_size = batch_size, shuffle = shuffle)
    for (batch in torch::enumerate(dl)) {
      x_batch <- batch[[1]]$reshape(batch[[1]]$shape[-2])
      y_batch <- batch[[2]]$reshape(batch[[2]]$shape[-2])
      optimizer$zero_grad()
      l <- loss_fun(model(x_batch), y_batch)
      l$backward()
      optimizer$step()
      batch_loss <- c(batch_loss, l$item())
    }
    ilh[epoch] <- mean(batch_loss)
    olh[epoch] <- loss_fun(model(test_data$x), test_data$y)$item()
    if (olh[epoch] < min_olh_loss) {
      best_model <- model$clone()
      min_olh_loss <- olh[epoch]
    }
  }
  list(model = model, best_model = best_model, ilh = ilh, olh = olh)
}
