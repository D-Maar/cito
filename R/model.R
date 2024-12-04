build_model <- function(object) {
  if(inherits(object, "citodnn")) {
    net <- build_dnn(input = object$model_properties$input,
                     output = object$model_properties$output,
                     hidden = object$model_properties$hidden,
                     activation = object$model_properties$activation,
                     bias = object$model_properties$bias,
                     dropout = object$model_properties$dropout,
                     embeddings = object$model_properties$embeddings)
  } else if(inherits(object, "citocnn")) {
    net <- build_cnn(input_shape = object$model_properties$input,
                     output_shape = object$model_properties$output,
                     architecture = object$model_properties$architecture)
  } else {
    stop("model not of class citodnn or citocnn")
  }
  return(net)
}



build_dnn = function(input, output, hidden, activation, bias, dropout, embeddings) {

  layers = list()
  if(is.null(hidden)) {
    if(is.null(embeddings)) layers[[1]] = torch::nn_linear(input, out_features = output, bias = bias)
    else layers[[1]] = torch::nn_linear(input+sum(sapply(embeddings$args, function(a) a$dim)), out_features = output, bias = bias)
  } else {
    if(length(hidden) != length(activation)) activation = rep(activation, length(hidden))
    if(length(hidden)+1 != length(bias)) bias = rep(bias, (length(hidden)+1))
    if(length(hidden) != length(dropout)) dropout = rep(dropout,length(hidden))

    counter = 1
    for(i in 1:length(hidden)) {
      if(counter == 1) {
        if(is.null(embeddings)) layers[[1]] = torch::nn_linear(input, out_features = hidden[1], bias = bias[1])
        else layers[[1]] = torch::nn_linear(input+sum(sapply(embeddings$args, function(a) a$dim)), out_features = hidden[1], bias = bias[1])
      } else {
        layers[[counter]] = torch::nn_linear(hidden[i-1], out_features = hidden[i], bias = bias[i])
      }
      counter = counter+1
      layers[[counter]]<- get_activation_layer(activation[i])

      counter = counter+1
      if(dropout[i]>0) {
        layers[[counter]] = torch::nn_dropout(dropout[i])
        counter = counter+1
      }
    }

    if(!is.null(output)) layers[[length(layers)+1]] = torch::nn_linear(hidden[i], out_features = output, bias = bias[i+1])
  }
  self = NULL
  if(!is.null(embeddings)) {
    net_embed <- torch::nn_module(
      initialize = function() {
        for(i in 1:length(embeddings$dims)) {
          self[[paste0("e_", i)]] = torch::nn_embedding(embeddings$dims[i], embeddings$args[[i]]$dim  )
        }
        for(i in 1:length(layers)) {
          self[[paste0("l_",i)]] = layers[[i]]
        }

      },
      forward = function(input_hidden, input_embeddings) {
        n_em = length(embeddings$dims)
        embeds =
          lapply(1:n_em, function(j) {
          return( torch::torch_squeeze( self[[paste0("e_", j)]](input_embeddings[,j,drop=FALSE]) ,2))
        })

        x = torch::torch_cat(c(embeds, input_hidden ), 2L)
        for(i in 1:length(layers)) {
          x = self[[paste0("l_",i)]](x)
        }
        return(x)
      }
    )
    net = net_embed()

    # set weights if provided by function
    for(i in 1:length(embeddings$dims)) {
      if(!is.null(embeddings$args[[i]]$weights)) net[[paste0("e_", i)]]$weight$set_data( torch::torch_tensor(embeddings$args[[i]]$weights,
                                                                                                             dtype = net[[paste0("e_", i)]]$weight$dtype))
    }

    # turn-off gradient if desired
    # set weights if provided by function
    for(i in 1:length(embeddings$dims)) {
      if(!(embeddings$args[[i]]$train)) net[[paste0("e_", i)]]$weight$requires_grad = FALSE
    }

  } else {
    net = do.call(torch::nn_sequential, layers)
  }
  return(net)
}
convBlock = torch::nn_module(
  initialize = function(in_channels, out_channels, kernel_size, stride, padding, dropout){
    self$conv=torch::nn_conv2d(in_channels = in_channels
                        , out_channels = out_channels
                        , kernel_size = kernel_size
                        , stride = stride
                        , padding = padding)
    self$batchNorm = torch::nn_batch_norm2d(out_channels)
    self$activation = torch::nn_relu()
    self$dropout_val = dropout
    if(dropout>0){
      self$dropout = torch::nn_dropout2d(p=dropout)
    }
  },
  forward = function(x){
    if(self$dropout_val>0){
      x |>
        self$conv() |>
        self$batchNorm() |>
        self$activation()|>
        self$dropout()
    } else {
      x |>
        self$conv() |>
        self$batchNorm() |>
        self$activation()
    }

  }
)
inceptionBlock_A_2D = torch::nn_module(


  initialize = function(in_channels, channel_mult=16L,dropout=0){

    self$branchA = torch::nn_sequential(
      convBlock(
        in_channels = in_channels
        , out_channels = 4L*channel_mult
        , kernel_size = 1L
        , stride = 1L
        , padding = 0L
        , dropout = dropout),
      convBlock(
        in_channels = 4L*channel_mult
        , out_channels = 6L*channel_mult
        , kernel_size = 3L
        , stride = 1L
        , padding = 1L
        , dropout = dropout),
      convBlock(
        in_channels = 6L*channel_mult
        , out_channels = 6L*channel_mult
        , kernel_size = 3L
        , stride = 1L
        , padding = 1L
        , dropout = dropout)
    )

    self$branchB = torch::nn_sequential(
      convBlock(
        in_channels = in_channels
        , out_channels = 3L*channel_mult
        , kernel_size = 1L
        , stride = 1L
        , padding = 0L
        , dropout = dropout
      ),
      convBlock(
        in_channels = 3L*channel_mult
        , out_channels = 4L*channel_mult
        , kernel_size = 3L
        , stride = 1L
        , padding = 1L
        , dropout = dropout
      )
    )

    self$branchC = torch::nn_sequential(
      torch::nn_avg_pool2d(
        kernel_size = c(3L,3L)
        , stride = 1L
        , padding = 1L
      ),
      convBlock(
        in_channels = in_channels
        , out_channels = 4L*channel_mult
        , kernel_size = 1L
        , stride = 1L
        , dropout = dropout
        , padding = 0L
      )
    )

    self$branchD = convBlock(
      in_channels = in_channels
      , out_channels = 4L*channel_mult
      , kernel_size = 1L
      , stride = 1L
      , padding = 0L
      , dropout = dropout
    )

  },
  forward = function(x){
    branchARes = self$branchA(x)
    branchBRes = self$branchB(x)
    branchCRes = self$branchC(x)
    branchDRes = self$branchD(x)
    # print(branchARes$size())
    # print(branchBRes$size())
    # print(branchCRes$size())
    # print(branchDRes$size())
    res = torch::torch_cat(list(branchARes, branchBRes, branchCRes, branchDRes),2L)
    res
  }
)
inceptionBlock_A_2D_reduction = torch::nn_module(


  initialize = function(in_channels, channel_mult=16L, dropout=0){

    self$branchA = torch::nn_sequential(
      convBlock(
        in_channels = in_channels
        , out_channels = 4L*channel_mult
        , kernel_size = 1L
        , stride = 1L
        , dropout = dropout
        , padding = c(0L, 1L)
      ),
      convBlock(
        in_channels = 4L*channel_mult
        , out_channels = 6L*channel_mult
        , kernel_size = 3L
        , dropout = dropout
        , stride = 1L
        , padding = 0L
      ),
      convBlock(
        in_channels = 6L*channel_mult
        , out_channels = 6L*channel_mult
        , kernel_size = 3L
        , stride = 1L
        , padding = 1L
        , dropout = dropout
      )
    )

    self$branchB = torch::nn_sequential(
      convBlock(
        in_channels = in_channels
        , out_channels = 3L*channel_mult
        , kernel_size = 1L
        , stride = 1L
        , dropout = dropout
        , padding = c(0L, 1L)
      ),
      convBlock(
        in_channels = 3L*channel_mult
        , out_channels = 4L*channel_mult
        , kernel_size = 3L
        , dropout = dropout
        , stride = 1L
        , padding = 0L
      )
    )

    self$branchC = torch::nn_sequential(
      torch::nn_avg_pool2d(
        kernel_size = c(3L,3L)
        , stride = 1L
        , padding = 0L
      ),
      convBlock(
        in_channels = in_channels
        , out_channels = 4L*channel_mult
        , kernel_size = 1L
        , stride = 1L
        , dropout = dropout
        , padding = c(0L,1L)
      )
    )

    self$branchD = convBlock(
      in_channels = in_channels
      , out_channels = 4L*channel_mult
      , kernel_size = c(3L,1L)
      , stride = 1L
      , dropout = dropout
      , padding = 0L
    )

  },
  forward = function(x){
    branchARes = self$branchA(x)
    branchBRes = self$branchB(x)
    branchCRes = self$branchC(x)
    branchDRes = self$branchD(x)
    # print(branchARes$size())
    # print(branchBRes$size())
    # print(branchCRes$size())
    # print(branchDRes$size())
    res = torch::torch_cat(list(branchARes, branchBRes, branchCRes, branchDRes),2L)
    res
  }
)

inceptionBlock_A_100_reduction = torch::nn_module(


  initialize = function(in_channels, channel_mult=16L, dropout=0){

    self$branchA = torch::nn_sequential(
      convBlock(
        in_channels = in_channels
        , out_channels = 4L*channel_mult
        , kernel_size = c(1L, 100L)
        , stride = 1L
        , dropout = dropout
        , padding = c(0L, 50L)
      ),
      convBlock(
        in_channels = 4L*channel_mult
        , out_channels = 6L*channel_mult
        , kernel_size = c(3L, 101L)
        , dropout = dropout
        , stride = 1L
        , padding = c(1L, 50L)
      ),
      convBlock(
        in_channels = 6L*channel_mult
        , out_channels = 6L*channel_mult
        , kernel_size = c(3L, 101L)
        , stride = 1L
        , padding = 0L
        , dropout = dropout
      )
    )

    self$branchB = torch::nn_sequential(
      convBlock(
        in_channels = in_channels
        , out_channels = 3L*channel_mult
        , kernel_size = c(1L, 100L)
        , stride = 1L
        , dropout = dropout
        , padding = c(0L, 50L)
      ),
      convBlock(
        in_channels = 3L*channel_mult
        , out_channels = 4L*channel_mult
        , kernel_size = c(3L, 101L)
        , dropout = dropout
        , stride = 1L
        , padding = 0L
      )
    )

    self$branchC = torch::nn_sequential(
      torch::nn_avg_pool2d(
        kernel_size = c(3L,101L)
        , stride = 1L
        , padding = c(0L, 50L)
      ),
      convBlock(
        in_channels = in_channels
        , out_channels = 4L*channel_mult
        , kernel_size = c(1L, 100L)
        , stride = 1L
        , dropout = dropout
        , padding = 0L
      )
    )

    self$branchD = convBlock(
      in_channels = in_channels
      , out_channels = 4L*channel_mult
      , kernel_size = c(3L,100L)
      , stride = 1L
      , dropout = dropout
      , padding = 0L
    )

  },
  forward = function(x){
    branchARes = self$branchA(x)
    branchBRes = self$branchB(x)
    branchCRes = self$branchC(x)
    branchDRes = self$branchD(x)
    # print(branchARes$size())
    # print(branchBRes$size())
    # print(branchCRes$size())
    # print(branchDRes$size())
    res = torch::torch_cat(list(branchARes, branchBRes, branchCRes, branchDRes),2L)
    res
  }
)


inceptionBlock_A_1D = torch::nn_module(


  initialize = function(in_channels, channel_mult=16L, dropout=0){

    self$branchA = torch::nn_sequential(
      convBlock(
        in_channels = in_channels
        , out_channels = 4L*channel_mult
        , kernel_size = 1L
        , stride = 1L
        , dropout = dropout
        , padding = 0L
      ),
      convBlock(
        in_channels = 4L*channel_mult
        , out_channels = 6L*channel_mult
        , kernel_size = c(1L,3L)
        , stride = 1L
        , dropout = dropout
        , padding = c(0L,1L)
      ),
      convBlock(
        in_channels = 6L*channel_mult
        , out_channels = 6L*channel_mult
        , kernel_size = c(1L,3L)
        , dropout = dropout
        , stride = 1L
        , padding = c(0L,1L)
      )
    )

    self$branchB = torch::nn_sequential(
      convBlock(
        in_channels = in_channels
        , out_channels = 3L*channel_mult
        , kernel_size = 1L
        , stride = 1L
        , dropout = dropout
        , padding = 0L
      ),
      convBlock(
        in_channels = 3L*channel_mult
        , out_channels = 4L*channel_mult
        , kernel_size = c(1L,3L)
        , stride = 1L
        , dropout = dropout
        , padding = c(0L,1L)
      )
    )

    self$branchC = torch::nn_sequential(
      torch::nn_avg_pool2d(
        kernel_size = c(1L,3L)
        , stride = 1L
        , padding = 0L
      ),
      convBlock(
        in_channels = in_channels
        , out_channels = 4L*channel_mult
        , kernel_size = 1L
        , stride = 1L
        , dropout = dropout
        , padding = c(0L,1L)
      )
    )

    self$branchD = convBlock(
      in_channels = in_channels
      , out_channels = 4L*channel_mult
      , kernel_size = 1L
      , stride = 1L
      , padding = 0L
      , dropout = dropout
    )

  },
  forward = function(x){
    branchARes = self$branchA(x)
    branchBRes = self$branchB(x)
    branchCRes = self$branchC(x)
    branchDRes = self$branchD(x)
    # print(branchARes$size())
    # print(branchBRes$size())
    # print(branchCRes$size())
    # print(branchDRes$size())
    res = torch::torch_cat(list(branchARes, branchBRes, branchCRes, branchDRes),2L)
    res
  }
)
inceptionBlock <- function(type, channel_mult, dropout){
  layer <- list(channel_mult=channel_mult, type=type, dropout = dropout)
  class(layer) <- c("inceptionBlock", "citolayer")
  return(layer)
}
build_cnn<-function (input_shape, output_shape, architecture)
{
      input_dim <- length(input_shape) - 1
    net_layers = list()
    counter <- 1
    flattened <- FALSE
    transfer <- FALSE
    for (layer in architecture) {
        if (inherits(layer, "transfer")) {
            if (!(input_dim == 2)) 
                stop("The pretrained models only work on images: [n, channels, x, y]")
            transfer_model <- get_pretrained_model(layer$name, 
                layer$pretrained)
            if (layer$freeze) 
                transfer_model <- freeze_weights(transfer_model)
            if (input_shape[1] != 3) {
                transfer_model <- tryCatch({
                  cur <- transfer_model
                  call_string <- "transfer_model"
                  while (length(cur$children) > 0) {
                    cur_name <- names(cur$children)[1]
                    call_string <- paste0(call_string, "[['", 
                      cur_name, "']]")
                    cur <- eval(rlang::parse_expr(call_string))
                  }
                  first_layer <- torch::nn_conv2d(in_channels = as.integer(input_shape[1]), 
                    out_channels = as.integer(cur$out_channels), 
                    kernel_size = as.integer(cur$kernel_size), 
                    stride = as.integer(cur$stride), padding = as.integer(cur$padding), 
                    dilation = as.integer(cur$dilation), groups = as.integer(cur$groups), 
                    bias = !is.null(cur$bias), padding_mode = cur$padding_mode)
                  call_string <- paste0(call_string, "<-first_layer")
                  eval(rlang::parse_expr(call_string))
                  transfer_model
                }, error = function(x) stop(paste0("automatic input layer adjustment to ", 
                  input_shape[1], " layers failed with error message:\n", 
                  x)))
            }
            if (!layer$replace_classifier) {
                transfer_model <- replace_output_layer(transfer_model, 
                  output_shape)
                if (layer$name == "inception_v3") {
                  n1 <- transfer_model[[1]]
                  n2 <- transfer_model[[2]]
                  n3 <- transfer_model[[3]]
                  n4 <- transfer_model[[4]]
                  n5 <- transfer_model[[5]]
                  n6 <- transfer_model[[6]]
                  n7 <- transfer_model[[7]]
                  n8 <- transfer_model[[8]]
                  n9 <- transfer_model[[9]]
                  n10 <- transfer_model[[10]]
                  n11 <- transfer_model[[11]]
                  n12 <- transfer_model[[12]]
                  n13 <- transfer_model[[13]]
                  n14 <- transfer_model[[14]]
                  n15 <- transfer_model[[15]]
                  n16 <- transfer_model[[16]]
                  n17 <- transfer_model[[17]]
                  n18 <- transfer_model[[18]]
                  n19 <- transfer_model[[19]]
                  n20 <- transfer_model[[20]]
                  n21 <- transfer_model[[21]]
                  n21 <- torch::nn_sequential(torch::nn_flatten(), 
                    n21)
                  transfer_model <- torch::nn_module(initialize = function(n1, 
                    n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, 
                    n12, n13, n14, n15, n16, n17, n18, n19, n20, 
                    n21) {
                    self$n1 = n1
                    self$n2 = n2
                    self$n3 = n3
                    self$n4 = n4
                    self$n5 = n5
                    self$n6 = n6
                    self$n7 = n7
                    self$n8 = n8
                    self$n9 = n9
                    self$n10 = n10
                    self$n11 = n11
                    self$n12 = n12
                    self$n13 = n13
                    self$n14 = n14
                    self$n15 = n15
                    self$n16 = n16
                    self$n17 = n17
                    self$n18 = n18
                    self$n19 = n19
                    self$n20 = n20
                    self$n21 = n21
                  }, forward = function(x) {
                    self$n21(self$n20(self$n19(self$n18(self$n17(self$n16(self$n15(self$n14(self$n13(self$n12(self$n11(self$n10(self$n9(self$n8(self$n7(self$n6(self$n5(self$n4(self$n3(self$n2(self$n1(x)))))))))))))))))))))
                  })
                  transfer_model <- transfer_model(n1, n2, n3, 
                    n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, 
                    n14, n15, n16, n17, n18, n19, n20, n21)
                  purrr::walk(transfer_model$parameters, function(param) param$requires_grad_(FALSE))
                  purrr::walk(n1$parameters, function(param) param$requires_grad_(TRUE))
                  purrr::walk(n21$parameters, function(param) param$requires_grad_(TRUE))
                  transfer_model[["n1"]] <- n1
                  transfer_model[["n21"]] <- n21
                }
                return(transfer_model)
            }
            transfer <- TRUE
            input_shape <- get_transfer_output_shape(layer$name)
        }
        else if (inherits(layer, "inceptionBlock")) {
            if (layer$type == "2D") {
                net_layers[[counter]] <- inceptionBlock_A_2D(input_shape[1], 
                  layer$channel_mult, layer$dropout)
                input_shape[1] <- 18L * layer$channel_mult
                counter <- counter + 1
            }
            else if (layer$type == "red") {
                net_layers[[counter]] <- inceptionBlock_A_2D_reduction(input_shape[1], 
                  layer$channel_mult, layer$dropout)
                input_shape <- c(18L * layer$channel_mult, input_shape[2:3] - 
                  c(2, 0))
                counter <- counter + 1
            }
            else if (layer$type == "1D") {
                net_layers[[counter]] <- inceptionBlock_A_1D(input_shape[1], 
                  layer$channel_mult, layer$dropout)
                input_shape[1] <- 18L * layer$channel_mult
                counter <- counter + 1
            }
            else if (layer$type == "red100") {
                net_layers[[counter]] <- inceptionBlock_A_100_reduction(input_shape[1], 
                  layer$channel_mult, layer$dropout)
                input_shape <- c(18L * layer$channel_mult, input_shape[2:3] - 
                  c(2, 99))
                counter <- counter + 1
            }
        }
        else if (inherits(layer, "conv")) {
            net_layers[[counter]] <- switch(input_dim, torch::nn_conv1d(input_shape[1], 
                layer[["n_kernels"]], layer[["kernel_size"]], 
                padding = layer[["padding"]], stride = layer[["stride"]], 
                dilation = layer[["dilation"]], bias = layer[["bias"]]), 
                torch::nn_conv2d(input_shape[1], layer[["n_kernels"]], 
                  layer[["kernel_size"]], padding = layer[["padding"]], 
                  stride = layer[["stride"]], dilation = layer[["dilation"]], 
                  bias = layer[["bias"]]), torch::nn_conv3d(input_shape[1], 
                  layer[["n_kernels"]], layer[["kernel_size"]], 
                  padding = layer[["padding"]], stride = layer[["stride"]], 
                  dilation = layer[["dilation"]], bias = layer[["bias"]]))
            counter <- counter + 1
            input_shape <- get_output_shape(input_shape = input_shape, 
                n_kernels = layer[["n_kernels"]], kernel_size = layer[["kernel_size"]], 
                stride = layer[["stride"]], padding = layer[["padding"]], 
                dilation = layer[["dilation"]])
            if (layer[["normalization"]]) {
                net_layers[[counter]] <- switch(input_dim, torch::nn_batch_norm1d(input_shape[1]), 
                  torch::nn_batch_norm2d(input_shape[1]), torch::nn_batch_norm3d(input_shape[1]))
                counter <- counter + 1
            }
            net_layers[[counter]] <- get_activation_layer(layer[["activation"]])
            counter <- counter + 1
            if (layer[["dropout"]] > 0) {
                net_layers[[counter]] <- switch(input_dim, torch::nn_dropout(layer[["dropout"]]), 
                  torch::nn_dropout2d(layer[["dropout"]]), torch::nn_dropout3d(layer[["dropout"]]))
                counter <- counter + 1
            }
        }
        else if (inherits(layer, "maxPool")) {
            net_layers[[counter]] <- switch(input_dim, torch::nn_max_pool1d(layer[["kernel_size"]], 
                padding = layer[["padding"]], stride = layer[["stride"]], 
                dilation = layer[["dilation"]]), torch::nn_max_pool2d(layer[["kernel_size"]], 
                padding = layer[["padding"]], stride = layer[["stride"]], 
                dilation = layer[["dilation"]]), torch::nn_max_pool3d(layer[["kernel_size"]], 
                padding = layer[["padding"]], stride = layer[["stride"]], 
                dilation = layer[["dilation"]]))
            counter <- counter + 1
            input_shape <- get_output_shape(input_shape = input_shape, 
                n_kernels = input_shape[1], kernel_size = layer[["kernel_size"]], 
                stride = layer[["stride"]], padding = layer[["padding"]], 
                dilation = layer[["dilation"]])
        }
        else if (inherits(layer, "avgPool")) {
            net_layers[[counter]] <- switch(input_dim, torch::nn_avg_pool1d(layer[["kernel_size"]], 
                padding = layer[["padding"]], stride = layer[["stride"]]), 
                torch::nn_avg_pool2d(layer[["kernel_size"]], 
                  padding = layer[["padding"]], stride = layer[["stride"]]), 
                torch::nn_avg_pool3d(layer[["kernel_size"]], 
                  padding = layer[["padding"]], stride = layer[["stride"]]))
            counter <- counter + 1
            input_shape <- get_output_shape(input_shape = input_shape, 
                n_kernels = input_shape[1], kernel_size = layer[["kernel_size"]], 
                stride = layer[["stride"]], padding = layer[["padding"]], 
                dilation = rep(1, input_dim))
        }
        else if (inherits(layer, "linear")) {
            if (!flattened) {
                net_layers[[counter]] <- torch::nn_flatten()
                counter <- counter + 1
                input_shape <- prod(input_shape)
                flattened <- T
            }
            net_layers[[counter]] <- torch::nn_linear(in_features = input_shape, 
                out_features = layer[["n_neurons"]], bias = layer[["bias"]])
            input_shape <- layer[["n_neurons"]]
            counter <- counter + 1
            if (layer[["normalization"]]) {
                net_layers[[counter]] <- torch::nn_batch_norm1d(layer[["n_neurons"]])
                counter <- counter + 1
            }
            net_layers[[counter]] <- cito:::get_activation_layer(layer[["activation"]])
            counter <- counter + 1
            if (layer[["dropout"]] > 0) {
                net_layers[[counter]] <- torch::nn_dropout(layer[["dropout"]])
                counter <- counter + 1
            }
        }
    }
    if (!flattened) {
        net_layers[[counter]] <- torch::nn_flatten()
        counter <- counter + 1
        input_shape <- prod(input_shape)
    }
    if (!is.null(output_shape)) 
        net_layers[[counter]] <- torch::nn_linear(in_features = input_shape, 
            out_features = output_shape)
    net <- do.call(torch::nn_sequential, net_layers)
    if (transfer) {
        net <- replace_classifier(transfer_model, net)
    }
    return(net)
}
