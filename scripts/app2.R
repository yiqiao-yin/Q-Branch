#### BEGIN SCRIPT ####

# Library
library(quantmod)
library(dygraphs)
library(DT)
library(plotly)
library(keras)

#### DEFINE FUNCTIONS ####

# Def
KerasNST <- function(
    X = X,
    y = y,
    cutoff = 0.8,
    validation_split = 1 - cutoff,
    max_len = 1,
    useModel = "lstm",
    num_hidden = 2,
    l1.units = 2,
    l2.units = 4,
    l3.units = 6,
    activation = 'tanh',
    loss = 'loss_mean_squared_error',
    useDice = TRUE,
    optimizer = optimizer_rmsprop(),
    batch_size = 128,
    epochs = 10,
    verbatim = TRUE
) {
  
  # Check shapes
  # this ensures that max_len can be divided by number of columns
  # in the explanatory data matrix
  print("First, check if max_len can be divided by number of cols.")
  if (ncol(X) %% max_len == 0) {
    print("... checking ...")
    print(ncol(X) %% max_len == 0)
    print("... it means it is divisible, pass and continue ...")
  } else {
    print("Warning: number of col in X cannot divide max_len!")
    print("Reset to 1.")
    max_len = 1
  } # Done
  
  # Package
  library(keras)
  
  # Separate scenarios:
  # if X is NULL, this is not allowed
  # if X is filled as a data frame (we only take data frame),
  # then we pursue X (this input data frame) as covariate matrix
  # and in this case the vector y (assuming it has the same length as X)
  # will be response variable
  # training data
  train_idx = 1:round(cutoff*nrow(y),0)
  x_train <- array(X[train_idx,], dim = c(length(train_idx), max_len, round(ncol(X)/max_len)))
  y_train <- array(y[train_idx,], dim = c(length(train_idx), ncol(y)))
  
  # testing data
  x_test <- array(X[-train_idx,], dim = c(nrow(y) - length(train_idx), max_len, round(ncol(X)/max_len)))
  y_test <- array(y[-train_idx,], dim = c(nrow(y) - length(train_idx), ncol(y)))
  
  # shape
  dim(x_train); dim(x_test); dim(y_train); dim(y_test)
  
  # Defining the Model
  if (tolower(useModel) == "lstm") {
    if (num_hidden == 1) {
      model <- keras_model_sequential() %>%
        layer_lstm(l1.units, input_shape = c(max_len, round(ncol(X)/max_len))) %>%
        layer_dense(ncol(y)) %>% layer_activation(activation)
      summary(model)
    } else if (num_hidden == 2) {
      model <- keras_model_sequential() %>%
        layer_lstm(l1.units, input_shape = c(max_len, round(ncol(X)/max_len))) %>%
        layer_dense(units = l2.units, activation = activation, use_bias = FALSE) %>%
        layer_dense(ncol(y)) %>% layer_activation(activation)
      summary(model)
    } else if (num_hidden == 3) {
      model <- keras_model_sequential() %>%
        layer_lstm(l1.units, input_shape = c(max_len, round(ncol(X)/max_len))) %>%
        layer_dense(units = l2.units, activation = activation, use_bias = FALSE) %>%
        layer_dense(units = l3.units, activation = activation, use_bias = FALSE) %>%
        layer_dense(ncol(y)) %>% layer_activation(activation)
      summary(model)
    } else {
      print("Too many layers implemented, set to default: one hidden layer")
      model <- keras_model_sequential() %>%
        layer_lstm(l1.units, input_shape = c(max_len, round(ncol(X)/max_len))) %>%
        layer_dense(ncol(y)) %>% layer_activation(activation)
      summary(model)
    }
  } else if (tolower(useModel) == "gru") {
    if (num_hidden == 1) {
      model <- keras_model_sequential() %>%
        layer_gru(units = l1.units, return_sequences = TRUE, input_shape = dim(x_train)[-1]) %>%
        bidirectional(layer_gru(units = l1.units)) %>%
        layer_dense(units = ncol(y_test), activation = activation)
      summary(model)
    } else if (num_hidden == 2) {
      model <- keras_model_sequential() %>%
        layer_gru(units = l1.units, return_sequences = TRUE,input_shape = dim(x_train)[-1]) %>%
        bidirectional(layer_gru(units = l1.units)) %>%
        layer_dense(units = l2.units, activation = activation) %>%
        layer_dense(units = ncol(y_test), activation = activation)
      summary(model)
    } else if (num_hidden == 3) {
      model <- keras_model_sequential() %>%
        layer_gru(units = l1.units, return_sequences = TRUE,input_shape = dim(x_train)[-1]) %>%
        bidirectional(layer_gru(units = l1.units)) %>%
        layer_dense(units = l2.units, activation = activation) %>%
        layer_dense(units = l3.units, activation = activation) %>%
        layer_dense(units = ncol(y_test), activation = activation)
      summary(model)
    }
  }
  
  # Dice Loss
  # Wiki: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
  dice <- custom_metric("dice", function(y_true, y_pred, smooth = 1.0) {
    y_true_f <- k_flatten(y_true)
    y_pred_f <- k_flatten(y_pred)
    intersection <- k_sum(y_true_f * y_pred_f)
    (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
  })
  
  # Compile and Train
  # compile the model with appropriate loss function, optimizer, and metrics:
  if (useDice) {
    model %>% compile(
      loss = loss,
      optimizer = optimizer,
      metrics = dice )
  } else {
    model %>% compile(
      loss = loss,
      optimizer = optimizer )
  } # done
  history = model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    validation_split = validation_split,
    epochs = epochs ); if (verbatim) {plot(history)}
  
  # Prediction
  yhat_train_mat = predict(model, x_train)
  yhat_test_mat = predict(model, x_test)
  if (verbatim) {
    print("Note:")
    print("-- Use yhat_test_mat = predict(model, x_test) to make prediction")
    print(paste0("-- This assumes x_test has dimension: ", paste0(dim(x_test)[-1], collapse = " x "))) }
  head(yhat_test_mat)
  if (verbatim) {
    par(mfrow=c(1,2))
    matplot(
      y_test,
      type = 'l',
      xaxs = "i",
      yaxs = "i" ,
      lwd = 3,
      xlab = "Sequential Index (Time or Day)",
      ylab = "Real Values",
      main = paste0("Real Data: Y Matrix"))
    eachLoss = sapply(1:ncol(yhat_test_mat), function(s) mean(abs(y_test[,s] - yhat_test_mat[,s]), na.rm = TRUE))
    matplot(
      yhat_test_mat,
      type = 'l',
      xaxs = "i",
      yaxs = "i" ,
      lwd = 3,
      xlab = "Sequential Index (Time or Day)",
      ylab = "Predicted Values",
      main = paste0("Prediction: YHat Matrix \n(average of the MAEs for all paths: ", mean(round(eachLoss,3), na.rm = TRUE), ")"))
  } # Done
  
  # Return
  return(
    list(
      Model = list(model = model, weights = model$weights),
      X = X, # original explanatory data matrix
      y = y, # original response data matrix
      x_train = x_train,
      y_train = y_train,
      x_test = x_test,
      y_test = y_test,
      yhat_train_mat = yhat_train_mat,
      yhat_test_mat = yhat_test_mat
    )
  )
}

# Def
Seq2Seq <- function(
    target = "AAPL",
    past_N_days = 50,
    seed = 2021,
    num_hidden = 3,
    l1.units = 256,
    l2.units = 128,
    l3.units = 32,
    epoch = 30,
    rescale = TRUE
)   {
  # Get Data
  symbols = target
  myList <- list()
  myList <- lapply(symbols, function(x) {quantmod::getSymbols(x, from = as.Date("1990-05-18"),  auto.assign = FALSE)})
  names(myList) <- symbols
  myList[[1]] = myList[[1]][,-5]
  
  # Process Data
  if (rescale) {
    X = as.matrix(as.data.frame(myList[[1]]))[-nrow(myList[[1]]), ]
    X = apply(X, 2, function(c) (c - min(c))/(max(c)-min(c)))
    y = as.matrix(as.data.frame(myList[[1]]))[-1, ]
  } else {
    X = as.matrix(as.data.frame(myList[[1]]))[-nrow(myList[[1]]), ]
    y = as.matrix(as.data.frame(myList[[1]]))[-1, ]
  }; cutoff = 0.9
  
  # Model Fitting
  set.seed(seed)
  tmp = KerasNST(
    X = X,
    y = y,
    cutoff = cutoff,
    validation_split = 1 - cutoff,
    max_len = 1,
    useModel = "lstm",
    num_hidden = num_hidden,
    l1.units = l1.units,
    l2.units = l2.units,
    l3.units = l3.units,
    activation = 'relu',
    loss = 'categorical_crossentropy',
    useDice = TRUE,
    optimizer = optimizer_rmsprop(1e-4),
    batch_size = 128,
    epochs = epoch,
    verbatim = TRUE )
  
  # Plot
  y_test = as.ts(tmp$y_test)
  yhat_test_mat = as.ts(tmp$yhat_test_mat)
  testL = (nrow(myList[[1]]) - nrow(as.ts(tmp$y_test)) + 1):nrow(myList[[1]])
  rownames(y_test) <- rownames(as.data.frame(myList[[1]]))[testL]
  colnames(y_test) = colnames(as.data.frame(myList[[1]]))
  rownames(yhat_test_mat) = rownames(as.data.frame(myList[[1]]))[testL]
  colnames(yhat_test_mat) = colnames(as.data.frame(myList[[1]]))
  shortL = (nrow(y_test)-past_N_days):(nrow(y_test))
  realPlt <- dygraph(y_test[shortL,-5]) %>% dyCandlestick()
  forecastPlt <- dygraph(yhat_test_mat[shortL,-5]) %>% dyCandlestick()
  
  # Output
  return(list(
    realPlt = realPlt,
    forecastPlt = forecastPlt
  ))
} # End of function


# Def
BasicMM <- function(
    target = "AAPL",
    start_date = "1990-01-01",
    past_N_days = 10
) {
  
  # Get Data
  myList <- list()
  myList <- lapply(target, function(x) {quantmod::getSymbols(x, from = as.Date(start_date), auto.assign = FALSE)})
  names(myList) <- target
  target <- myList[[1]]
  
  # Data
  price <- target[, 4] # closing
  
  # Return Data
  returnDataDaily <- quantmod::dailyReturn(price)
  returnDataWeekly <- quantmod::weeklyReturn(price)
  returnDataMonthly <- quantmod::monthlyReturn(price)
  
  # Report
  statTable <- data.frame(rbind(
    DailyData = round(c("Ave" = mean(returnDataDaily), "SD" = sd(returnDataDaily),
                        "SharpeRatio" = mean(returnDataDaily)/sd(returnDataDaily)), 4),
    WeeklyData = round(c(mean(returnDataWeekly), sd(returnDataWeekly),
                         mean(returnDataWeekly)/sd(returnDataWeekly)), 4),
    MonthlyData = round(c(mean(returnDataMonthly), sd(returnDataMonthly),
                          mean(returnDataMonthly)/sd(returnDataMonthly)), 4)
  ))
  
  # Recent Report
  statTableRecent <- data.frame(cbind(
    RecentAve = round(mean(tail(returnDataDaily, past_N_days)), 4),
    RecentSD = round(sd(tail(returnDataDaily, past_N_days)), 4),
    RecentSharpeRatio = round(
      mean(tail(returnDataDaily, past_N_days))/sd(tail(returnDataDaily, past_N_days)), 4)
  ))
  
  # Plot
  dyPlt <- dygraph(target[, c(1:4)]) %>% dyCandlestick()
  
  # Output
  return(list(
    returnDataDaily = returnDataDaily,
    returnDataWeekly = returnDataWeekly,
    returnDataMonthly = returnDataMonthly,
    Statistics = statTable,
    RecentStats = statTableRecent,
    Plot = dyPlt
  ))
} # End of function

# Def
AdvBuySellAlgorithm <- function(
    target = "AAPL",
    start_date = "1990-01-04",
    buyCoef = -10,
    sellCoef = +10,
    height = 30,
    movingAvEnv = c(10, 20, 50, 70, 100)
) {
  
  # Data
  # target = "AAPL"
  TICKERNAME = target
  myList <- list()
  myList <- lapply(target, function(x) {quantmod::getSymbols(x, from = as.Date(start_date), auto.assign = FALSE)})
  names(myList) <- target
  target <- myList[[1]]
  HIST <- hist(
    quantmod::dailyReturn(target), breaks = 30,
    xlab = "Value", main = "Histogram of Target Stock")
  
  # Create Distance Data
  target = na.omit(target)
  smaData <- cbind(target[, 1:4])
  Nom <- paste0("Lag=", movingAvEnv)
  for (i in movingAvEnv) {smaData <- cbind(smaData, SMA(target[, 4], n = i))}
  smaData <- na.omit(smaData)
  distData <- smaData[, 4]
  for (i in 5:ncol(smaData)) {distData <- cbind(distData, smaData[, 1] - smaData[, i])}
  p1 <- matplot(distData, type = "l",
                xlab = "Time Stamp", ylab = "Value",
                main = "Distance Data for Target Stock"); abline(
                  h = max(distData[, 6]), lty = 3, col = "green"); abline(
                    h = min(distData[, 6]), lty = 3, col = "red")
  
  # Buy Signal
  buyData <- distData[, 1]
  for (i in 2:ncol(distData)) {
    buyData <- cbind(
      buyData,
      as.numeric(distData[, i] < buyCoef * sd(distData[, i])))
  }
  colnames(buyData) <- c("Close", Nom)
  
  # Sell Signal
  sellData <- distData[, 1]
  for (i in 2:ncol(distData)) {
    sellData <- cbind(
      sellData, 
      as.numeric(distData[, i] > sellCoef * sd(distData[, i])))
  }
  colnames(sellData) <- c("Close", Nom)
  
  # All Signal
  buysellTable <- cbind(smaData[, 1:4], distData[, 1], rowMeans(buyData[, -1]), rowMeans(sellData[, -1]))
  buysellTable[, -c(1:5)] <- buysellTable[, -c(1:5)] * height
  colnames(buysellTable) <- c(colnames(smaData)[1:4], "TargetClosingPrice", "BuySignal", "SellSignal")
  buysellTable = buysellTable[,-5] # get rid of duplicated column: closing price
  #p2 <- matplot(buysellTable, type = "l", xlab = "Time Stamp", ylab = "Value + Signal", main = "Path with Buy/Sell Signals")
  
  # GGplot
  #library(ggplot2)
  #library(reshape2)
  #ts_buysellTable <- data.frame(cbind(rownames(data.frame(buysellTable)), data.frame(buysellTable)))
  #colnames(ts_buysellTable) <- c("Date", "TargetClosingPrice", "BuySignal", "SellSignal")
  #melt_ts_buysellTable <- melt(ts_buysellTable, id = "Date")
  #p2 <- ggplot(melt_ts_buysellTable, aes(x = Date, y = value, colour = variable, group = variable)) + geom_line()
  
  # Statistics
  signalStatistics <- data.frame(
    Ave = round(apply(buysellTable[, -1], 2, function(i) mean(as.numeric(i > 0))), 4),
    SD = round(apply(buysellTable[, -1], 2, function(i) sd(as.numeric(i > 0))), 4),
    Var = round(apply(buysellTable[, -1], 2, function(i) sd(as.numeric(i > 0))^2), 4),
    Max = apply(buysellTable[, -1], 2, max),
    EX_to_3 = round(apply(buysellTable[, -1], 2, function(COL) {mean(COL)/3}), 4),
    VX_to_3 = round(apply(buysellTable[, -1], 2, function(i) sd(as.numeric(i > 0))^2/3), 4),
    EX_to_6 = round(apply(buysellTable[, -1], 2, function(COL) {mean(COL)/6}), 4), 
    VX_to_6 = round(apply(buysellTable[, -1], 2, function(i) sd(as.numeric(i > 0))^2/6), 4),
    EX_to_9 = round(apply(buysellTable[, -1], 2, function(COL) {mean(COL)/9}), 4),
    VX_to_9 = round(apply(buysellTable[, -1], 2, function(i) sd(as.numeric(i > 0))^2/9), 4),
    EX_to_12 = round(apply(buysellTable[, -1], 2, function(COL) {mean(COL)/12}), 4),
    VX_to_12 = round(apply(buysellTable[, -1], 2, function(i) sd(as.numeric(i > 0))^2/12), 4)  )
  signalStatistics <- signalStatistics[-c(1:3), ]
  
  # Dygraph
  p2 <- dygraphs::dygraph(
    buysellTable[, 1:4],
    main = paste("AI Recommended Buy/Sell Signal for Ticker: ", TICKERNAME)) %>% 
    dyCandlestick() %>%
    dyLegend(show = "follow")
  for (i in 1:nrow(buysellTable)) {
    if (buysellTable$BuySignal[i] > 0) {
      p2 <- p2 %>%
        dyAnnotation(
          index(buysellTable)[i], text = paste0("Buy:", buysellTable$BuySignal[i]),
          width = 60, height = 20,
          tooltip = paste("Buy:", buysellTable$BuySignal[i]))
    }
    if (buysellTable$SellSignal[i] > 0) {
      p2 <- p2 %>% dyAnnotation(
        index(buysellTable)[i], text = paste("Sell:", buysellTable$SellSignal[i]),
        width = 60, height = 20,
        tooltip = paste("Sell:", buysellTable$SellSignal[i]))
    }
  }
  
  return(list(
    Histogram = HIST,
    DistancePlot = p1,
    DistanceData = distData,
    buyData = buyData,
    sellData = sellData,
    buysellTable = buysellTable,
    Statistics = signalStatistics,
    FinalPlot = p2
  ))
} # end of function

# Def
QuantGrowthStrategy <- function(
    symbolsIDX = c("SPY"),
    symbolsTECH = c("AAPL", "FB", "NVDA"),
    symbolsFINANCIAL = c("BAC", "GS", "JPM"),
    symbolsINDUSTRIAL = c("LMT", "GD", "BA"),
    symbolsCONSUMER = c("WMT", "TGT", "PEP"),
    symbolsSERVICE = c("AMZN", "DIS", "BABA"),
    pastNforGrowth = 20,
    topGrowthNum = 4,
    intializedValue = 1,
    howMuchtoInvest = 1e4
) {
  
  # ASSET PRICING, GROWTH STRATEGY
  # Carhart (1995)
  # Source: https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1540-6261.1997.tb03808.x
  # Original idea @ Carhart (1995) PhD Dissertation
  
  # Package
  library(quantmod)
  
  # Initialize: an empty space for data
  symbols <- c(symbolsIDX, symbolsTECH, symbolsFINANCIAL, symbolsINDUSTRIAL, symbolsCONSUMER, symbolsSERVICE)
  myList <- list()
  myList <- lapply(symbols, function(x) {getSymbols(x, from = as.Date("1990-01-04"),  auto.assign = FALSE)})
  names(myList) <- symbols
  
  #################### Visualization ####################
  
  # Clean Data
  AnnRet <- c()
  AnnRet <- sapply(1:length(symbols), function(i) AnnRet <- c(
    AnnRet,
    round(mean(quantmod::yearlyReturn(myList[[i]])), 5) ))
  QuartRet <- c()
  QuartRet <- sapply(1:length(symbols), function(i) QuartRet <- c(
    QuartRet,
    round(mean(quantmod::quarterlyReturn(myList[[i]])), 5) ))
  WeekRet <- c()
  WeekRet <- sapply(1:length(symbols), function(i) WeekRet <- c(
    WeekRet,
    round(mean(quantmod::weeklyReturn(myList[[i]])), 5) ))
  DayRet <- c()
  DayRet <- sapply(1:length(symbols), function(i) DayRet <- c(
    DayRet,
    round(mean(quantmod::weeklyReturn(myList[[i]])), 5) ))
  past5Vol <- c()
  past5Vol <- sapply(1:length(symbols), function(i) past5Vol <- c(
    past5Vol,
    mean(tail(myList[[i]][, 5], 5)) ))
  data_for_viz <- cbind(symbols, AnnRet, QuartRet, WeekRet, DayRet, past5Vol)
  data_for_viz <- cbind(
    data_for_viz,
    AnnRetLev = ifelse(
      AnnRet > mean(AnnRet) + sd(AnnRet), 1,
      ifelse(AnnRet > mean(AnnRet), 2,
             ifelse(AnnRet > mean(AnnRet) - sd(AnnRet), 3, 4))))
  data_for_viz <- cbind(
    data_for_viz,
    QuartRetLev = ifelse(
      QuartRet > mean(QuartRet) + sd(QuartRet), 1, 
      ifelse(QuartRet > mean(QuartRet), 2,
             ifelse(QuartRet > mean(QuartRet) - sd(QuartRet), 3, 4))))
  data_for_viz <- cbind(
    data_for_viz,
    sector = c(rep("Mkt", length(symbolsIDX)), 
               rep("Tech", length(symbolsTECH)),
               rep("Finance", length(symbolsFINANCIAL)), 
               rep("Industrial", length(symbolsINDUSTRIAL)),
               rep("Consumer", length(symbolsCONSUMER)),
               rep("Service", length(symbolsSERVICE))) )
  
  # Visualization
  library(plotly)
  dtaViz <- data.frame(data_for_viz)
  slope <- 2e-5*0.65
  dtaViz$AdjChipsExch <- as.numeric(as.character(dtaViz$past5Vol)) * (as.numeric(as.character(dtaViz$WeekRet)) + 1)
  dtaViz$size <- sqrt(as.numeric(as.character(dtaViz$AdjChipsExch)) * slope)
  colors <- c('#4AC6B7', '#1972A4', '#965F8A', '#FF7070', '#C61951')
  PLT1 <- subplot(
    plot_ly(x = as.numeric(dtaViz$AnnRet), type = "histogram", name = "Annual Return"),
    plotly_empty(),
    plot_ly(
      dtaViz, x = ~DayRet, y = ~AnnRet, color = ~sector,
      size = ~size, colors = colors,
      type = 'scatter', mode = 'markers',
      contours = list(coloring='heatmap'),
      sizes = c(min(dtaViz$size), max(dtaViz$size)),
      marker = list(symbol = 'circle', sizemode = 'diameter'),
      text = ~paste(
        'symbols:', symbols,
        '<br>Sector:', sector,
        '<br>Past 5D Ave Vol:', past5Vol,
        '<br>Adj. Chips Exchanged:', AdjChipsExch,
        '<br>Ave. Daily Returns:', DayRet,
        '<br>Ave. Weekly Returns:', WeekRet,
        '<br>Ave. Quarterly Returns:', QuartRet)) %>%
      layout(
        title = 'Data: Daily Return v. Annual Return \n(size: ave. of past 10 days volume; color: sector)',
        xaxis = list(title = 'Daily Returns', gridcolor = 'rgb(0, 204, 0)'),
        yaxis = list(title = 'Annual Returns', gridcolor = 'rgb(0, 204, 0)') ),
    plot_ly(y = as.numeric(dtaViz$DayRet), type = "histogram", name = "Daily Return"),
    nrows = 2, heights = c(0.2, 0.8), widths = c(0.8, 0.2)
  )
  
  # Basic Candlestick Grid Plot
  which_stock = 1; length(myList)
  plotList <- function() {
    lapply(
      1:length(myList),
      function(which_stock) {
        stockData = myList[[which_stock]]
        df <- data.frame(Date=index(stockData), coredata(stockData))
        df <- tail(df, 30)
        names(df) <- c("Date", "Open", "High", "Low", "Close", "Volume", "Adjusted")
        df %>% plot_ly(
          x = ~Date, type="candlestick",
          open = ~Open, close = ~Close,
          high = ~High, low = ~Low,
          name = symbols[which_stock],
          text = ~paste(
            'symbols:', symbols[which_stock], 
            '<br>Sector:', as.character(dtaViz[dtaViz$symbols == symbols[which_stock], "sector"])[1],
            '<br>Volume:', as.numeric(tail(df, 1)[, 6]),
            '<br>Ave. Daily Returns:', as.numeric(as.character(dtaViz[dtaViz$symbols == symbols[which_stock], "DayRet"])), 
            '<br>Ave. Weekly Returns:', as.numeric(as.character(dtaViz[dtaViz$symbols == symbols[which_stock], "WeekRet"])),
            '<br>Ave. Quarterly Returns:', as.numeric(as.character(dtaViz[dtaViz$symbols == symbols[which_stock], "QuartRet"])))) %>% 
          layout(xaxis = list(rangeslider = list(visible = F)))
      }
    )
  }
  
  # Plot
  if (sqrt(length(myList)) %% 1 == 0) { gridSIZE <- round(sqrt(length(myList)))
  } else { gridSIZE <- round(sqrt(length(myList))) + 1 }
  PLT2 <- subplot(plotList(), nrows = gridSIZE); PLT2
  
  ################# End Visualization ###################
  
  # Return Data
  returnData <- cbind()
  lastPeriodReturnData <- cbind()
  for (i in 1:length(symbols)) {
    returnData <- cbind(
      returnData,
      quantmod::dailyReturn(as.xts(data.frame(myList[i]))))
    lastPeriodReturnData <- cbind(
      lastPeriodReturnData,
      quantmod::dailyReturn(as.xts(data.frame(myList[i])))/lag(
        quantmod::dailyReturn(as.xts(data.frame(myList[i]))), pastNforGrowth) - 1 )
  } # end of loop
  returnData <- data.frame(na.omit(returnData)); names(returnData) <- symbols
  lastPeriodReturnData <- data.frame(na.omit(lastPeriodReturnData)); names(lastPeriodReturnData) <- symbols
  
  # Check
  n1 <- nrow(returnData); n2 <- nrow(lastPeriodReturnData)
  if (n1 > n2) {returnData <- returnData[-c(1:(n1-n2)), ]}
  dim(returnData); dim(lastPeriodReturnData)
  
  ################# Growth Strategy ###################
  
  # Create Growth Strategy
  # Dynamic Programming:
  #      start with a for loop and implement policy to concatenate different values
  growthStrategy <- c()
  selectedStockIndex <- c()
  listNom <- rbind()
  for (i in 1:nrow(returnData)) {
    if (i < pastNforGrowth) {
      growthStrategy <- c(growthStrategy, mean(as.numeric(as.character(returnData[i, ]))))
    } else if (i %% pastNforGrowth == 0) {
      Nom <- names(sort(lastPeriodReturnData[i, ], decreasing = TRUE)[1:topGrowthNum])
      listNom <- rbind(listNom, Nom)
      selectedStockIndex <- sort(sapply(1:length(Nom), function(j) {which(symbols == Nom[j])}))
      growthStrategy <- c(
        growthStrategy,
        mean(as.numeric(as.character(returnData[i, selectedStockIndex]))))
    } else {
      growthStrategy <- c(
        growthStrategy,
        mean(as.numeric(as.character(returnData[i, selectedStockIndex]))))
    }
  } # end of loop
  
  # OLS Model
  newData <- data.frame(Strategy = growthStrategy, MKT = returnData$SPY)
  linearModel <- lm(Strategy~MKT, newData)
  summary(linearModel)
  
  ######################## Helper Function ####################################
  
  ## Calculated bivariate normal density at /one/ point (x,y).
  ## Must pass full covariance matrix (Sig), or sd1, sd2 and rho.
  ## See  below.
  fxy = function(x, y, mu, Sig, sd1, sd2, rho) {
    
    if(missing(mu)) mu=c(0,0)
    
    if(!missing(Sig)) {
      sd1 = sqrt(Sig[1,1])
      sd2 = sqrt(Sig[2,2])
      if(Sig[1,2] != Sig[2,1]) {
        print("Covariance matrix is not symmetric... Returning .")
        return(NULL)
      }
      rho = Sig[1,2]/(sd1*sd2)
    }
    else if(missing(rho) || missing(sd1) || missing(sd2)) {
      sd1 = sd2 = 1
      rho = 0
    }
    
    Q = (x-mu[1])^2/sd1^2 + (y-mu[2])^2/sd2^2 -
      2*rho*(x-mu[1])*(y-mu[2])/(sd1*sd2)
    
    1/(2*pi*sd1*sd2*sqrt(1-rho^2))*exp(-Q/(2*(1-rho^2)))
  }
  
  
  ## Creates covariance matrix from sd.x, sd.y, and rho
  calc.Sig = function(sd.x, sd.y, rho) {
    
    sig.xy = rho*sd.x*sd.y
    matrix(c(sd.x^2, sig.xy, sig.xy, sd.y^2), nrow=2)
  }
  
  
  ## Returns bivariate normal density for specified x-y grid
  dmvnorm = function(x, y, mu, Sig) {
    
    if(missing(mu)) mu = c(0,0)
    if(missing(Sig)) Sig = diag(2)
    
    outer(x, y, fxy, mu, Sig)
  }
  ########################## End of Helper Function ###############################
  
  # Surface: parameters drawn from data
  mu1 <- mean(newData$Strategy)
  mu2 <- mean(newData$MKT)
  rho <- cor(newData$Strategy, newData$MKT)
  sig1 <- sd(newData$Strategy)
  sig2 <- sd(newData$MKT)
  covar12 <- cov(newData$Strategy, newData$MKT)
  sigma <- matrix(c(sig1^2, rho*sig1*sig2*covar12, rho*sig1*sig2*covar12, sig2^2)*1e5, ncol=2)
  simAxis <- mvtnorm::rmvnorm(n=100, mean=c(1,2), sigma=sigma)
  mu = colMeans(simAxis); Sigma = var(simAxis); mu; Sigma
  simSurface = t(t(simAxis) - mu) %*% Sigma %*% t(simAxis) - mu
  simSurface <- simSurface/1e5
  z_up <- simSurface + matrix(3*sd(simSurface), 100, 100)
  z_down <- simSurface - matrix(3*sd(simSurface), 100, 100)
  PLT3 <- plot_ly(z = ~simSurface, type = "surface") %>% 
    add_surface(z = ~z_up, opacity = 0.5) %>% 
    add_surface(z = ~z_down, opacity = 0.5) %>% 
    layout(title = "Multivariate Gaussian Simulation Using Parameters from Strategy and Market with +/- 3x Standard Deviation")
  
  # Surface: parameters drawn from theoretical value
  N = 100
  x = y = seq(-3.2,3.2,le=N)  # create x-y grid of size NxN
  mu = c(0, 0)
  sigma <- matrix(c(1, 0, 0, 1), ncol=2)
  z = dmvnorm(x, y, mu, sigma)
  z = z - 2*mean(z)
  z_uuuuu <- z + matrix(9*sd(z), N, N)
  z_uuuu <- z + matrix(9*sd(z), N, N)
  z_uuu <- z + matrix(1.96*sd(z), N, N)
  z_uu <- z + matrix(1.5*sd(z), N, N)
  z_dd <- z - matrix(1.5*sd(z), N, N); z_dd = - (z_dd - 6*mean(z))
  z_ddd <- z - matrix(1.96*sd(z), N, N); z_ddd = - (z_ddd - 6*mean(z))
  z_dddd <- z - matrix(9*sd(z), N, N)
  z_ddddd <- z - matrix(12*sd(z), N, N)
  PLT3 <- plot_ly(z = ~simSurface, type = "surface") %>% 
    add_surface(z = ~z_uuu, opacity = 0.7) %>% 
    add_surface(z = ~z_uu, opacity = 0.7) %>% 
    add_surface(z = ~z_dd, opacity = 0.7) %>% 
    add_surface(z = ~z_ddd, opacity = 0.7)
  
  # List of Stocks Held
  dateListNom <- c()
  for (i in 1:nrow(returnData)) {
    if (i %% pastNforGrowth == 0) {
      dateListNom <- c(dateListNom, rownames(returnData)[i])
    }
  }
  rownames(listNom) <- dateListNom
  
  # Performance Plot by Converting Returns back to Values
  library(dygraphs)
  proposedStrategyPath <- cumprod(1 + growthStrategy)
  marketPath <- cumprod(1 + returnData$SPY)
  pathData <- data.frame(Strategy = proposedStrategyPath, Market = marketPath)
  rownames(pathData) <- rownames(returnData)
  dyPlot <- dygraph(
    pathData,
    main = paste0(
      "Proposed: Growth Strategy SR=", round(mean(newData[, 1])/sd(newData[, 1]), 4),
      " vs. Benchmark: Market Index Fund SR=", round(mean(newData[, 2])/sd(newData[, 2]), 4)
    )) %>%
    dyRebase(value = intializedValue) %>%
    dyLegend(show = "follow")
  
  # Update Holdings
  for (k in 1:length(dateListNom)) {
    dyPlot <- dyPlot %>%
      dyEvent(dateListNom[k], paste0(listNom[k, ], collapse = "_"),
              labelLoc = "bottom",
              strokePattern = "dotted") }
  
  # Final Visualization
  # dyPlot
  
  # Execution
  eachWeight <- howMuchtoInvest/length(selectedStockIndex)
  sharesVector <- c()
  weightVector <- sapply(selectedStockIndex, function(x) {sharesVector <- c(
    sharesVector, eachWeight/tail(data.frame(myList[x])[, 4], 1))})
  #weightVector
  
  # Output
  #return(list(Stocks = symbols, EqualWeight = weightVector))
  
  # Output
  return(list(
    Original_Stock_Data_List = myList,
    Return_Data = returnData,
    Return_Data_for_Past_Period = lastPeriodReturnData,
    Name_of_Stock_Held = listNom,
    Return_Data_New = newData,
    Path_Data_New = pathData,
    Linear_Model = linearModel,
    Visualization = dyPlot,
    PLT1 = PLT1,
    PLT2 = PLT2,
    # PLT3 = PLT3,
    ExeStocks = symbols[c(selectedStockIndex)],
    ExeShsEqWeight = weightVector
  ))
} # enc of function

#### DESIGN APP ####

# Design App
shinyApp(
  # UI
  ui = tagList(
    shinythemes::themeSelector(),
    navbarPage(
      # theme = "cerulean",  # <--- To use a theme, uncomment this
      "YIN'S Q BRANCH",
      fluid = TRUE,
      collapsible = TRUE,
      ######## Navbar 1 ########
      tabPanel("Navbar 1: TIMING STOCK MARKET",
               sidebarPanel(
                 width = 4,
                 fluidRow(
                   column( 3, textInput("start_date", "Start Date:", "1991-01-01") ),
                   column( 3, textInput("ticker", "Target Stock:", "AAPL") ),
                   column( 6, sliderInput("coef", "Buy/Sell Coefficients", min = -30, max = +30, value = c(-10, 10), step = 0.1) )
                 ),
                 fluidRow(
                   column( 4, sliderInput("pastNdays", "Comparison: All Data vs. Past N Days Data.", min = 3, max = 250, value = 20, step = 1, animate = TRUE) ),
                   column( 4, sliderInput("height", "Height of Signals", min = 1, max = 500, value = 30, step = 1) ),
                   column( 4, textInput("movingAvEnv", "Multivariate Moving Average", "10, 20, 50, 70, 100, 120, 150, 170, 200, 250") )
                 ),
                 fluidRow(
                   column( 4, sliderInput("seed", "Set seed (for reproducbility): ", min = 1, max = 1e3, value = 100, step = 1) ),
                   column( 4, textInput("layers", "Enter a vector of number(s) for layers (must be 1, 2, or 3 numbers separated with comma and space):", "24, 12, 5") ),
                   column( 4, sliderInput("epoch", "Number of epochs: ", min = 1, max = 100, value = 5, step = 1) ),
                   column( 4, checkboxInput("rescale", "Rescale data to between 0 and 1:", TRUE) )
                 ),
                 helpText("Note: Buy/Sell Coefficients"),
                 helpText("- Default value for buy and sell is [-2, +2], resonating the ~1.96 cutoff to cover 95% confidence interval in statistics."),
                 helpText("- Implication: price of interests is the bottom 2.5% for buy and top 2.5% for sell."),
                 helpText("- Suggest: change the values so that the average is as little as 1% to create meaningful actionable area."),
                 helpText("Note: Past N Days"),
                 helpText("- Summary of statistics makes more sense when comparing results between all data and recent data."),
                 helpText("- Default is set to be 20 days, i.e. comparison summary statistics between that of all data and that of the past 20 business days."),
                 helpText("Note: Height"),
                 helpText("- Height is for visualization, i.e. the height of the signals on chart."),
                 helpText("- Default height is set at 30. If all moving averages are triggered, the max signal is breached and the value is 30."),
                 helpText("- Implication: This is a price that is statistically unlikely based on all historical price data and will likely reverse."),
                 helpText("Note: Moving Average Envelop"),
                 helpText("- Enter any sets of numeric values separated with ', '; error may appear if value is longer than ticker's entire length."),
                 submitButton("Submit", width = "100%")
               ),
               mainPanel(
                 width = 8,
                 height = 24,
                 tabsetPanel(
                   tabPanel("Basics",
                            h3("Chart for Entered Ticker"),
                            dygraphOutput("basicMMstats_Plot", width = "100%", height = "470px"),
                            h3("Summary of Statistics"),
                            helpText("The following table is the summary of statistics calculated using ALL data."),
                            dataTableOutput("basicMMstats_Statistics"),
                            helpText("The following table is the summary of statistics calculated using recent data."),
                            dataTableOutput("basicMMstats_RecentStats"),
                            helpText("Note:"),
                            helpText("- Be aware of change of volatility, i.e. standard deviation or variance."),
                            helpText("- Be aware of change of expected returns or losses. If values are greater than 2 SD, it won't stay there much longer.")
                   ),
                   tabPanel("Yin's Timer",
                            h3("Visualization"),
                            helpText("This path presents stock performance along with buy and sell signals (if zoom in, height of signals can be controlled from left)."),
                            dygraphOutput("timing_FinalPlot", width = "100%", height = "470px"),
                            h3("Buy Sell Table Along with Price"),
                            helpText("Price Data (closing price) of target stock and related buy/sell signals."),
                            dataTableOutput("timing_buysellTable"),
                            h3("Statistics from Timing Results"),
                            helpText("The following table summarizes the frequency and variations of buy/sell signals."),
                            dataTableOutput("timing_Statistics"),
                            helpText("Comment: I would recommend to pick stocks with long history and to keep buy signals to be 1-2% and sell signals to be less than 1%.")
                   ),
                   tabPanel("Yin's Forecast",
                            h3("Visualization: Real"),
                            dygraphOutput("realPlt", width = "100%", height = "430px"),
                            h3("Visualization: Predicted"),
                            dygraphOutput("forecastPlt", width = "100%", height = "430px"))
                 )
               )
      ),
      ######## Navbar 2 ########
      tabPanel("Navbar 2: CONSTRUCT PORTFOLIO", 
               sidebarPanel(
                 width = 4,
                 h3("Enter Preferred Tickers"),
                 fluidRow(
                   column(12, textInput("symbolsIDX", "Market Index Fund", "SPY, QQQ"))
                 ),
                 fluidRow(
                   column(3, textInput("symbolsTECH", "Tech Sector", "MSFT, GOOGL, FB")),
                   column(3, textInput("symbolsFINANCIAL", "Financial Sector", "V, JPM, MA")),
                   column(3, textInput("symbolsINDUSTRIAL", "Industrial Sector", "HON, LMT, RTX")),
                   column(3, textInput("symbolsSERVICE", "Service Sector", "AMZN, DIS, BABA"))
                 ),
                 fluidRow(
                   column(3, textInput("symbolsCONSUMER", "Consumer Sector", "AAPL, TSLA"))
                 ),
                 helpText("Note: "),
                 helpText("- Enter Stock Tickers (separate by comma and a space, i.e. ', ')"),
                 helpText("- Keep it short. Ex: Default 37 stocks takes about ~ 60 sec."),
                 helpText("- Early morning/late evening sites will be busy. If error appears, try again in a few hours."),
                 hr(),
                 fluidRow(
                   column(3, sliderInput(inputId = "pastNforGrowth", label = "Evaluation Window:", value = 20, min = 10, max = 50, step = 1)),
                   column(3, sliderInput(inputId = "topGrowthNum", label = "Number of Top Stocks", value = 4, min = 2, max = 30, step = 1)),
                   column(3, numericInput(inputId = "intializedValue", label = "Initial Value", value = 1)),
                   column(3, numericInput(inputId = "howMuchtoInvest", label = "Initial Investment", value = 1e6))
                 ),
                 helpText("Note Initial Value: the value the chart starts at, i.e. default is $1"),
                 helpText("Note Inivital Investment: the initial investment user plans to execute the algorithm, i.e. default is $1M."),
                 hr(), helpText("Source:"),
                 uiOutput("src_indexfund"),
                 uiOutput("src_tech"),
                 uiOutput("src_financial"),
                 uiOutput("src_industrial"),
                 uiOutput("src_consumer"),
                 uiOutput("src_service"),
                 uiOutput("src_channelup"),
                 uiOutput("src_earnings"),
                 hr(), 
                 submitButton("Submit", width = "100%")
               ),
               mainPanel(
                 width = 8,
                 tabsetPanel(
                   tabPanel("Quant: Growth Strategy",
                            h3("Charts"),
                            plotlyOutput("Visualization_data_2", width = "100%", height = "1200px"),
                            h3("Cross Returns"),
                            plotlyOutput("Visualization_data_1", width = "100%", height = "400px"),
                            helpText("Note:"),
                            helpText("- Visualization of raw data composed of returns from different unit of analysis."),
                            helpText("- Two axis: x is daily returns and y is annual returns."),
                            helpText("- Size of circles: average volume from past 10 trading days."),
                            helpText("- Color: sectors include Market Index, Technology, Financial, Industrial, and Consumer."),
                            helpText("- Selection bias: default tickers are based on my experience and as of this point I recommend to start with this stock pool."),
                            h3("Performance"),
                            helpText("Paths for Benchmark and Proposed Portfolio by Growth Strategy starting from initial value (default $1 and can be changed by user)."),
                            dygraphOutput("Visualization_performance", width = "100%", height = "500px"),
                            h3("Execution"),
                            helpText("The holdings and amount of shares the algorithm suggests in portfolio right now according to initial investment (default $1M and can be changed by user)."),
                            dataTableOutput("Execution"),
                            h3("Summary"),
                            dataTableOutput("reportLM"),
                            h3("Results from OLS Regression"),
                            helpText("The estimated intercept is alpha. The t value to test whether the estimate is statistically significant."),
                            verbatimTextOutput("linearModel")
                            # h3("SIMULATION"),
                            # plotlyOutput("Visualization_data_3", width = "100%", height = "2400px"),
                            # helpText("Note:"),
                            # helpText("- Visualization of Multivariate Gaussian."),
                            # helpText("- Spiked surface has parameters mean matrix and covariance matrix are drawn from data from Strategy and Market Benchmark."),
                            # helpText("- Spiked surface has parameters mean matrix and covariance matrix are drawn from theoretical values, i.e. Bivariate Normal Distribution.")
                   ),
                   tabPanel("Tab 2", "This panel is intentionally left blank")
                 )
               )),
      tags$head(
        conditionalPanel(
          condition="input.goButton > 0 | $('html').hasClass('shiny-busy')",
          tags$div(
            c("Calculating... Please wait... Patience is the key to success. -- Yiqiao Yin",
              "Calculating... Please wait... Trees that are slow to grow bear the best fruit. -- Moliere",
              "Calculating... Please wait... He that can have patience can have what he will. -- Benjamin Franklin",
              "Calculating... Please wait... Patience is bitter, but its fruit is sweet. -- Jean-Jacques Rousseau",
              "Calculating... Please wait... The two most powerful warriors are patience and time. -- Leo Tolstoy" )[sample(5,1)] )) )
    )
  ),
  # Server
  server = function(input, output) {
    ######## Navbar 1 ########
    # Basic MM:
    basicMoneyManagement <- reactive({
      BasicMM(
        target = input$ticker,
        start_date = input$start_date,
        past_N_days = input$pastNdays ) })
    output$basicMMstats_Plot <- renderDygraph({ basicMoneyManagement()$Plot })
    output$basicMMstats_Statistics <- renderDataTable({ data.frame(basicMoneyManagement()$Statistics )})
    output$basicMMstats_RecentStats <- renderDataTable({ data.frame(basicMoneyManagement()$RecentStats )})
    
    # Timing:
    timing <- reactive({
      AdvBuySellAlgorithm(
        target = input$ticker,
        start_date = input$start_date,
        buyCoef = input$coef[1],
        sellCoef = input$coef[2],
        height = input$height,
        movingAvEnv = c(as.numeric(as.character(unlist(strsplit(input$movingAvEnv, ", "))))) )})
    output$timing_FinalPlot <- renderDygraph({ timing()$FinalPlot })
    output$timing_buysellTable <- renderDataTable({ data.frame(cbind( timing()$buysellTable)) })
    output$timing_Statistics <- renderDataTable({ data.frame(timing()$Statistics) })
    
    # Forecast:
    sequence_to_sequence <- reactive({
      if (length(as.numeric(unlist(strsplit(input$layers, ", ")))) == 3) {
        Seq2Seq(
          target = input$ticker,
          past_N_days = input$pastNdays,
          seed = input$seed,
          num_hidden = length(as.numeric(unlist(strsplit(input$layers, ", ")))),
          l1.units = as.numeric(unlist(strsplit(input$layers, ", ")))[1],
          l2.units = as.numeric(unlist(strsplit(input$layers, ", ")))[2],
          l3.units = as.numeric(unlist(strsplit(input$layers, ", ")))[3],
          epoch = input$epoch,
          rescale = input$rescale )
      } else if (length(as.numeric(unlist(strsplit(input$layers, ", ")))) == 2) {
        Seq2Seq(
          target = input$ticker,
          past_N_days = input$pastNdays,
          seed = input$seed,
          num_hidden = length(as.numeric(unlist(strsplit(input$layers, ", ")))),
          l1.units = as.numeric(unlist(strsplit(input$layers, ", ")))[1],
          l2.units = as.numeric(unlist(strsplit(input$layers, ", ")))[2],
          l3.units = 10,
          epoch = input$epoch,
          rescale = input$rescale )
      } else if (length(as.numeric(unlist(strsplit(input$layers, ", ")))) == 1) {
        Seq2Seq(
          target = input$ticker,
          past_N_days = input$pastNdays,
          seed = input$seed,
          num_hidden = length(as.numeric(unlist(strsplit(input$layers, ", ")))),
          l1.units = as.numeric(unlist(strsplit(input$layers, ", ")))[1],
          l2.units = 32,
          l3.units = 10,
          epoch = input$epoch,
          rescale = input$rescale )
      } else {
        Seq2Seq(
          target = input$ticker,
          past_N_days = input$pastNdays,
          seed = input$seed,
          num_hidden = 3,
          l1.units = 128,
          l2.units = 64,
          l3.units = 10,
          epoch = input$epoch,
          rescale = input$rescale )
      }
      
      
      
    })
    output$realPlt <- renderDygraph({ sequence_to_sequence()$realPlt })
    output$forecastPlt <- renderDygraph({ sequence_to_sequence()$forecastPlt })
    
    ######## Navbar 2 ########
    # Input:
    urlchannelup <- reactive({
      a("Technical: Channel Up",
        href = paste0(
          "https://finviz.com/screener.ashx?v=211&f=cap_largeover,sh_avgvol_o1000,ta_pattern_channelup,ta_sma20_pa&ft=3" ))
    })
    output$src_channelup <- renderUI({
      tagList(urlchannelup()) })
    urlearnings <- reactive({
      a("Fundamentals: Earnings",
        href = paste0(
          "https://finviz.com/screener.ashx?v=211&f=cap_largeover,earningsdate_thisweek,sh_avgvol_o2000" ))
    })
    output$src_earnings <- renderUI({
      tagList(urlearnings()) })
    urlindexfund <- reactive({
      a("Index Fund", 
        href = paste0(
          "https://finviz.com/screener.ashx?v=111&f=cap_microover,geo_usa,ind_exchangetradedfund")) })
    output$src_indexfunc <- renderUI({
      tagList(urlindexfund()) })
    urltechnology <- reactive({
      a("Technology", 
        href = paste0(
          "https://finviz.com/screener.ashx?v=111&f=cap_largeover,geo_usa,sec_technology&o=-marketcap")) })
    output$src_tech <- renderUI({
      tagList(urltechnology()) })
    urlfinancial <- reactive({
      a("Financial", 
        href = paste0(
          "https://finviz.com/screener.ashx?v=111&f=cap_largeover,geo_usa,sec_financial&o=-marketcap")) })
    output$src_financial <- renderUI({
      tagList(urlfinancial()) })
    urlindustrial <- reactive({
      a("Industrial", 
        href = paste0(
          "https://finviz.com/screener.ashx?v=111&f=cap_largeover,geo_usa,sec_industrialgoods&o=-marketcap")) })
    output$src_industrial <- renderUI({
      tagList(urlindustrial()) })
    urlconsumer <- reactive({
      a("Consumer", 
        href = paste0(
          "https://finviz.com/screener.ashx?v=111&f=cap_largeover,geo_usa,sec_consumergoods&o=-marketcap")) })
    output$src_consumer <- renderUI({
      tagList(urlconsumer()) })
    urlservice <- reactive({
      a("Service", 
        href = paste0(
          "https://finviz.com/screener.ashx?v=111&f=cap_largeover,sec_services&o=-marketcap")) })
    output$src_service <- renderUI({
      tagList(urlservice()) })
    
    # Quant: Growth Strategy
    results <- reactive({QuantGrowthStrategy(
      # Get Data
      symbolsIDX = c(unlist(strsplit(input$symbolsIDX, ", "))),
      symbolsTECH = c(unlist(strsplit(input$symbolsTECH, ", "))),
      symbolsFINANCIAL = c(unlist(strsplit(input$symbolsFINANCIAL, ", "))),
      symbolsINDUSTRIAL = c(unlist(strsplit(input$symbolsINDUSTRIAL, ", "))),
      symbolsCONSUMER = c(unlist(strsplit(input$symbolsCONSUMER, ", "))),
      symbolsSERVICE = c(unlist(strsplit(input$symbolsSERVICE, ", "))),
      pastNforGrowth = input$pastNforGrowth,
      topGrowthNum = input$topGrowthNum,
      intializedValue = input$intializedValue,
      howMuchtoInvest = input$howMuchtoInvest )})
    output$reportLM <- renderDataTable({
      data.frame( Content = rbind(
        Description = paste0(
          "Given entered stock tickers and parameters, the constructed portfolio has an alpha of ",
          round(results()$Linear_Model$coefficients[1], 7)*100, "% based on daily data, which is measured by the intercept of the linear model. ",
          "This translates to beating the market (say S&P 500 Index Fund: SPY) by ", round((1 + results()$Linear_Model$coefficients[1])^(250) - 1, 3)*100,
          "% on an annual basis. ",
          "The portfolio has a market beta of ", round(results()$Linear_Model$coefficients[2], 3), ". ",
          "This measure implies that if market changes +/- 1% then proposed portfolio will also change +/- ", 
          round(results()$Linear_Model$coefficients[2], 3), "% on average. ",
          "The proposed strategy has Sharpe Ratio of ",
          round(mean(results()$Return_Data_New[, 1]), 5), "/", round(sd(results()$Return_Data_New[, 1]), 5), "=",
          round(mean(results()$Return_Data_New[, 1])/sd(results()$Return_Data_New[, 1]), 3),
          " while market has Sharpe Ratio of ", 
          round(mean(results()$Return_Data_New[, 2]), 5), "/", round(sd(results()$Return_Data_New[, 2]), 5), "=",
          round(mean(results()$Return_Data_New[, 2])/sd(results()$Return_Data_New[, 2]), 3),
          " based on daily data. If based on monthly data, these numbers translate to ",
          round(mean(na.omit(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 20)]/Lag(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 20)], 1) - 1)), 5), 
          "/",
          round(sd(na.omit(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 20)]/Lag(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 20)], 1) - 1)), 5),
          "=",
          round(
            mean(na.omit(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 20)]/Lag(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 20)], 1) - 1)) /
              sd(na.omit(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 20)]/Lag(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 20)], 1) - 1)), 3),
          " and ",
          round(mean(na.omit(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 20)]/Lag(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 20)], 1) - 1)), 5),
          "/",
          round(sd(na.omit(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 20)]/Lag(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 20)], 1) - 1)), 5),
          "=",
          round(
            mean(na.omit(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 20)]/Lag(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 20)], 1) - 1)) /
              sd(na.omit(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 20)]/Lag(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 20)], 1) - 1)), 3),
          " for proposed strategy and market, respectively. If based on annual data, we have Sharpe Ratio to be ",
          round(mean(na.omit(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 252)]/Lag(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 252)], 1) - 1)), 5), 
          "/",
          round(sd(na.omit(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 252)]/Lag(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 252)], 1) - 1)), 5),
          "=",
          round(
            mean(na.omit(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 252)]/Lag(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 252)], 1) - 1)) /
              sd(na.omit(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 252)]/Lag(results()$Path_Data_New[, 1][seq(1, length(results()$Path_Data_New[, 1]), 252)], 1) - 1)), 3),
          " and ",
          round(mean(na.omit(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 252)]/Lag(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 252)], 1) - 1)), 5),
          "/",
          round(sd(na.omit(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 252)]/Lag(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 252)], 1) - 1)), 5),
          "=",
          round(
            mean(na.omit(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 252)]/Lag(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 252)], 1) - 1)) /
              sd(na.omit(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 252)]/Lag(results()$Path_Data_New[, 2][seq(1, length(results()$Path_Data_New[, 2]), 252)], 1) - 1)), 3),
          " for proposed strategy and market, respectively.",
          " Be aware that Sharpe Ratio will make much more sense when it comes to comparison of two paths using the same unit of analysis. ",
          " In addition, please see the following:"),
        "(1)" = "Results from OLS Regression: a linear model measuring proposed strategy returns using market returns.",
        "(2)" = "Performance: visualization of paths for proposed strategy and market benchmark.",
        "(3)" = "Execution: stocks and shares held for current period using proposed strategy."
      ) )})
    output$Visualization_data_1 <- renderPlotly({ results()$PLT1 })
    output$Visualization_data_2 <- renderPlotly({ results()$PLT2 })
    # output$Visualization_data_3 <- renderPlotly({ results()$PLT3 })
    output$linearModel <- renderPrint({ summary(results()$Linear_Model) })
    output$Visualization_performance <- renderDygraph({ results()$Visualization })
    output$Execution <- renderDataTable({
      data.frame(
        Stocks = results()$ExeStocks,
        Shares = round(results()$ExeShsEqWeight, 2) ) })
  }
)

#### END SCRIPT ####