QuantGrowthStrategy <- function(
  symbolsIDX = c("SPY"),
  symbolsTECH = c("AAPL", "FB", "NVDA", "GOOGL", "AMZN"),
  symbolsFINANCIAL = c("BAC", "GS", "JPM", "MS", "V", "PYPL"),
  symbolsINDUSTRIAL = c("LMT", "GD", "BA", "CAT"),
  symbolsCONSUMER = c("WMT", "TGT", "PEP", "DIS", "KO"),
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
  symbols <- c(symbolsIDX, symbolsTECH, symbolsFINANCIAL, symbolsINDUSTRIAL, symbolsCONSUMER)
  myList <- list()
  myList <- lapply(symbols, function(x) {getSymbols(x, from = as.Date("2000-01-04"),  auto.assign = FALSE)})
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
  past10Vol <- c()
  past10Vol <- sapply(1:length(symbols), function(i) past10Vol <- c(
    past10Vol,
    mean(tail(myList[[i]][, 5], 10)) ))
  data_for_viz <- cbind(symbols, AnnRet, QuartRet, WeekRet, DayRet, past10Vol)
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
    sector = c(rep("Mkt", length(symbolsIDX)), rep("Tech", length(symbolsTECH)),
               rep("Finance", length(symbolsFINANCIAL)), rep("Industrial", length(symbolsINDUSTRIAL)),
               rep("Consumer", length(symbolsCONSUMER)) ) )

  # Visualization
  library(plotly)
  dtaViz <- data.frame(data_for_viz)
  slope <- 2e-5*0.9
  dtaViz$size <- sqrt(as.numeric(as.character(dtaViz$past10Vol)) * slope)
  colors <- c('#4AC6B7', '#1972A4', '#965F8A', '#FF7070', '#C61951')
  PLT1 <- plot_ly(dtaViz, x = ~DayRet, y = ~AnnRet, color = ~sector, size = ~size, colors = colors,
                  type = 'scatter', mode = 'markers', sizes = c(min(dtaViz$size), max(dtaViz$size)),
                  marker = list(symbol = 'circle', sizemode = 'diameter'),
                  text = ~paste('symbols:', symbols,
                                '<br>Sector:', sector,
                                '<br>Ave. Daily Returns:', DayRet,
                                '<br>Ave. Weekly Returns:', AnnRet,
                                '<br>Ave. Quarterly Returns:', QuartRet)) %>%
    layout(title = 'Data: Daily Return v. Annual Return \n(size: ave. of past 10 days volume; color: sector)',
           xaxis = list(title = 'Daily Returns', gridcolor = 'rgb(0, 204, 0)'),
           yaxis = list(title = 'Annual Returns', gridcolor = 'rgb(0, 204, 0)'))

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
    ExeStocks = symbols[c(selectedStockIndex)],
    ExeShsEqWeight = weightVector
  ))
}

