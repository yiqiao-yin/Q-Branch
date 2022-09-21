# Q-Branch

[![AnYinProduction](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://yinscapital.com/research/) [![Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This github repo introduces a sample flagship product of Yins Capital. A live UI is published [here](https://y-yin.shinyapps.io/YINS-Q-BRANCH/) using [R Shiny](https://shiny.rstudio.com/tutorial/) to assit user-friendly interface. More sample products can be accessed [here](https://github.com/yiqiao-yin/YinsCapital).

<p align="center">

<img src="https://github.com/yiqiao-yin/Q-Branch/blob/main/figs/main.gif" width="460" height="300"/>

</p>

## Product Introduction

This software product walks the reader through a basic pipeline, one of the major pipelines developed by [Yins Capital](https://www.YinsCapital.com/). A basic illustration can be seen in the following diagram.

<p align="center">
<img src="https://github.com/yiqiao-yin/Q-Branch/blob/main/figs/product-introduction.png" width="700" height="350"/>

</p>

## Usage

*Pre-requisite*: This repo has the following dependencies. Please make sure you have these libraries installed.

    # Library
    # install.packages(c("package1", "package2", ...))
    library(quantmod)
    library(dygraphs)
    library(DT)
    library(plotly)

Sample usage: clone repo by running the following in a *Git* command window.

    git clone https://github.com/yiqiao-yin/Q-Branch.git

The following code assumes a location of *XXX* where this repo is stored. Using *source()* function, one can load the defined functions into *RStudio*. The following code gives you some sample visualization.

    source("XXX") # path of the directory where the script of QuantGrowthStrategy() is saved
    tmp = QuantGrowthStrategy()
    tmp$Visualization
    tmp$PLT1
    tmp$ExeStocks
    tmp$ExeShsEqWeight

<p align="center">

<img src="https://github.com/yiqiao-yin/Q-Branch/blob/main/figs/cross-section-returns.png" width="460" height="300"/>

</p>

<p align="center">

<img src="https://github.com/yiqiao-yin/Q-Branch/blob/main/figs/growth-strategy.png" width="460" height="300"/>

</p>
