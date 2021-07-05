# Q-Branch

[![AnYinProduction](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://yinscapital.com/research/)
[![Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This github repo introduces a sample flagship product of Yins Capital.

<p align="center">
  <img width="460" height="300" src="https://github.com/yiqiao-yin/Q-Branch/blob/main/figs/main.gif">
</p>

## Product Introduction

This software product walks the reader through a basic pipeline, one of the major pipelines developed by [Yins Capital](https://www.YinsCapital.com/). A basic illustration can be seen in the following diagram.

$$
\begin{bmatrix}\text{Client Requirements}\end{bmatrix}
\rightarrow
\begin{bmatrix}\text{Data Mining: Proposal of Initial Portfolio}\end{bmatrix}
\rightarrow
$$
$$
\begin{bmatrix}\text{Visualization + Simulation: Amendment of Proposed Portfolio}\end{bmatrix}
\rightarrow
$$
$$
\begin{bmatrix}\text{Profit Maximization + Potential Exit Strategy}\end{bmatrix}
$$

## Usage

*Pre-requisite*: This repo has the following dependencies. Please make sure you have these libraries installed.

```
# Library
# install.packages(c("package1", "package2", ...))
library(quantmod)
library(dygraphs)
library(DT)
library(plotly)
```

Sample usage: clone repo by running the following in a *Git* command window.

```
git clone https://github.com/yiqiao-yin/Q-Branch.git
```

The following code assumes a location of *XXX* where this repo is stored. Using *source()* function, one can load the defined functions into *RStudio*. The following code gives you some sample visualization.

```
source("XXX") # path of the directory where the script of QuantGrowthStrategy() is saved
tmp = QuantGrowthStrategy()
tmp$Visualization
tmp$PLT1
tmp$ExeStocks
tmp$ExeShsEqWeight
```

<p align="center">
  <img width="460" height="300" src="https://github.com/yiqiao-yin/Q-Branch/blob/main/figs/cross-section-returns.png">
</p>

<p align="center">
  <img width="460" height="300" src="https://github.com/yiqiao-yin/Q-Branch/blob/main/figs/growth-strategy.png">
</p>
