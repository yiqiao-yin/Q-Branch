# Q-Branch

[![AnYinProduction](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://yinscapital.com/research/)
[![Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This github repo introduces a sample flagship product of Yins Capital.

<p align="center">
  <img width="460" height="300" src="https://github.com/yiqiao-yin/Q-Branch/blob/main/figs/main.gif">
</p>

## Usage

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
