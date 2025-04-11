# RLDC-LLM
Rinforcement Learning Incorporating Disagreement and Connectedness Methods with Large Language Models for WTI Crude Oil Trading.

## Overview
The architecture of the RLDC-LLM framework, which uses Proximal Policy Optimization (PPO) to improve WTI crude oil futures trading performance. The twelve assets that were chosen for the framework are combined with insights from the static and dynamic connectivity analyses.
![It consists of twelve assets contained with the WTI to perceive the four different features from the proposed
Feature Extractor, i.e. time series feature, sentiment feature, basic disagreement feature, and cross disagreement feature.
Except for the WTI, the other eleven assets’ features are calculated by static or dynamic connectedness to enhance the
correlation with WTI trading, all twelve assets are then concatenated together as the input for the PPO module. The
multi-head attention module and the LSTM Layer are used to capture the critical features from the observation features.
The actor and critic networks share the same architecture. The main difference is that the output nodes for the actor
correspond to the three actions “buy”, “hold”, and “sell”, and the output node for the critic is to estimate the state value.
A well-designed reward with profit or loss and the transaction cost incurred is built to guide the training of the agent.](PPO.png)

## Quick start
1. Install related environment: pip install -r requirements.txt
2. Modify Data/887637.WI_processed.csv to your own data
3. Run the Append_single_asset.py for the demo. 
