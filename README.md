# Deep Learning for Variable Annuity Risk Management

This repository contains the code and resources for my thesis on applying deep learning techniques to variable annuity risk management.

## Overview

Variable annuities (VAs) are complex insurance products that offer investment opportunities with guaranteed benefits. This thesis explores novel approaches to managing the risks associated with these products using deep learning and reinforcement learning techniques.

## Key Components

### 1. LSTM Metamodels for VA Valuation
- Implementation of LSTM-based neural networks for efficient VA contract valuation
- Transfer learning techniques to adapt models to new VA products with limited data
- Comparison with traditional nested simulation approaches

### 2. Deep Hedging with Reinforcement Learning
- Markov Decision Process (MDP) formulation for hedging VA contracts
- Implementation of Proximal Policy Optimization (PPO) algorithms for dynamic hedging
- Evaluation of hedging performance under different market conditions and transaction costs

### 3. Transfer Learning for Risk Management
- Policy transfer techniques for adapting hedging strategies to new VA products
- Reward shaping methods to incorporate domain knowledge
- Evaluation metrics for measuring transfer learning effectiveness

## Data and Simulation

The repository includes code for simulating VA contract dynamics under various asset models:
- Geometric Brownian Motion (GBM)
- Regime-Switching GBM (RS-GBM)
- Heston stochastic volatility model

## Usage

Detailed instructions for running experiments and reproducing results can be found in the respective directories.
