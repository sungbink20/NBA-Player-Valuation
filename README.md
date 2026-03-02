# 🏀 NBA Trade Value Prediction with Graph Neural Networks  

## Overview  

This project models the NBA trade market as a graph learning problem.  
We construct player–team interaction graphs and use Graph Neural Networks (GNNs) to predict player trade value and salary.

Built as a final project for CS224W (Machine Learning with Graphs) at Stanford University.

---

## Problem Motivation  

Traditional player valuation models rely on tabular statistics and regression.  
However, NBA trade value is inherently relational:

- Players belong to teams  
- Teams interact through trades  
- Player performance depends on team context  
- Salary reflects market dynamics  

To better capture these structured dependencies, we model the NBA ecosystem as a graph.

---

## Graph Construction  

We experiment with two graph formulations:

### Homogeneous Graph
- Nodes: Players  
- Edges: Shared team membership or trade relationships  
- Baseline: Standard GraphSAGE  

### Heterogeneous Graph
- Node types: Players, Teams  
- Edge types:
  - Player → Team (membership)
  - Team → Player
  - Team ↔ Team (trade interactions)

This enables type-specific message passing and richer relational modeling.

---

## Model Architectures  

Implemented models:

- MLP (tabular baseline)
- Homogeneous GraphSAGE
- Heterogeneous GraphSAGE

Each model predicts:
- Player salary (regression task)
- Trade value proxy score

---

## Evaluation Metrics  

- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

The heterogeneous GraphSAGE model outperformed both:
- The tabular MLP baseline  
- The homogeneous GNN  

Highlighting the importance of structured relational modeling.

---

## Tech Stack  

- PyTorch  
- PyTorch Geometric  
- Pandas / NumPy  
- Custom preprocessing and graph construction pipeline  

---

## Key Takeaways  

- Graph modeling better captures market dynamics than flat tabular regression  
- Heterogeneous message passing improves predictive performance  
- Relational inductive bias provides measurable gains in valuation accuracy
- https://medium.com/@vikgarrett/predicting-nba-player-market-value-with-graph-neural-networks-18f56005a684?postPublishedType=initial 
