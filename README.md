# Approximationalgorithm for Maximum Budgeted Allocation
This repository contains an implementation of the approximationalgorithm from [Chakrabarty and Goel (2010)](https://epubs.siam.org/doi/10.1137/080735503) for the maximum budgeted allocation problem.

### Feasible Instances
The approximationalgorithm requires that the maximum bid of each agent is less than or equal to its budget.

## Installation

### Conda environment 

Create a new conda environment with python 3.14.x. Then install the following dependencies. 
- graph-tool 
- graph-tool-base
- ortools
- scipy

Activate your conda environment and use this commands to install the dependencies.

`conda install graph-tool graph-tool-base`  
`pip install ortools scipy`

### Install package
Activate your conda environment and install this package using `pyproject.toml` via  `pip install -e .` in the project root.

## Example
![Example 1 as a graph.](/data/example_1/example_1.png)

In example $1$, there are three agents, labeled $i$, each with a budget of $3$, and three items labeled $j$. The edge labels show the bids of an agent for the item connected by tge edge. The optimal allocation assigns item $j_1$ to agent $i_1$, item $j_2$ to agent $i_2$ and item $j_3$ to agent $i_3$, resulting in a total revenue of $7$.

## Usage
See `main.py` or tests for usage examples. 

You can run the `main.py` file with `python -m maximum_budgeted_allocation.main` in project root.# maximum-budgeted-allocation
