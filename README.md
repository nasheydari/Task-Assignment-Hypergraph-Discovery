# Task-Assignment-Hypergraph-Discovery


Here you can find the code for the paper "Assigning Entities to Teams as a Hypergraph Discovery Problem"

#### Install Required Packages

```bash
pip install -r dependency.txt
```





Set the config file in configs directory and run "run.py" with the specified config file.

## Parameters/Configs

#### Mode Parameters

   - data: APS/MAG
   - Algorithm: CSA/CSA_Bipartite/Greedy/GreedyBipartite/RandomGreedy/RandomGreedyBipartite
   

#### Folder Paths 
   - logging_path: logging path
   - folder_path: data folder path
   - res_path: result saving path

#### Optimization Parameters
   - epoch: number of optimization rounds
   - Nt: number of optimization rounds for each temperature
   - tol: convergence tolerace
   - temp: initial temperature
   - temp_decay: temperature decay rate
   - nremove: number of removed assignments in each perturbation
   - nswap: number of assignment swaps in each perturbation
   - sym: symmetric algebraic connectivity: default False
   - pen, pen1, pen2, pen3: coefficients of the penalty terms
   - return: probability of going back to the best found solution at each round
   - pack: energy pack size



#### Robustness

To evaluate the solution robustness tp node failures, run "robustness.py" file with the config specified in "Robustness.json"

## Parameters/Configs for Robustness

  - set any of APS/Bipartite_APS/Best_APS/Initial_APS/Greedy_APS/MAG/Bipartite_MAG/Best_MAG/Greedy_MAG/Initial_MAG to True to evaluate the corresponding solution

  - Natt: number of removed (attacked) nodes 
  - niter: number of times the solution is evaluated
