# GraphTransformer-DGraphFin
This is the Graph Transformer solution for [DGraphFin leaderboard](https://dgraph.xinye.com/leaderboards/dgraphfin).

## Dependencies
```{bash}
python = 3.8
torch = 1.10.1+cu102
torch-geometric = 2.0.4
scikit-learn = 0.20.4
torch-scatter = 2.0.9
torch-sparse = 0.6.12
```

## DGraph-Fin Dateset
The dataset [DGraph-Fin](https://dgraph.xinye.com/dataset) should be download and placed in place it under the folder `./dataset/DGraph/raw`.

## Run Command
```{bash}
python main.py --model transformer --epochs 1000 --runs 10
```

## Results
Performance on **DGraph-Fin** (10 runs):

Best Valid AUC: 0.7717 ± 0.0017

Best Test AUC: 0.7838 ± 0.0015
