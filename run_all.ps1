conda activate causal_effect_ci
python ./experiments/nonlinear.py --d_nodes=5 --k_edge_multiplier=1 --repetitions=200
python ./experiments/nonlinear.py --d_nodes=5 --k_edge_multiplier=2 --repetitions=200
python ./experiments/nonlinear.py --d_nodes=10 --k_edge_multiplier=1 --repetitions=200
python ./experiments/nonlinear.py --d_nodes=10 --k_edge_multiplier=2 --repetitions=200
python ./experiments/nonlinear.py --d_nodes=20 --k_edge_multiplier=1 --repetitions=200
python ./experiments/nonlinear.py --d_nodes=20 --k_edge_multiplier=2 --repetitions=200