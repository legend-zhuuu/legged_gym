PYTHONPATH=.:./rsl_rl python legged_gym/scripts/train.py --task=aliengo --max_iterations=10000 --run_name=perlin_measure --num_envs=8192 --rl_device=cuda:1 --sim_device=cuda:1 --headless
