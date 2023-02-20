PYTHONPATH=.:./rsl_rl python legged_gym/scripts/train.py --task=aliengo --max_iterations=10 --run_name=debug --num_envs=256 --rl_device=cuda:0 --sim_device=cuda:0 --headless
