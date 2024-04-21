echo "Running experiment with all agents using the same model..."
# python gen_math_multi.py --models 0 0 0
python gen_math_multi.py --models 1 1 1
python gen_math_multi.py --models 2 2 2 

echo "Running experiment with each agent using a different model..."
python gen_math_multi.py --models 0 1 2

echo "Running experiment with two agents using the same model and one using a different model..."
python gen_math_multi.py --models 0 0 1
