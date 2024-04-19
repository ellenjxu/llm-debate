for agents in {5..7}; do
    for rounds in {1..4}; do
        python gen_math.py --agents $agents --rounds $rounds
    done
done