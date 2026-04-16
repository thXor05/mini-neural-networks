# train_slp.py

from models.slp import SLP

epochs = 1000
lr = 0.1

def train_slp():
    slp = SLP(input_size=2, output_size=1)

    print("\nTraining started...")
    
    for epoch in range(epochs):
        slp.backward([0, 0], [0], lr)
        slp.backward([0, 1], [0], lr)
        slp.backward([1, 0], [0], lr)
        slp.backward([1, 1], [1], lr)

        print(f"Epoch {epoch}/{epochs}")
        print("Output for [0, 0]:", slp.forward([0, 0]))
        print("Output for [0, 1]:", slp.forward([0, 1]))
        print("Output for [1, 0]:", slp.forward([1, 0]))
        print("Output for [1, 1]:", slp.forward([1, 1]))
        print("-" * 32)

    print("\nTraining finished\n")
    print("-" * 64)

