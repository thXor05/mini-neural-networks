# train_mlp.py

from models.mlp import MLP

epochs = 100000
lr = 0.01

def train_mlp():
    input_size = 2
    hidden_size = 2
    output_size = 1

    mlp = MLP([input_size, hidden_size, output_size])

    print("\nTraining started...")
    
    for epoch in range(epochs):
        mlp.backward([0, 0], [0], lr)
        mlp.backward([0, 1], [1], lr)
        mlp.backward([1, 0], [1], lr)
        mlp.backward([1, 1], [0], lr)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print("Output for [0, 0]:", mlp.forward([0, 0])[-1])
            print("Output for [0, 1]:", mlp.forward([0, 1])[-1])
            print("Output for [1, 0]:", mlp.forward([1, 0])[-1])
            print("Output for [1, 1]:", mlp.forward([1, 1])[-1])
            print("-" * 32)

    print("\nTraining finished\n")
    print("-" * 64)

