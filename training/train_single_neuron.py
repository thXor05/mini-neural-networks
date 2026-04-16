# train_single_neuron.py

from models.single_neuron import SingleNeuron

epochs = 1000
lr = 0.1

def train_single_neuron():
    single_neuron = SingleNeuron()

    print("\nTraining started...")
    
    for epoch in range(epochs):
        single_neuron.backward(0, 0, lr)
        single_neuron.backward(1, 1, lr)

        print(f"Epoch {epoch}/{epochs}")
        print(f"w = {single_neuron.w}, b = {single_neuron.b}")
        print("Output for 0:", single_neuron.forward(0))
        print("Output for 1:", single_neuron.forward(1))
        print("-" * 32)

    print("\nTraining finished\n")
    print("-" * 64)

