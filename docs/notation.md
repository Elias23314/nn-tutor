# This is the master list of notation that the program should be teaching

The network reads MNIST digits: 784 pixels in, two hidden layers of 16 neurons, ten outputs (13,002 numbers in all).
ReLU in both hidden layers, sigmoid outputs.

Loss: the average of ten squared gaps, L = (1/10) Σₖ (ŷₖ − yₖ)², so ∂L/∂ŷₖ = (2/10)(ŷₖ − yₖ).
The goal y is one-hot: 1 for the true digit, 0 for the other nine.

Default η = 2 (the loss averages ten outputs, so each gradient is small). Learning rates offered: 0.5, 2, 10, 40.
One round of practice is a batch of 32 examples; 60,000 training digits make 1,875 rounds per pass. 10,000 test digits.

## Names

| Words (layers 1–3) | Term | Symbol | In code | Meaning |
| --- | --- | --- | --- | --- |
| pixel | input | x | `x` | Pixel value in [0, 1]; 784 per image |
| | | row r, column c | `pixelIndex(r, c)` | 1-based pixel position in the 28 × 28 image |
| neuron a … p | | | `neuronName(1, j)` | Hidden layer 1 |
| neuron A … P | | | `neuronName(2, j)` | Hidden layer 2 |
| the 7 output | output | ŷₖ | `yHat[k]` | Output for digit k |
| goal | target | yₖ | `label` | 1 for the true digit, else 0 |
| strength | weight | w | `weights[j * inSize + i]` | Connection from neuron i to neuron j of the next layer |
| | bias | b | `biases[j]` | Bias of neuron j |
| | weighted sum | z | `z` | z = Σ w·a + b |
| firing | activation | a | `a` | ReLU(z) in hidden layers, σ(z) at the outputs |
| | ReLU's slope | r | `reluDerivative` | 1 if the neuron fires, 0 if it is silent |
| mistake | loss | L | `loss` | Average of ten squared gaps |
| steepness | gradient | ∂L/∂w | `grads` | |
| learning rate | learning rate | η | `eta` | |

## Chain rule for a connection into an output (e.g. G → 7)

∂L/∂w = ∂L/∂ŷ₇ · ∂ŷ₇/∂z₇ · ∂z₇/∂w = (2/10)(ŷ₇ − 1) · ŷ₇(1 − ŷ₇) · a_G

For a pixel's connection into hidden layer 1, the error comes back along 16 × 10 = 160 paths:

∂L/∂w = x · r₁ · Σₘ wₘ rₘ Σₖ wₘₖ ŷₖ(1 − ŷₖ) ∂L/∂ŷₖ

## Starting values

The engine starts from a seeded network (`createNetwork()`, seed 7) with zero biases, so every
learner sees the same starting network. Hidden-layer weights are uniform in ±√(6 / in); output
weights start small, uniform in ±¼√(6 / (in + out)), so no output saturates and gets stuck at 0.

Data: the standard MNIST files live in `public/mnist/` (gzipped) and load with `loadMnist()`.
After one pass (1,875 rounds at η = 2) the seeded network reads 92.7% of the 10,000 test digits;
5 is its weakest digit (0.86).

## Numbers from the mockups

These came from the mockups' own network. They will not match the seeded network until we fix the
starting weights to reproduce them.

For the training 7 before practice: it says 6 (0.67), the 7 output is 0.308, L = 0.27722.
G fires at 0.965 and G → 7 starts at 0.2396. Its gradient is (−0.138)(0.213)(0.965) = −0.0285,
so one step at η = 2 moves it to 0.2965. Nudging to 0.25 gives a measured steepness of −0.02847.
