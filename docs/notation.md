# This is the master list of notation that the program should be teaching

Loss: L = (ŷ − y)², so ∂L/∂ŷ = 2(ŷ − y). ReLU in the hidden layer, linear output. Default η = 0.1.

## Names

| Symbol | In code | Meaning |
| --- | --- | --- |
| x₁, x₂ | `x1`, `x2` | Inputs |
| h1 to h4 | `h1` … `h4` | Hidden neurons |
| ŷ, y | `yHat`, `y` | Output, target |
| wᵢⱼ (e.g. w₁₃) | `w13` | Weight from input i to hidden neuron j |
| bⱼ | `b3` | Bias of hidden neuron j |
| vⱼ | `v3` | Weight from hidden neuron j to the output |
| c | `c` | Output bias |
| z, a | `z`, `a` | Weighted sum, activation output |
| η | `eta` | Learning rate |

## Starting values

| Neuron | From x₁ | From x₂ | Bias | To output |
| --- | --- | --- | --- | --- |
| h1 | w₁₁ = 0.6 | w₂₁ = −0.4 | b₁ = 0.1 | v₁ = 0.5 |
| h2 | w₁₂ = −0.5 | w₂₂ = 0.8 | b₂ = 0.2 | v₂ = −0.6 |
| h3 | w₁₃ = 0.9 | w₂₃ = 0.3 | b₃ = −0.1 | v₃ = 0.4 |
| h4 | w₁₄ = 0.2 | w₂₄ = −0.7 | b₄ = 0.05 | v₄ = 0.3 |
| Output | | | c = 0.1 | |

## Numbers to check against

For x = (1, 0), y = 1: ŷ = 0.845, L = 0.024025, ∂L/∂w₁₃ = −0.124.
One step at η = 0.1 moves w₁₃ from 0.9 to 0.9124, and the loss drops to about 0.003.