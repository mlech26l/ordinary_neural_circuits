#include "NeuralInterface.h"
#include "neuron_util.h"
#include <math.h>

void NeuralInterfaceInput::Sync(NeuralNetwork &nn) {
  float potential =
      (MaximumPotential - MinmumPotential) * value + MinmumPotential;

  if (neuronIndex >= 0)
    nn.ForcePotentialOf(neuronIndex, potential);
}

void NeuralInterfaceOutput::Sync(NeuralNetwork &nn) {
  if (neuronIndex >= 0) {
    float potential = nn.GetPotentialOf(neuronIndex);
    value = bounded_affine(MinmumPotential, MinimumValue, MaximumPotential,
                           MaximumValue, potential);
  }
}

void NeuralBiInterfaceInput::Sync(NeuralNetwork &nn) {
  if (value >= 0) {
    float corValue = value / MaximumValue;
    float potential =
        (MaximumPotential - MinmumPotential) * corValue + MinmumPotential;
    if (neuronIndexPositiv >= 0)
      nn.ForcePotentialOf(neuronIndexPositiv, potential);
    if (neuronIndexNegativ >= 0)
      nn.ForcePotentialOf(neuronIndexNegativ, MinmumPotential);
  } else {
    float corValue = value / (-MinimumValue);
    float potential =
        (MaximumPotential - MinmumPotential) * -corValue + MinmumPotential;
    if (neuronIndexNegativ >= 0)
      nn.ForcePotentialOf(neuronIndexNegativ, potential);
    if (neuronIndexPositiv >= 0)
      nn.ForcePotentialOf(neuronIndexPositiv, MinmumPotential);
  }
}

void NeuralBiInterfaceOutput::Sync(NeuralNetwork &nn) {
  float res_value = 0;
  if (neuronIndexNegativ >= 0) {
    float negPotential = nn.GetPotentialOf(neuronIndexNegativ);
    res_value -= bounded_affine(MinmumPotential, 0, MaximumPotential,
                                -MinimumValue, negPotential);
  }
  if (neuronIndexPositiv >= 0) {
    float negPotential = nn.GetPotentialOf(neuronIndexPositiv);
    res_value += bounded_affine(MinmumPotential, 0, MaximumPotential,
                                MaximumValue, negPotential);
  }
  value = res_value;
}
