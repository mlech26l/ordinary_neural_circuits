#pragma once
#include "neural_network.h"
#include <math.h>

class NeuralInterfaceInput {
private:
  float MaximumPotential = -20;
  float MinmumPotential = -70;

  float MaximumValue = 1;
  float MinimumValue = 0;

  float value;
  int neuronIndex;

public:
  NeuralInterfaceInput(int nid) {
    neuronIndex = nid;
    value = 0;
  }
  void Reset() { value = 0; }
  void SetMaximumValue(float val) { MaximumValue = val; }
  void BindToNeuralNetwork(int nid) { neuronIndex = nid; }
  void Sync(NeuralNetwork &nn);

  void SetValue(float v) {
    value = (v - MinimumValue) / (MaximumValue - MinimumValue);
  }
};
class NeuralBiInterfaceInput {
private:
  float MaximumPotential = -20;
  float MinmumPotential = -70;

  float MaximumValue = 1;
  float MinimumValue = -1;

  float value;
  int neuronIndexPositiv;
  int neuronIndexNegativ;

public:
  NeuralBiInterfaceInput(int nidPos, int nidNeg) {
    neuronIndexPositiv = nidPos;
    neuronIndexNegativ = nidNeg;
    value = 0;
  }
  void Reset() { value = 0; }
  void BindToNeuralNetwork(int nidPos, int nidNeg) {
    neuronIndexPositiv = nidPos;
    neuronIndexNegativ = nidNeg;
  }
  void SetMinimumValue(float val) { MinimumValue = val; }
  void SetMaximumValue(float val) { MaximumValue = val; }
  void Sync(NeuralNetwork &nn);

  void SetValue(float v) { value = v; }
};

class NeuralBiInterfaceOutput {
private:
  float MaximumPotential = -20;
  float MinmumPotential = -60;

  float MaximumValue = 1;
  float MinimumValue = -1;

  float value;
  int neuronIndexPositiv;
  int neuronIndexNegativ;

public:
  NeuralBiInterfaceOutput(int nidPos, int nidNeg) {
    neuronIndexPositiv = nidPos;
    neuronIndexNegativ = nidNeg;
    value = 0;
  }
  void Reset() { value = 0; }
  void BindToNeuralNetwork(int nidPos, int nidNeg) {
    neuronIndexPositiv = nidPos;
    neuronIndexNegativ = nidNeg;
  }
  void Sync(NeuralNetwork &nn);
  void SetMinimumPotential(int minPotential) { MinmumPotential = minPotential; }
  void SetMaximumPotential(int maxPotential) {
    MaximumPotential = maxPotential;
  }
  void SetMinimumValue(float val) { MinimumValue = val; }
  void SetMaximumValue(float val) { MaximumValue = val; }

  float GetValue() { return value; }
};

class NeuralInterfaceOutput {
private:
  float MaximumPotential = -20;
  float MinmumPotential = -60;

  float MaximumValue = 1;
  float MinimumValue = 0;

  float value;
  int neuronIndex;

public:
  NeuralInterfaceOutput(int nid) {
    neuronIndex = nid;
    value = 0;
  }
  void Reset() { value = 0; }
  void BindToNeuralNetwork(int nid) { neuronIndex = nid; }
  void Sync(NeuralNetwork &nn);
  void SetMaximumValue(float val) { MaximumValue = val; }
  void SetMinimumPotential(int minPotential) { MinmumPotential = minPotential; }
  void SetMaximumPotential(int maxPotential) {
    MaximumPotential = maxPotential;
  }

  float GetValue() { return value; }
};
