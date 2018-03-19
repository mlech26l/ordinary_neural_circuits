#include "lifnet.h"
#include <fstream>
LifNet::LifNet(std::string filename) {
  NeuralNetSerializer deserializer(filename);
  nn = deserializer.Deserialize();

  for (int i = 0; i < nn->GetSize(); i++) {
    vleakParameters.push_back(nn->GetValuePointerOfVleak(i));
    vleakBackupValues.push_back(nn->GetVleakOf(i));

    gleakParameters.push_back(nn->GetValuePointerOfGleak(i));
    gleakBackupValues.push_back(nn->GetGleakOf(i));

    cmParameters.push_back(nn->GetValuePointerOfCapacity(i));
    cmBackupValues.push_back(nn->GetCmOf(i));

    std::vector<Synapse> &syns = nn->GetSynapsesOf(i);
    for (unsigned int x = 0; x < syns.size(); x++) {
      parameters.push_back(&syns[x].w);
      backupValues.push_back(syns[x].w);

      sigmaParameters.push_back(&syns[x].sigma);
      sigmaBackupValues.push_back(syns[x].sigma);
    }
  }
}
void LifNet::AddSensoryNeuron(int indx, float input_bound) {
  NeuralInterfaceInput nii(indx);
  nii.SetMaximumValue(input_bound);
  inputs.push_back(nii);
}
void LifNet::AddMotorNeuron(int indx, float output_bound) {
  NeuralInterfaceOutput nio(indx);
  nio.SetMaximumValue(output_bound);
  outputs.push_back(nio);
}
void LifNet::AddBiMotorNeuron(int indxPos, int indxNeg, float min_value,
                              float max_value) {
  NeuralBiInterfaceOutput nio(indxPos, indxNeg);
  nio.SetMinimumValue(min_value);
  nio.SetMaximumValue(max_value);
  biOutputs.push_back(nio);
}
void LifNet::AddBiSensoryNeuron(int indxPos, int indxNeg, float min_value,
                                float max_value) {
  NeuralBiInterfaceInput nio(indxPos, indxNeg);
  nio.SetMinimumValue(min_value);
  nio.SetMaximumValue(max_value);
  biInputs.push_back(nio);
}
void LifNet::DumpState(std::string filename) {
  std::ofstream file;
  file.open(filename, std::ios::app);
  for (int i = 0; i < nn->GetSize(); i++) {
    float pot = nn->GetPotentialOf(i);
    file << pot << ";";
  }
  file << std::endl;
  file.close();
}
void LifNet::DumpClear(std::string filename) {
  std::ofstream file;
  file.open(filename, std::ios::out);
  for (int i = 0; i < nn->GetSize(); i++) {
    file << i << ";";
  }
  file << std::endl;
  file.close();
}
boost::python::list LifNet::Update(boost::python::list &inputsArr, float deltaT,
                                   int simulationSteps) {
  boost::python::list a;

  for (int step = 0; step < simulationSteps; step++) {
    // if (totalTime < 0.1f) {
    //   nn->ForcePotentialOf(0, 0);
    // }
    totalTime += deltaT;
    unsigned int listPtr = 0;
    for (unsigned int i = 0; i < inputs.size(); i++) {
      inputs[i].SetValue(boost::python::extract<float>(inputsArr[listPtr]));
      inputs[i].Sync(*nn);
      listPtr++;
    }
    for (unsigned int i = 0; i < biInputs.size(); i++) {
      biInputs[i].SetValue(boost::python::extract<float>(inputsArr[listPtr]));
      biInputs[i].Sync(*nn);
      listPtr++;
    }
    nn->DoSimulationStep(deltaT);
  }
  for (unsigned int i = 0; i < outputs.size(); i++) {
    outputs[i].Sync(*nn);
    a.append(outputs[i].GetValue());
  }
  for (unsigned int i = 0; i < biOutputs.size(); i++) {
    biOutputs[i].Sync(*nn);
    a.append(biOutputs[i].GetValue());
  }
  return a;
}
void LifNet::Reset() {
  nn->Reset();
  for (unsigned int i = 0; i < outputs.size(); i++) {
    outputs[i].Reset();
  }
  for (unsigned int i = 0; i < inputs.size(); i++) {
    inputs[i].Reset();
  }
  // First run
  totalTime = 0;
}
void LifNet::WriteToFile(std::string filename) {
  NeuralNetSerializer serializer(filename);
  serializer.Serialize(nn);
}
LifNet::~LifNet() { delete nn; }
void LifNet::AddNoise(float variance, int samples) {
  std::normal_distribution<float> norm(0, variance);
  std::uniform_int_distribution<int> dis(0, parameters.size() - 1);
  for (int i = 0; i < samples; i++) {
    int which = dis(gen);
    float x = norm(gen);
    float newValue = (*parameters[which]) + x;
    if (newValue < 0)
      newValue = 0;
    else if (newValue > 3.0f)
      newValue = 3.0f;
    (*parameters[which]) = newValue;
  }
}
void LifNet::AddNoiseSigma(float variance, int samples) {
  std::normal_distribution<float> norm(0, variance);
  std::uniform_int_distribution<int> dis(0, sigmaParameters.size() - 1);
  for (int i = 0; i < samples; i++) {
    int which = dis(gen);
    float x = norm(gen);
    float newValue = (*sigmaParameters[which]) + x;
    if (newValue < 0.05f)
      newValue = 0.05f;
    else if (newValue > 0.5f)
      newValue = 0.5f;
    (*sigmaParameters[which]) = newValue;
  }
}
void LifNet::AddNoiseGleak(float variance, int samples) {
  std::normal_distribution<float> norm(0, variance);
  std::uniform_int_distribution<int> dis(0, gleakParameters.size() - 1);
  for (int i = 0; i < samples; i++) {
    int which = dis(gen);
    float x = norm(gen);
    float newValue = (*gleakParameters[which]) + x;
    if (newValue < 0.05f)
      newValue = 0.05f;
    else if (newValue > 5.0f)
      newValue = 5.0f;
    (*gleakParameters[which]) = newValue;
  }
}
void LifNet::AddNoiseCm(float variance, int samples) {
  std::normal_distribution<float> norm(0, variance);
  std::uniform_int_distribution<int> dis(0, cmParameters.size() - 1);
  for (int i = 0; i < samples; i++) {
    int which = dis(gen);
    float x = norm(gen);
    float newValue = (*cmParameters[which]) + x;
    if (newValue < 0.001)
      newValue = 0.001;
    else if (newValue > 1.0f)
      newValue = 1.0f;
    (*cmParameters[which]) = newValue;
  }
}
void LifNet::AddNoiseVleak(float variance, int samples) {
  std::normal_distribution<float> norm(0, variance);
  std::uniform_int_distribution<int> dis(0, vleakParameters.size() - 1);
  for (int i = 0; i < samples; i++) {
    int which = dis(gen);
    float x = norm(gen);
    float newValue = (*vleakParameters[which]) + x;
    if (newValue < -90.0)
      newValue = -90.0f;
    else if (newValue > -0.0f)
      newValue = 0.0f;
    (*vleakParameters[which]) = newValue;
  }
}
void LifNet::UndoNoise() {
  for (unsigned int i = 0; i < parameters.size(); i++) {
    (*parameters[i]) = backupValues[i];
  }
  for (unsigned int i = 0; i < sigmaParameters.size(); i++) {
    (*sigmaParameters[i]) = sigmaBackupValues[i];
  }
  for (unsigned int i = 0; i < vleakParameters.size(); i++) {
    (*vleakParameters[i]) = vleakBackupValues[i];
  }
  for (unsigned int i = 0; i < gleakParameters.size(); i++) {
    (*gleakParameters[i]) = gleakBackupValues[i];
  }
  for (unsigned int i = 0; i < cmParameters.size(); i++) {
    (*cmParameters[i]) = cmBackupValues[i];
  }
}
void LifNet::CommitNoise() {
  for (unsigned int i = 0; i < parameters.size(); i++) {
    backupValues[i] = (*parameters[i]);
  }
  for (unsigned int i = 0; i < sigmaParameters.size(); i++) {
    sigmaBackupValues[i] = (*sigmaParameters[i]);
  }
  for (unsigned int i = 0; i < vleakParameters.size(); i++) {
    vleakBackupValues[i] = (*vleakParameters[i]);
  }
  for (unsigned int i = 0; i < gleakParameters.size(); i++) {
    gleakBackupValues[i] = (*gleakParameters[i]);
  }
  for (unsigned int i = 0; i < cmParameters.size(); i++) {
    cmBackupValues[i] = (*cmParameters[i]);
  }
}
void LifNet::SeedRandomNumberGenerator(int seed) { gen.seed(seed); }
