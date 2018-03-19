#include "neural_net_serializer.h"
#include "neuron_util.h"
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>

NeuralNetSerializer::NeuralNetSerializer(std::string file) { filename = file; }
NeuralNetwork *NeuralNetSerializer::Deserialize() {
  std::unordered_map<std::string, float> variableTable;
  NeuralNetwork *nn = NULL;
  std::ifstream file;
  file.open(filename, std::ios::in);

  if (!file.good()) {
    std::cerr << "Could not open file '" << filename << "'" << std::endl;
    return NULL;
  }
  int lineNumber = 0;
  while (!file.eof()) {
    lineNumber++;
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                    std::istream_iterator<std::string>{}};

    if (line.length() > 0 && line.c_str()[0] == '#')
      continue;
    if (tokens.size() < 1)
      continue;
    if (tokens[0].compare("size") == 0) {
      if (tokens.size() != 2) {
        std::cerr << "Parse error in line " << lineNumber
                  << ": unexpected number of arguments of 'size' ("
                  << tokens.size() - 1 << ") expected: 1" << std::endl;
        continue;
      }
      int size = std::stoi(tokens[1]);
      nn = new NeuralNetwork(size);
    } else if (tokens[0].compare("cm") == 0) {
      if (tokens.size() != 3) {
        std::cerr << "Parse error in line " << lineNumber
                  << ": unexpected number of arguments of 'cm' ("
                  << tokens.size() - 1 << ") expected: 2" << std::endl;
        continue;
      }
      int neuron = std::stoi(tokens[1]);
      float cm = GetValueOrParse(tokens[2], variableTable);

      nn->SetCmOf(neuron, cm);
    } else if (tokens[0].compare("gleak") == 0) {
      if (tokens.size() != 3) {
        std::cerr << "Parse error in line " << lineNumber
                  << ": unexpected number of arguments of 'gleak' ("
                  << tokens.size() - 1 << ") expected: 2" << std::endl;
        continue;
      }
      int neuron = std::stoi(tokens[1]);
      float gleak = GetValueOrParse(tokens[2], variableTable);

      nn->SetGleakOf(neuron, gleak);
    } else if (tokens[0].compare("vleak") == 0) {
      if (tokens.size() != 3) {
        std::cerr << "Parse error in line " << lineNumber
                  << ": unexpected number of arguments of 'vleak' ("
                  << tokens.size() - 1 << ") expected: 2" << std::endl;
        continue;
      }
      int neuron = std::stoi(tokens[1]);
      float vleak = GetValueOrParse(tokens[2], variableTable);

      nn->SetVleakOf(neuron, vleak);
    } else if (tokens[0].compare("const") == 0) {
      if (tokens.size() != 3) {
        std::cerr << "Parse error in line " << lineNumber
                  << ": unexpected number of arguments of 'const' ("
                  << tokens.size() - 1 << ") expected: 2" << std::endl;
        continue;
      }
      int neuron = std::stoi(tokens[1]);
      float constValue = GetValueOrParse(tokens[2], variableTable);

      nn->AddConstNeuron(neuron, constValue);
    } else if (tokens[0].compare("ex") == 0) {
      if (tokens.size() != 4 && tokens.size() != 5) {
        std::cerr << "Parse error in line " << lineNumber
                  << ": unexpected number of arguments of 'ex' ("
                  << tokens.size() - 1 << ") expected: 3" << std::endl
                  << "~" << line << std::endl;
        continue;
      }
      int src = std::stoi(tokens[1]);
      int dest = std::stoi(tokens[2]);
      float weight = GetValueOrParse(tokens[3], variableTable);

      float sigma = nn->Sigmoid_sigma;
      if (tokens.size() == 5) {
        sigma = GetValueOrParse(tokens[4], variableTable);
      }
      nn->AddExcitatorySynapse(src, dest, weight, sigma);
    } else if (tokens[0].compare("inh") == 0) {
      if (tokens.size() != 4 && tokens.size() != 5) {
        std::cerr << "Parse error in line " << lineNumber
                  << ": unexpected number of arguments of 'in' ("
                  << tokens.size() - 1 << ") expected: 3" << std::endl;
        continue;
      }
      int src = std::stoi(tokens[1]);
      int dest = std::stoi(tokens[2]);
      float weight = GetValueOrParse(tokens[3], variableTable);

      float sigma = nn->Sigmoid_sigma;
      if (tokens.size() == 5) {
        sigma = GetValueOrParse(tokens[4], variableTable);
      }
      nn->AddInhibitorySynapse(src, dest, weight, sigma);
    } else if (tokens[0].compare("gj") == 0) {
      if (tokens.size() != 4) {
        std::cerr << "Parse error in line " << lineNumber
                  << ": unexpected number of arguments of 'ex' ("
                  << tokens.size() - 1 << ") expected: 3" << std::endl;
        continue;
      }
      int src = std::stoi(tokens[1]);
      int dest = std::stoi(tokens[2]);
      float weight = GetValueOrParse(tokens[3], variableTable);

      nn->AddGapJunction(src, dest, weight);
    } else if (tokens[0].compare("def") == 0) {
      if (tokens.size() != 3) {
        std::cerr << "Parse error in line " << lineNumber
                  << ": unexpected number of arguments of 'def' ("
                  << tokens.size() - 1 << ") expected: 2" << std::endl;
        continue;
      }
      std::string name = tokens[1];
      float value = std::stof(tokens[2]);

      variableTable.insert({name, value});
    } else {
      std::cerr << "Unknown token " << tokens[0] << " at line " << lineNumber
                << std::endl;
    }
  }
  file.close();
  return nn;
}
void NeuralNetSerializer::Serialize(NeuralNetwork *nn) {
  std::ofstream file;
  file.open(filename, std::ios::out);
  file << "size " << nn->GetSize() << std::endl;
  for (int i = 0; i < nn->GetSize(); i++) {
    file << "cm " << i << " " << nn->GetCmOf(i) << std::endl;
    file << "gleak " << i << " " << nn->GetGleakOf(i) << std::endl;
  }
  for (int i = 0; i < nn->GetSize(); i++) {
    file << "vleak " << i << " " << nn->GetVleakOf(i) << std::endl;
  }
  std::vector<std::pair<int, float>> &constNeurons = nn->GetConstNeurons();
  for (size_t i = 0; i < constNeurons.size(); i++) {
    file << "const " << constNeurons[i].first << " " << constNeurons[i].second
         << std::endl;
  }
  for (int i = 0; i < nn->GetSize(); i++) {
    std::vector<Synapse> &synapses = nn->GetSynapsesOf(i);
    for (size_t t = 0; t < synapses.size(); t++) {
      if (synapses[t].type == Excitatory) {
        file << "ex ";
      } else if (synapses[t].type == Inhibitory) {
        file << "inh ";
      } else {
        file << "gj ";
      }
      file << synapses[t].source << " " << i << " " << synapses[t].w;
      if (synapses[t].type == Excitatory || synapses[t].type == Inhibitory) {
        file << " " << synapses[t].sigma;
      }
      file << std::endl;
    }
  }
  file.close();
}
