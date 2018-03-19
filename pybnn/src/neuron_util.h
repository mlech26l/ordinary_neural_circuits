#ifndef NEURON_UTIL_H_
#define NEURON_UTIL_H_
#include <string>
#include <unordered_map>
float sigmoid(float v_pre, float mu, float sigma);

float bounded_affine(float minx, float miny, float maxx, float maxy,
                     float value);
float GetValueOrParse(std::string str,
                      std::unordered_map<std::string, float> &variableTable);
#endif
