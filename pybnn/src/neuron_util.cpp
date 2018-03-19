#include "neuron_util.h"
#include <math.h>
#include <string>
#include <unordered_map>

float sigmoid(float v_pre, float mu, float sigma) {
  return 1.0f / (1.0f + exp(-sigma * (v_pre - mu)));
}
float bounded_affine(float minx, float miny, float maxx, float maxy,
                     float value) {
  float k = (maxy - miny) / (maxx - minx);
  float d = miny - k * minx;
  float f = k * value + d;
  if (f > maxy)
    f = maxy;
  else if (f < miny)
    f = miny;
  return f;
}
float GetValueOrParse(std::string str,
                      std::unordered_map<std::string, float> &variableTable) {
  if (variableTable.find(str) != variableTable.end()) {
    return variableTable[str];
  }
  return std::stof(str);
}
