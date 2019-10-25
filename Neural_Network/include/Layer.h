#ifndef __LAYER_H
#define __LAYER_H

#include <cmath>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <vector>

#include <cstdlib>

class Layer
{

friend class NeuralNetwork;
public:
	Layer(uint32_t numNeuron, uint32_t prev_numNeuron);
	~Layer(){};

private:
	static double randomWeight()
	{return ((double)rand() / RAND_MAX);};
	uint32_t numNeuron_;
	std::vector<std::vector<double>> weights_;
	std::vector<std::vector<double>> delta_weights_;
	std::vector<double> outputs_;
	std::vector<double> gradient_;
};

#endif