#include <Layer.h>

Layer::Layer(uint32_t numNeuron, uint32_t prev_numNeuron)
{
	numNeuron_ = numNeuron;	
	for (uint32_t i = 0; i < numNeuron; i++){
		
		outputs_.push_back(0);
		gradient_.push_back(0);
	}


	for (uint32_t i = 0; i <= prev_numNeuron; i++)// including the bias
	{
		delta_weights_.push_back(std::vector<double>(numNeuron + 1));//including bias

		//initialize weights randomly
		std::vector<double> neuron_weights;
		for(uint32_t j = 0; j < numNeuron + 1; j++)
		{
			neuron_weights.push_back(randomWeight());
		}
		weights_.push_back(neuron_weights);
	}
	outputs_.push_back(1.0);// push bach a bias for each layer

}
