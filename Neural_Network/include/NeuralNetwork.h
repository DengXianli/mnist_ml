#ifndef __NEURALNETWORK_H
#define __NEURALNETWORK_H

#include <stdint.h>
#include <vector>
#include <Layer.h>
#include <data_handler.hpp>

class NeuralNetwork
{
	inline static double SigmoidActivationFunctiion(double val)
	{
		return 1.0 / (1.0 + std::exp( -val) );
	}
	inline static double DerivativeActivationFunction(double val)
	{
		return val * (1 - val);
	}
public:
	struct TrainerSettings
    {
        // Learning params
        double      learningRate = 0.001;
        double      momentum = 0.9;
        bool        useBatchLearning = false;

        // Stopping conditions
        uint32_t    maxEpochs = 150;
        double      desiredAccuracy = 90;
    };
	NeuralNetwork(const std::vector<uint32_t> &topology);
	~NeuralNetwork(){};

	void set_trainerSettings(TrainerSettings&);
	void Train(data_handler *);

private:

	void RunEpoch( std::vector<data *>* training_data );
	void GetSetAccuracyAndMSE( std::vector<data *>* dataset, double& accuracy);
	void UpdateWeights();
	void feedForward(std::vector<uint8_t> * inputVals); // data type depends on the inputs
	void backProp(int desire_output);
	int getResult();// for get current Result index from NN output
	uint8_t getResultLabel(int result_ind); // get the label for result index

	//NeuralNetwork settings
	std::vector<Layer> layers_;
	//std::vector<int> outputs_;
	std::vector<uint8_t> output_labels_;
	uint32_t numLayers;
	std::map<uint8_t, int> class_map_;
	std::vector<uint8_t> label_vector_;

	// Training settings
    double                      learningRate_;             // Adjusts the step size of the weight update
    double                      momentum_;                 // Improves performance of stochastic learning (don't use for batch)
    double                      desiredAccuracy_;          // Target accuracy for training
    uint32_t                    maxEpochs_;                // Max number of training epochs
    bool                        useBatchLearning_;         // Should we use batch learning


    // Current training status
    uint32_t                    currentEpoch_;             // Epoch counter
    double                      trainingSetAccuracy_;
    double                      validationSetAccuracy_;
    double                      testSetAccuracy_;
    double                      trainingSetMSE_;
    double                      validationSetMSE_;
    double                      testSetMSE_;

	
};


#endif
