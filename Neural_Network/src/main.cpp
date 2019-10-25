#include "NeuralNetwork.h"
#include "cmdParser.h"
#include "data_handler.hpp"
#include <iostream>


int main( int argc, char* argv[] )
{
    cli::Parser cmdParser( argc, argv );
    cmdParser.set_required<std::string>( "ff", "FeatureFile", "Path to training data feature file." );
    cmdParser.set_required<std::string>( "lf", "LabelFile", "Path to training data label file." );
    cmdParser.set_required<uint32_t>( "hlayer", "NumHiddenLayer", "Num hidden layers." );
    cmdParser.set_required<uint32_t>( "n", "NumNeuron", "Num neurons in each layers." );

    if ( !cmdParser.run() )
    {
        std::cout << "Invalid command line arguments";
        return 1;
    }

    std::string trainingDataPath_feature = cmdParser.get<std::string>( "ff" ).c_str();
    std::string trainingDataPath_label = cmdParser.get<std::string>( "lf" ).c_str();
    uint32_t const numHiddenLayer = cmdParser.get<uint32_t>( "hlayer" );
    uint32_t const numNeuron = cmdParser.get<uint32_t>( "n" );

    data_handler *dh = new data_handler();
	dh->read_feature_vector(trainingDataPath_feature);
	dh->read_feature_labels(trainingDataPath_label);
	dh->split_data();
	dh->count_classes();

    // Create neural network
    std::vector<uint32_t> topology;
    topology.push_back(dh->get_feature_vector_size());
    for(uint32_t i =0; i < numHiddenLayer; i++){
    	topology.push_back(numNeuron);
    }
    topology.push_back(dh->get_num_classes());
    NeuralNetwork nn(topology);
    std::cout << std::endl << "Neural Network Training Setting: " << std::endl;
    std::cout << "Number of Layers: "<< topology.size() << std::endl;
    std::cout << "Topology of Network: " << std::endl;
    for(int i = 0; i < topology.size(); i++)
    {
        std::cout<<(int)topology[i]<<" ";
    }
    std::cout<<std::endl;


    // Create neural network trainer
    NeuralNetwork::TrainerSettings trainerSettings;
    trainerSettings.learningRate = 0.001;
    trainerSettings.momentum = 0.0;
    trainerSettings.useBatchLearning = false;
    trainerSettings.maxEpochs = 200;
    trainerSettings.desiredAccuracy = 90;

    nn.set_trainerSettings(trainerSettings);
    nn.Train(dh);

    return 0;
}