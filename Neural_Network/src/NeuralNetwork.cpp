#include "NeuralNetwork.h"
#include <iostream>


NeuralNetwork::NeuralNetwork(const std::vector<uint32_t> &topology)
{
    numLayers = topology.size();
    layers_.push_back(Layer(topology[0], 0));
    for (uint32_t layerNum = 1; layerNum < numLayers; ++layerNum){
        layers_.push_back(Layer(topology[layerNum], topology[layerNum - 1]));
    }
}

void NeuralNetwork::Train(data_handler* dh)
{
    class_map_ = dh->get_class_map();
    label_vector_ = dh->get_label_vector();
    // Reset training state
    currentEpoch_ = 0;
    trainingSetAccuracy_ = 0;
    validationSetAccuracy_ = 0;
    testSetAccuracy_ = 0;
    //trainingSetMSE_ = 0;
    //validationSetMSE_ = 0;
    //testSetMSE_ = 0;

    // Print header
    //-------------------------------------------------------------------------

    std::cout	<< std::endl << " Neural Network Training Starting: " << std::endl
                << "==========================================================================" << std::endl
                << " LR: " << learningRate_ << ", Momentum: " << momentum_ << ", Max Epochs: " << maxEpochs_ << std::endl
                << "==========================================================================" << std::endl << std::endl;
    // Train network using training dataset for training
    //--------------------------------------------------------------------------------------------------------

    while ( (validationSetAccuracy_ < desiredAccuracy_ ) && (currentEpoch_ <= maxEpochs_) )
    {
        // Use training set to train network
        RunEpoch( dh->get_training_data() );

        // Get generalization set accuracy and MSE
        GetSetAccuracyAndMSE( dh->get_validation_data(), validationSetAccuracy_ );

        std::cout << "Epoch :" << currentEpoch_;
        std::cout << " Training Set Accuracy:" << trainingSetAccuracy_; 
        //<< "%, MSE: " << trainingSetMSE_;
        std::cout << " Validation Set Accuracy:" << validationSetAccuracy_ << std::endl;
        //<< "%, MSE: " << validationSetMSE_ << std::endl;

        currentEpoch_++;
    }

    // Get test set accuracy and MSE
    GetSetAccuracyAndMSE( dh->get_test_data(), testSetAccuracy_ );

    // Print validation accuracy and MSE
    std::cout << std::endl << "Training Complete!!! - > Elapsed Epochs: " << currentEpoch_ << std::endl;
    std::cout << " Test Set Accuracy: " << testSetAccuracy_ << std::endl;
    //std::cout << " Validation Set MSE: " << testSetMSE_ << std::endl << std::endl;

}

void NeuralNetwork::RunEpoch( std::vector<data *>* training_data )
{
    int incorrectEntries = 0;
    data* trainingEntry = training_data->at(0);
    // If using batch learning - update the weights
    if ( useBatchLearning_ )
    {
        UpdateWeights();
    }

    //double MSE = 0; placeholder for calculation of MSE;
    //for ( auto trainingEntry : *training_data )
    for(int it = 0; it < training_data->size(); it++)
    {
        data* trainingEntry = training_data->at(it);
        // Feed inputs through network and back propagate errors
        feedForward( trainingEntry->get_feature_vector() );
        backProp( trainingEntry->get_enumerated_label() );
        if( getResult() != trainingEntry->get_enumerated_label() )
        {
            incorrectEntries++;
        }
    }
    
    

    // Update training accuracy and MSE
    trainingSetAccuracy_ = 100.0 - ( (double)incorrectEntries / training_data->size() * 100.0 );
    //m_trainingSetMSE = MSE / ( m_pNetwork->m_numOutputs * trainingSet.size() );
}

void NeuralNetwork::GetSetAccuracyAndMSE( std::vector<data *>* dataset, double& accuracy)
{

    accuracy = 0;

    int numIncorrectResults = 0;
    for ( auto dataEntry : *dataset )
    {
        feedForward( dataEntry->get_feature_vector() );

        if( getResult() != dataEntry->get_enumerated_label() )
        {
            numIncorrectResults++;
        }
    }

    accuracy = 100.0f - ( (double)numIncorrectResults / dataset->size() * 100.0 );
}

void NeuralNetwork::set_trainerSettings(TrainerSettings& trainerSettings)
{
    learningRate_ = trainerSettings.learningRate;
    momentum_ = trainerSettings.momentum;
    useBatchLearning_ = trainerSettings.useBatchLearning;
    maxEpochs_ = trainerSettings.maxEpochs;
    desiredAccuracy_ = trainerSettings.desiredAccuracy;
}

void NeuralNetwork::UpdateWeights()
{
    for(size_t layer_ind = 1; layer_ind < layers_.size();layer_ind++)
    {
        for (size_t input_ind = 0; input_ind < layers_[layer_ind - 1].outputs_.size(); input_ind++)
        {
            for (size_t output_ind = 0; output_ind < layers_[layer_ind].outputs_.size() - 1; output_ind++)
            {
                layers_[layer_ind].weights_[input_ind][output_ind] -= learningRate_ * layers_[layer_ind].delta_weights_[input_ind][output_ind];
                //printf("%8.5f\n", layers_[layer_ind].delta_weights_[input_ind][output_ind] );

                if (useBatchLearning_)
                {
                    layers_[layer_ind].delta_weights_[input_ind][output_ind] = 0;
                }
            }
        }
            
    }
}
void NeuralNetwork::feedForward(std::vector<uint8_t> * inputVals)
{
    for (size_t ind = 0; ind < inputVals->size(); ind++)
    {
        layers_.front().outputs_[ind] = double(inputVals->at(ind));
    }

    for (size_t layer = 1; layer < layers_.size(); layer++)
    {
        std::vector<std::vector<double>>& weights = layers_[layer].weights_;
        for (size_t output_ind = 0; output_ind < layers_[layer].outputs_.size() - 1; output_ind++)
        {
            double sum = 0;
            for (size_t input_ind = 0; input_ind < layers_[layer - 1].outputs_.size(); input_ind++)
            {
                //including the bias at the end of output array
                sum += layers_[layer - 1].outputs_[input_ind] * weights[output_ind][input_ind];
                //printf("%0.4f  ", weights[output_ind][input_ind]);
            }
            

            layers_[layer].outputs_[output_ind] = SigmoidActivationFunctiion(sum);
        }
    }
    /**
    for(size_t ind = 0; ind < layers_.back().outputs_.size(); ind++)
    {
        printf("%0.4f  ", layers_.back().outputs_[ind]);;
    }
    std::cout<<std::endl;
    **/

} 
void NeuralNetwork::backProp(int desire_output)
{
    std::vector<double> desire_output_ary(class_map_.size());
    desire_output_ary[desire_output] = 1;
    //calculate gradient and delta weights for the output layer
    int output_layer = layers_.size() - 1;
    for (size_t output_ind = 0; output_ind < class_map_.size(); output_ind++)
    {
        layers_[output_layer].gradient_[output_ind] = (layers_[output_layer].outputs_[output_ind] - desire_output_ary[output_ind]) 
        * DerivativeActivationFunction(layers_[output_layer].outputs_[output_ind]);
        int input_layer = output_layer - 1;

        //calculate delta weights including with bias
        for (size_t input_ind = 0; input_ind < layers_[input_layer].outputs_.size(); input_ind++)
        {
            if(useBatchLearning_)
            {
                layers_[output_layer].delta_weights_[output_ind][input_ind] += layers_[output_layer].gradient_[output_ind] * layers_[input_layer].outputs_[input_ind];
            }else
            {
                layers_[output_layer].delta_weights_[output_ind][input_ind] = layers_[output_layer].gradient_[output_ind] * layers_[input_layer].outputs_[input_ind]
                + momentum_ * layers_[output_layer].delta_weights_[output_ind][input_ind];
            }
            
        }
    }
    //calculate gradient and delta weights for hidden layers
    for (size_t layer_ind = layers_.size() - 2; layer_ind > 0; layer_ind--)
    {
        
        //calculate gradients for hidden layers
        for (size_t neuron_ind = 0; neuron_ind < layers_[layer_ind].outputs_.size() -1; neuron_ind++)
        {
            double sum = 0;
            //don't need the weight with the bias of the next layer
            for (size_t next_ind = 0; next_ind < layers_[layer_ind + 1].outputs_.size() - 1;next_ind++)
            {

                sum += layers_[layer_ind + 1].weights_[neuron_ind][next_ind] * layers_[layer_ind + 1].gradient_[next_ind];
            }
            layers_[layer_ind].gradient_[neuron_ind] = sum * DerivativeActivationFunction(layers_[layer_ind].outputs_[neuron_ind]);
        }

        //calculate delta weights for hidden layers
        for (size_t output_ind = 0; output_ind < layers_[layer_ind].outputs_.size() -1; output_ind++)
        {
            for (size_t input_ind = 0; input_ind < layers_[layer_ind - 1].outputs_.size(); input_ind++)
            {
                if(useBatchLearning_)
                {
                    layers_[layer_ind].delta_weights_[output_ind][input_ind] += layers_[layer_ind].gradient_[output_ind] * layers_[layer_ind - 1].outputs_[input_ind]; 
                }else
                {
                    layers_[layer_ind].delta_weights_[output_ind][input_ind] = layers_[layer_ind].gradient_[output_ind] * layers_[layer_ind - 1].outputs_[input_ind] 
                    + momentum_ *layers_[layer_ind].delta_weights_[output_ind][input_ind];
                }
                
            }
        }
    }
    
    if ( !useBatchLearning_ )
    {
        UpdateWeights();
    }
}
int NeuralNetwork::getResult()
{

    double max_val = 0;// most big output value from nn for each possibility
    int most_like_ind = 0;
    for (int ind =0; ind < (int)layers_.back().outputs_.size() - 1; ind++)
    {
        if (layers_.back().outputs_[ind] > max_val)
        {
            max_val = layers_.back().outputs_[ind];
            most_like_ind = ind;
        }
    }

    return (most_like_ind);
}

uint8_t NeuralNetwork::getResultLabel(int result_ind)
{
    return label_vector_[result_ind];
}