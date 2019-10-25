#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include "stdint.h"

#include "data.hpp"

class data_handler
{
	std::vector<data *> *data_array;
	std::vector<data *> *training_data;
	std::vector<data *> *test_data;
	std::vector<data *> *validation_data;

	int num_classes;
	int feature_vector_size;
	std::map<uint8_t, int> class_map;
	std::vector<uint8_t> label_vector;

	const double TRAIN_SET_PERCENT = 0.75;
	const double TEST_SET_PERCENT = 0.20;
	const double VALIDATIOIN_SET_PERCENT = 0.05;

public:
	data_handler();
	~data_handler();

	void read_feature_vector(std::string path);
	void read_feature_labels(std::string path);
	void split_data();
	void count_classes();

	uint32_t convert_to_little_endian(const unsigned char* bytes);
	int get_num_classes(){return num_classes;};
	int get_feature_vector_size(){return feature_vector_size;};
	uint8_t get_label(int index);
	std::map<uint8_t, int> get_class_map(){return class_map;};
	std::vector<uint8_t> get_label_vector(){return label_vector;};

	std::vector<data *> * get_training_data();
	std::vector<data *> * get_test_data();
	std::vector<data *> * get_validation_data();
	
};



#endif