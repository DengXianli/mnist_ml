CC=g++
SRC := $(MNIST_ML_ROOT)/Neural_Network/src
CFLAGS := -std=c++11 
INCLUDE_DIR := $(MNIST_ML_ROOT)/Neural_Network/include/
OBJ_DIR := $(MNIST_ML_ROOT)/Neural_Network/obj

all: main

main: $(SRC)/main.cpp objdir $(OBJ_DIR)/NeuralNetwork.o $(OBJ_DIR)/Layer.o
	$(CC) $(CFLAGS) -o main $(SRC)/main.cpp $(OBJ_DIR)/*.o -L$(MNIST_ML_ROOT)/lib/ -ldata -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include
	rm -r $(OBJ_DIR)

objdir:
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/NeuralNetwork.o: $(SRC)/NeuralNetwork.cpp objdir $(OBJ_DIR)/Layer.o
	$(CC) -fPIC $(CFLAGS) -o $(OBJ_DIR)/NeuralNetwork.o -L$(MNIST_ML_ROOT)/lib/ -ldata -I$(MNIST_ML_ROOT)/include \
	-I$(INCLUDE_DIR) -c $(SRC)/NeuralNetwork.cpp

$(OBJ_DIR)/Layer.o: $(SRC)/Layer.cpp objdir
	$(CC) -fPIC $(CFLAGS) -o $(OBJ_DIR)/Layer.o -L$(MNIST_ML_ROOT)/lib/ -ldata -I$(MNIST_ML_ROOT)/include \
	-I$(INCLUDE_DIR) -c $(SRC)/Layer.cpp

clean:
	rm main
	rm -r obj
	
	