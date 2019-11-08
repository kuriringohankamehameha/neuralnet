#ifndef NN_INFERENCE
#define NN_INFERENCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int dims[3];
    float*** T;
}Tensor;

typedef struct {
    float* biases;
    float**** weights;
    int dims[3];
}Layer;

// All Tensor based operations
Tensor* sigmoid_activation(Tensor* input, int raw_input);
Tensor* relu_activation(Tensor* input, int raw_input);
Tensor* linear_activation(Tensor* input, int raw_input);
Tensor* flattenW(Tensor* input, int raw_input);
Tensor* flattenH(Tensor* input, int raw_input);
Tensor* flattenD(Tensor* input, int raw_input);
Tensor* add(Tensor** input_tensors, int num_tensors, int raw_inputs);
Tensor* average(Tensor** input_tensors, int num_tensors, int raw_inputs);
Tensor* dot(Tensor* input1, Tensor* input2, int raw_input);
Tensor* multiply(Tensor* input1, Tensor* input2, int raw_input);

void print_tensor(Tensor* t);
float**** alloc_4D(int b, int d, int h, int w);
float*** alloc_3D(int d, int h, int w);
void free_tensor(Tensor* t);
Tensor* make_tensor(int d, int h, int w, float*** array);
void free_layer(Layer* layer);

#endif
