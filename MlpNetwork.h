//MlpNetwork.h

#ifndef MLPNETWORK_H
#define MLPNETWORK_H

#include "Dense.h"

#define MLP_SIZE 4

/**
 * @struct digit
 * @brief Identified (by Mlp network) digit with
 *        the associated probability.
 * @var value - Identified digit value
 * @var probability - identification probability
 */
typedef struct digit
{
    unsigned int value;
    float probability;
} digit;

const Matrix::dims img_dims = {28, 28};
const Matrix::dims weights_dims[] = {{128, 784},
                                     {64,  128},
                                     {20,  64},
                                     {10,  20}};
const Matrix::dims bias_dims[] = {{128, 1},
                                  {64,  1},
                                  {20,  1},
                                  {10,  1}};

class MlpNetwork
{
 public:
  MlpNetwork (Matrix const wights[], Matrix const biases[]);
  digit operator() (const Matrix &input);

 private:
  // the layers of the network (4 layers)
  Dense layer_1;
  Dense layer_2;
  Dense layer_3;
  Dense layer_4;
};

#endif // MLPNETWORK_H