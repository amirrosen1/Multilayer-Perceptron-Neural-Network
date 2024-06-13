#include "MlpNetwork.h"
#define FIRST_LAYER 128
#define SECOND_LAYER 64
#define THIRD_LAYER 20
#define FINAL_LAYER 10
#define DIMENSION_OF_COL 1

// constructor
MlpNetwork::MlpNetwork (Matrix const weights[], Matrix const biases[])
    : layer_1{weights[0], biases[0],
              activation::relu},
      layer_2{weights[1], biases[1],
              activation::relu},
      layer_3{weights[2], biases[2],
              activation::relu},
      layer_4{weights[3], biases[3],
              activation::softmax}
{
  for (int i = 1; i < MLP_SIZE; i++)
  {
    if (weights[i].get_cols() != weights[i - 1].get_rows())
    {
      throw std::domain_error("The dimensions of the weights are not correct");
    }
  }
}

// operator()- applies the entire network on input and returns digit struct
digit MlpNetwork::operator() (const Matrix &input)
{
  Matrix vector = input;
  vector.vectorize ();
  Matrix vector_first (FIRST_LAYER, DIMENSION_OF_COL);
  vector_first = layer_1 (vector);
  Matrix vector_second (SECOND_LAYER, DIMENSION_OF_COL);
  vector_second = layer_2 (vector_first);
  Matrix vector_third (THIRD_LAYER, DIMENSION_OF_COL);
  vector_third = layer_3 (vector_second);
  Matrix vector_final (FINAL_LAYER, DIMENSION_OF_COL);
  vector_final = layer_4 (vector_third);
  digit result;
  result.value = vector_final.argmax ();
  result.probability = vector_final[(int) result.value];
  return result;
}