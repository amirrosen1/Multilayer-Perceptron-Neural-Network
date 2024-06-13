#ifndef DENSE_H
#define DENSE_H

#include "Activation.h"
#include "Matrix.h"

// typedef for the activation function
typedef Matrix (*func_ptr) (const Matrix &);

class Dense
{
 public:
  Dense (Matrix const &weights, Matrix const &bias,
         func_ptr ActivationFunction_);
  Matrix get_weights () const;
  Matrix get_bias () const;
  func_ptr get_activation () const;
  Matrix operator() (Matrix &input) const;

 private:
  Matrix weights_;
  Matrix biases_;
  func_ptr ActivationFunction;
};

#endif //DENSE_H