#include "Dense.h"

// constructor
Dense::Dense (Matrix const &weights, Matrix const &bias,
              func_ptr ActivationFunction_)
    : weights_{weights}, biases_{bias}, ActivationFunction{ActivationFunction_}
{
  if (bias.get_cols () != 1 || weights.get_rows () != bias.get_rows ())
  {
    throw std::domain_error ("Invalid dimensions");
  }
}

// get weights
Matrix Dense::get_weights () const
{
  return weights_;
}

// get bias
Matrix Dense::get_bias () const
{
  return biases_;
}

// get activation function
func_ptr Dense::get_activation () const
{
  return ActivationFunction;
}

// operator()- applies the layer on input and returns the output matrix
Matrix Dense::operator() (Matrix &input) const
{
  Matrix output = weights_ * input;
  output = output + biases_;
  output = ActivationFunction (output);
  return output;
}