#include "Activation.h"
#include <cmath>

namespace activation
{
    /**
     * ReLU activation function
     * @param m the matrix
     * @return the new updated matrix
     */
    Matrix relu (const Matrix &m)
    {
      Matrix new_matrix (m.get_rows (), m.get_cols ());
      for (int i = 0; i < m.get_rows (); i++)
      {
        for (int j = 0; j < m.get_cols (); j++)
        {
          if (m (i, j) < 0)
          {
            new_matrix (i, j) = 0;
          }
          else
          {
            new_matrix (i, j) = m (i, j);
          }
        }
      }
      return new_matrix;
    }

    /**
     * Softmax activation function
     * @param m the matrix
     * @return the new updated matrix
     */
    Matrix softmax (const Matrix &m)
    {
      Matrix new_matrix (m.get_rows (), m.get_cols ());
      float sum = 0;
      for (int i = 0; i < m.get_rows (); i++)
      {
        for (int j = 0; j < m.get_cols (); j++)
        {
          sum += std::exp (m (i, j));
        }
      }
      for (int i = 0; i < m.get_rows (); i++)
      {
        for (int j = 0; j < m.get_cols (); j++)
        {
          new_matrix (i, j) = std::exp (m (i, j)) / sum;
        }
      }
      return new_matrix;
    }
}