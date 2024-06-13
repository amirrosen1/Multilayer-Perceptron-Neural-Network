#include "Matrix.h"
#include <iostream>
#include <cmath>

#define MIN_VALUE 0.1
#define POWER 2

using std::cout;
using std::endl;

// constructor and default constructor
Matrix::Matrix (int rows, int cols) : intial_dims_{rows, cols}
{
  if (rows <= 0 || cols <= 0)
  {
    throw std::domain_error ("Dimensions are not equal");
  }
  m_vals_ = new float[rows * cols];
  for (int i = 0; i < rows * cols; i++)
  {
    m_vals_[i] = 0;
  }
}

// copy constructor
Matrix::Matrix (const Matrix &other) : intial_dims_{other.intial_dims_}
{
  m_vals_ = new float[other.intial_dims_.rows * other.intial_dims_.cols];
  for (int i = 0; i < other.intial_dims_.rows * other.intial_dims_.cols; i++)
  {
    m_vals_[i] = other.m_vals_[i];
  }
}

// destructor
Matrix::~Matrix ()
{
  delete[] m_vals_;
}

// get rows
int Matrix::get_rows () const
{
  return intial_dims_.rows;
}

// get cols
int Matrix::get_cols () const
{
  return intial_dims_.cols;
}

// transpose the matrix
Matrix &Matrix::transpose ()
{
  int rows = intial_dims_.rows;
  int cols = intial_dims_.cols;
  auto *temp = new float[rows * cols];
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      temp[j * rows + i] = m_vals_[i * cols + j];
    }
  }
  delete[] m_vals_;
  m_vals_ = temp;
  intial_dims_.rows = cols;
  intial_dims_.cols = rows;
  return *this;
}

// vectorize the matrix
Matrix &Matrix::vectorize ()
{
  int rows = intial_dims_.rows;
  int cols = intial_dims_.cols;
  auto *temp = new float[rows * cols];
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      temp[i * cols + j] = m_vals_[i * cols + j];
    }
  }
  delete[] m_vals_;
  m_vals_ = temp;
  intial_dims_.rows = rows * cols;
  intial_dims_.cols = 1;
  return *this;
}

// print the matrix
void Matrix::plain_print () const
{
  int rows = intial_dims_.rows;
  int cols = intial_dims_.cols;
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      cout << m_vals_[i * cols + j] << " ";
    }
    cout << endl;
  }
}

// dot product of two matrices
Matrix Matrix::dot (const Matrix &other) const
{
  int rows = intial_dims_.rows;
  int cols = intial_dims_.cols;
  if (rows != other.intial_dims_.rows || cols != other.intial_dims_.cols)
  {
    throw std::domain_error ("Dimensions are not equal");
  }
  Matrix result (rows, cols);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      result.m_vals_[i * cols + j] =
          m_vals_[i * cols + j] * other.m_vals_[i * cols + j];
    }
  }
  return result;
}

// sum of all elements in the matrix
float Matrix::sum () const
{
  int rows = intial_dims_.rows;
  int cols = intial_dims_.cols;
  float sum = 0;
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      sum += m_vals_[i * cols + j];
    }
  }
  return sum;
}

// return the frobenius norm of the given matrix
float Matrix::norm () const
{
  int rows = intial_dims_.rows;
  int cols = intial_dims_.cols;
  float sum = 0;
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      sum += (float) std::pow (m_vals_[i * cols + j], POWER);
    }
  }
  return std::sqrt (sum);
}

// return the brackets index of the largest element in the matrix
int Matrix::argmax () const
{
  int rows = intial_dims_.rows;
  int cols = intial_dims_.cols;
  float max = m_vals_[0];
  int index = 0;
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      if (m_vals_[i * cols + j] > max)
      {
        max = m_vals_[i * cols + j];
        index = i * cols + j;
      }
    }
  }
  return index;
}

// += operator
Matrix &Matrix::operator+= (const Matrix &other)
{
  int rows = intial_dims_.rows;
  int cols = intial_dims_.cols;
  if (rows != other.intial_dims_.rows || cols != other.intial_dims_.cols)
  {
    throw std::domain_error ("Dimensions are not equal");
  }
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      m_vals_[i * cols + j] += other.m_vals_[i * cols + j];
    }
  }
  return *this;
}

// + operator using +=
Matrix Matrix::operator+ (const Matrix &other) const
{
  if (intial_dims_.rows != other.intial_dims_.rows ||
      intial_dims_.cols != other.intial_dims_.cols)
  {
    throw std::domain_error ("Dimensions are not equal");
  }
  Matrix result (*this);
  result += other;
  return result;
}

// copy_and_swap function
void copy_and_swap (Matrix &first, Matrix &second)
{
  std::swap (first.m_vals_, second.m_vals_);
  std::swap (first.intial_dims_, second.intial_dims_);
}

// copy assignment operator
Matrix &Matrix::operator= (Matrix other)
{
  copy_and_swap (*this, other);
  return *this;
}

// multiply operator between two matrices
Matrix Matrix::operator* (const Matrix &other) const
{
  if (intial_dims_.cols != other.intial_dims_.rows)
  {
    throw std::domain_error ("Dimensions are not equal");
  }
  int rows = intial_dims_.rows;
  int cols = intial_dims_.cols;
  int other_cols = other.intial_dims_.cols;
  Matrix result (rows, other_cols);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < other_cols; ++j)
    {
      float sum = 0;
      for (int k = 0; k < cols; ++k)
      {
        sum += m_vals_[i * cols + k] * other.m_vals_[k * other_cols + j];
      }
      result.m_vals_[i * other_cols + j] = sum;
    }
  }
  return result;
}

// multiply operator between a matrix and a scalar on the right
Matrix Matrix::operator* (float scalar) const
{
  int rows = intial_dims_.rows;
  int cols = intial_dims_.cols;
  Matrix result (rows, cols);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      result.m_vals_[i * cols + j] = m_vals_[i * cols + j] * scalar;
    }
  }
  return result;
}

// multiply operator between a matrix and a scalar on the left using the right
Matrix operator* (float scalar, const Matrix &m)
{
  return m * scalar;
}

// return the i, j element of the matrix
float &Matrix::operator() (int i, int j) const
{
  if (i >= intial_dims_.rows || j >= intial_dims_.cols || i < 0 || j < 0)
  {
    throw std::out_of_range ("Index is out of range");
  }
  return m_vals_[i * intial_dims_.cols + j];
}

// return the i, j element of the matrix, and it can be changed
float &Matrix::operator() (int i, int j)
{
  if (i >= intial_dims_.rows || j >= intial_dims_.cols ||
      i < 0 || j < 0)
  {
    throw std::out_of_range ("Index out of range");
  }
  return m_vals_[i * intial_dims_.cols + j];
}

// return the i'th element of the matrix
float &Matrix::operator[] (int index) const
{
  if (index >= intial_dims_.rows * intial_dims_.cols || index < 0)
  {
    throw std::out_of_range ("Index out of range");
  }
  return m_vals_[index];
}

// return the i'th element of the matrix, and it can be changed
float &Matrix::operator[] (int index)
{
  if (index >= intial_dims_.rows * intial_dims_.cols || index < 0)
  {
    throw std::out_of_range ("Index out of range");
  }
  return m_vals_[index];
}

// output stream operator
std::ostream &operator<< (std::ostream &os, const Matrix &m)
{
  int rows = m.intial_dims_.rows;
  int cols = m.intial_dims_.cols;
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      if (m.m_vals_[i * cols + j] > MIN_VALUE)
      {
        os << "**";
      }
      else
      {
        os << "  ";
      }
    }
    os << endl;
  }
  return os;
}

// input stream operator
std::istream &operator>> (std::istream &is, Matrix &m)
{
  Matrix change (m.get_rows (), m.get_cols ());
  is.read ((char *) (change.m_vals_),
           (int) sizeof (float) * m.get_rows () * m.get_cols ());
  if (is.gcount () < (int) sizeof (float) * m.get_rows () * m.get_cols ())
  {
    throw std::runtime_error ("Input Matrix is too small");
  }
  m = change;
  return is;
}