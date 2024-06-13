#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>

class Matrix
{

 public:
  /**
   * @struct dims
   * @brief Matrix dimensions container. Used in MlpNetwork.h and main.cpp
   */
  struct dims
  {
      int rows, cols;
  };

  Matrix (int rows = 1, int cols = 1);
  Matrix (const Matrix &other);
  ~Matrix ();
  int get_rows () const;
  int get_cols () const;
  Matrix &transpose ();
  Matrix &vectorize ();
  void plain_print () const;
  Matrix dot (const Matrix &other) const;
  float sum () const;
  float norm () const;
  int argmax () const;
  Matrix &operator+= (const Matrix &rhs);
  Matrix operator+ (const Matrix &other) const;
  Matrix &operator= (Matrix other);
  friend void copy_and_swap (Matrix &first, Matrix &second);
  Matrix operator* (const Matrix &other) const;
  Matrix operator* (float other) const;
  friend Matrix operator* (float other, const Matrix &m);
  float &operator() (int row, int col) const;
  float &operator() (int row, int col);
  float &operator[] (int index) const;
  float &operator[] (int index);
  friend std::ostream &operator<< (std::ostream &os, const Matrix &m);
  friend std::istream &operator>> (std::istream &is, Matrix &m);

 private:
  float *m_vals_;
  dims intial_dims_{};
};

#endif //MATRIX_H