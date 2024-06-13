//#include <iostream>
//#include "Matrix.cpp"
//#include "Activation.cpp"
//#include "Dense.cpp"
//#include "MlpNetwork.cpp"
//// read a file into memory
//#include <fstream>
//
//int main ()
//{
//  // test constructor
//  for(int i = 0;i<=3;i++){
//    for(int j=0;j<=3;j++){
//      Matrix a =Matrix(i,j);
//      printf("this is matrix (%d,%d)\n",i,j);
//      a.plain_print();
//    }
//  }
//
//  //copy ctor
//  Matrix a = Matrix(2,3);
//  a(1,1) = 2;
//  Matrix b = Matrix(a);
//  a(1,1)=3;
//  b.plain_print();
//  printf("\n");
//  a.plain_print();
//
//  // getterss
//  const Matrix a_c = Matrix(2,3);
//  printf("%d",a.get_cols());
//  printf("%d",a_c.get_cols());
//  printf("%d",a.get_rows());
//  printf("%d",a_c.get_rows());
//
//  //trasnsposr
//  Matrix a1 =Matrix(2, 2);
//  a(0,1)=2;
//  a.plain_print();
//  a.transpose();
//  a.plain_print();
//
//  Matrix a =Matrix(3,2);
//  a(2,1)=2;
//  a.plain_print();
//  a.transpose();
//  a.plain_print();
//  a.transpose();
//  a.plain_print();
//
//  Matrix a = Matrix(2,2);
//  a(0,0)=1;
//  a(0,1)=2;
//  a(1,0)=3;
//  a(1,1)=4;
//  a.plain_print();
//  a.vectorize();
//  a.plain_print();
//
//  Matrix a = Matrix(3,2);
//  a(0,0)=1;
//  a(0,1)=2;
//  a(1,0)=3;
//  a(1,1)=4;
//  a(2,0)=5;
//  a.plain_print();
//  a.vectorize();
//  a.plain_print();
//
//  Matrix a = Matrix(3,2);
//  a(0,0)=1;
//  a(0,1)=2;
//  a(1,0)=3;
//  a(1,1)=4;
//  a(2,0)=5;
//  Matrix b = Matrix(3,2);
//  b(0,0)=0;
//  b(0,1)=2;
//  b(1,0)=3;
//  b(1,1)=2;
//  b(2,0)=2;
//  a.plain_print();
//  printf("\n_______\n");
//  b.plain_print();
//  printf("\n_______\n");
//  Matrix z = a.dot (b);
//  z.plain_print();
//  printf("sum =31 got  %f \n",z.sum());
//  printf("norm =16.155 got  %f \n",z.norm());
//  printf("argmax index = 4 got  %d \n",z.argmax());
//  z.vectorize().plain_print();
//
//
////addition etc
//  Matrix a = Matrix(3,2);
//  a(0,0)=1;
//  a(0,1)=2;
//  a(1,0)=3;
//  a(1,1)=4;
//  a(2,0)=5;
//  Matrix b = Matrix(3,2);
//  b(0,0)=0;
//  b(0,1)=2;
//  b(1,0)=3;
//  b(1,1)=2;
//  b(2,0)=2;
//  a.plain_print();
//  printf("\n");
//  b.plain_print();
//  printf("\n");
//  (a+b).plain_print();
//  a+=b;
//  printf("\n");
//  a.plain_print();
//  a=b;
//  printf("\n");
//  a.plain_print();
//  printf("\n");
//  b(0,0)=1;
//  b.transpose();
//  b.plain_print();
//  printf("\n");
//
//  (a*b).plain_print();
//  printf("\n");
//
//  (b*a).plain_print();
//  printf("\n");
//  Matrix c = Matrix(3,2);
//  c(0,0)=1;
//  c(0,1)=1;
//  c(1,0)=1;
//  c(1,1)=1;
//  c(2,0)=1;
//  printf("\n");
//
//  c.plain_print();
//  printf("\n");
//
//  c = 3*c;
//  c.plain_print();
//
//  Matrix a = Matrix(3,2);
//  a(0,0)=1;
//  a(0,1)=2;
//  a(1,0)=3;
//  a(1,1)=4;
//  a(2,0)=5;
//  a.plain_print();
//  for(int i=0;i<a.get_rows()*a.get_cols();i++){
//    printf("%d %f\n",i,a[i]);
//  }
//  a.vectorize().plain_print();
//
//
//  std::ifstream is("im0", std::ios::in | std::ios::binary);
//  Matrix a = Matrix(28,28);
//  is>>a;
//  std::cout<<a;
//  return 0;
//}
