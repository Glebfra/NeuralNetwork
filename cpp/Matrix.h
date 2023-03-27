class Matrix {
public:
    Matrix(int cols_number, int rows_number);

    const int cols_number;
    const int rows_number;

    Matrix& operator + (Matrix s);
    Matrix& operator += (Matrix s);
    Matrix& operator * (Matrix s);
    Matrix& operator *= (Matrix s);
    Matrix& operator / (Matrix s);
    Matrix& operator /= (Matrix s);

    void matmul(Matrix second);

};