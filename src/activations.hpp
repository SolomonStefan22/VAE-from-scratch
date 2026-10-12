#pragma once
#include "matrix.hpp"
#include <cmath>

class ReLU{
    public:
        Matrix A_saved;

        Matrix forward(const Matrix&A){
            A_saved = A;
            Matrix H(A.rows, A.cols);
            for(size_t t = 0; t < A.data.size(); ++t)
                H.data[t] = A.data[t] > 0.0 ? A.data[t] : 0.0;
            return H;
        }

        Matrix backward(const Matrix& dH){
            check(dH.rows == A_saved.rows && dH.cols == A_saved.cols, "dH are forma gresita");
            Matrix dA(dH.rows, dH.cols);
            for(size_t t = 0; t < dH.data.size(); ++t)
                dA.data[t] = A_saved.data[t] > 0.0 ? dH.data[t] : 0.0;
            return dA;
        }
};

inline Matrix sigmoid(const Matrix& A){
    Matrix S(A.rows, A.cols);
    for(size_t t = 0; t < A.data.size(); ++t)
        S.data[t] = 1.0 / (1.0 + std::exp(-A.data[t]));
    return S;
}