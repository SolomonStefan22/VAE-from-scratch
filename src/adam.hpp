#pragma once
#include "matrix.hpp"
#include <cmath>

class AdamParam{
    public:
        Matrix m;
        Matrix v;

        AdamParam() {}

        AdamParam(size_t rows, size_t cols){
            m = Matrix(rows, cols);
            v = Matrix(rows, cols);
        }

        void step(Matrix& param, const Matrix& grad, double lr, int t) {
            check(param.rows == grad.rows && param.cols == grad.cols, "orectura are alta forma decat parametrul");
            const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
            double c1 = 1.0 - std::pow(b1, t);
            double c2 = 1.0 - std::pow(b2, t);

            for(size_t k = 0; k < param.data.size(); ++k){
                double g = grad.data[k];
                m.data[k] = b1 * m.data[k] + (1.0 - b1) * g;
                v.data[k] = b2 * v.data[k] + (1.0 - b2) * g * g;
                double m_hat = m.data[k] / c1;
                double v_hat = v.data[k] / c2;
                param.data[k] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
        
};