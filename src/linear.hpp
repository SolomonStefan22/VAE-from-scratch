#pragma once
#include "matrix.hpp"
#include "adam.hpp"
#include <random>
#include <cmath>

class Linear{
    public:
        Matrix W;
        Matrix b;
        Matrix gradW;
        Matrix gradb;
        Matrix X_saved;
        AdamParam adamW;
        AdamParam adamb;

        Linear(size_t n_in, size_t n_out, std::mt19937& rng){
            W = Matrix(n_in, n_out);
            b = Matrix(1, n_out);
            gradW = Matrix(n_in, n_out);
            gradb = Matrix(1, n_out);
            adamW = AdamParam(n_in, n_out);
            adamb = AdamParam(1, n_out);

            std::normal_distribution<double> normal(0.0, std::sqrt(2.0 / n_in));
            for(size_t t = 0; t < W.data.size(); ++t)
                W.data[t] = normal(rng);
        }

        Matrix forward(const Matrix&X){
            check(X.cols == W.rows, "X are numar gresit de coloane");
            X_saved = X;
            return X.matmul(W).add_row(b);
        }

        Matrix backward(const Matrix& dY){
            check(dY.rows == X_saved.rows && dY.cols == W.cols, "dY are forma gresita");
            gradW = X_saved.transpose().matmul(dY);
            gradb = dY.sum_rows();
            return dY.matmul(W.transpose());
        }

        void sgd_step(double lr){
            W = W.sub(gradW.scale(lr));
            b = b.sub(gradb.scale(lr));
        }

        void adam_step(double lr, int t){
            adamW.step(W, gradW, lr, t);
            adamb.step(b, gradb, lr, t);
        }
};