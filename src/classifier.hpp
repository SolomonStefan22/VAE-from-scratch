#pragma once
#include "linear.hpp"
#include "activations.hpp"
#include <random>
#include <vector>

class Classifier {
public:
    Linear l1;
    Linear l2;
    ReLU relu;

    Classifier(std::mt19937& rng) : l1(784, 400, rng), l2(400, 10, rng) {}

    Matrix forward(const Matrix& X){
        Matrix a = l1.forward(X);
        Matrix h = relu.forward(a);
        return l2.forward(h);
    }

    void backward(const Matrix& d_scores){
        Matrix dh = l2.backward(d_scores);
        Matrix da = relu.backward(dh);
        l1.backward(da);
    }

    void adam_step(double lr, int t){
        l1.adam_step(lr, t);
        l2.adam_step(lr, t);
    }
};

inline std::vector<int> argmax_rows(const Matrix& S){
    std::vector<int> out(S.rows);
    for(size_t i = 0; i < S.rows; ++i){
        size_t best = 0;
        for(size_t j = 1; j < S.cols; ++j)
            if(S.data[i * S.cols + j] > S.data[i * S.cols + best]) best = j;
        out[i] = (int)best;
    }
    return out;
}