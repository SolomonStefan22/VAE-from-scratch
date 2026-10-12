#pragma once
#include "matrix.hpp"
#include <cmath>

// softplus(l) = log(1 + e^l)
inline double softplus(double l){
    return (l > 0.0 ? l : 0.0) + std::log1p(std::exp(-std::fabs(l)));
}

inline double bce_loss(const Matrix& logits, const Matrix& X){
    check(logits.rows == X.rows && logits.cols == X.cols, "forme diferite");
    double s = 0.0;
    for(size_t t = 0; t < logits.data.size(); ++t)
        s += softplus(logits.data[t]) - X.data[t] * logits.data[t];
    return s / logits.rows;
}

inline Matrix bce_grad(const Matrix& logits, const Matrix& X){
    check(logits.rows == X.rows && logits.cols == X.cols, "forme diferite");
    Matrix G(logits.rows, logits.cols);
    for(size_t t = 0; t < logits.data.size(); ++t){
        double p = 1.0 / (1.0 + std::exp(-logits.data[t]));
        G.data[t] = (p - X.data[t]) / logits.rows;
    }
    return G;
}

inline Matrix softmax(const Matrix& S) {
    Matrix P(S.rows, S.cols);
    for(size_t i = 0; i < S.rows; ++i) {
        double m = S.data[i * S.cols];
        for(size_t j = 1; j < S.cols; ++j)
            if(S.data[i * S.cols + j] > m) m = S.data[i * S.cols + j];

        double sum = 0.0;
        for(size_t j = 0; j < S.cols; ++j){
            double e = std::exp(S.data[i * S.cols + j] - m);
            P.data[i * S.cols + j] = e;
            sum += e;
        }
        for(size_t j = 0; j < S.cols; ++j)
            P.data[i * S.cols + j] /= sum;
    }
    return P;
}

inline double softmax_ce_loss(const Matrix& S, const std::vector<int>& labels) {
    double total = 0.0;
    for(size_t i = 0; i < S.rows; ++i){
        double m = S.data[i * S.cols];
        for(size_t j = 1; j < S.cols; ++j)
            if(S.data[i * S.cols + j] > m)
                m = S.data[i * S.cols + j];

        double sum = 0.0;
        for(size_t j = 0; j < S.cols; ++j)
            sum += std::exp(S.data[i * S.cols + j] - m);

        double log_sum = m + std::log(sum);
        total += log_sum - S.data[i * S.cols + labels[i]];
    }
    return total / S.rows;
}

inline Matrix softmax_ce_grad(const Matrix& S, const std::vector<int>& labels) {
    Matrix G = softmax(S);
    for(size_t i = 0; i < S.rows; ++i)
        G.data[i * S.cols + labels[i]] -= 1.0;
    return G.scale(1.0 / S.rows);
}