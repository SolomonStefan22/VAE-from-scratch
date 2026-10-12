#pragma once
#include "matrix.hpp"
#include <vector>
#include <random>

inline Matrix take_rows(const Matrix& X, const std::vector<size_t>& order, size_t start, size_t count){
    Matrix Bm(count, X.cols);
    for(size_t r = 0; r < count; ++r){
        size_t src = order[start + r];
        for(size_t c = 0; c < X.cols; ++c)
            Bm.data[r * X.cols + c] = X.data[src * X.cols + c];
    }
    return Bm;
}

inline std::vector<int> take_labels(const std::vector<int>& y, const std::vector<size_t>& order, size_t start, size_t count){
    std::vector<int> out(count);
    for(size_t r = 0; r < count; ++r) 
        out[r] = y[order[start + r]];
    return out;
}

inline void shuffle(std::vector<size_t>& order, std::mt19937& rng){
    for(size_t i = order.size() - 1; i > 0; --i){
        std::uniform_int_distribution<size_t> pick(0, i);
        size_t j = pick(rng);
        size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
}