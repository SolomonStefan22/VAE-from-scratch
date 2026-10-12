#pragma once
#include "matrix.hpp"
#include <vector>
#include <cstdio>
#include <filesystem>
#include <string>

inline std::string in_results(const char* name){
    std::filesystem::create_directories("results");
    return std::string("results/") + name;
}

inline unsigned char to_byte(double v){
    if(v < 0.0)
        v = 0.0;
    if(v > 1.0)
        v = 1.0;
    return (unsigned char)(v * 255.0 + 0.5);
}

inline void save_grid_pgm(const char* path, const Matrix& M, size_t first, size_t count, size_t per_row){
    const size_t side = 28; 
    check(M.cols == side * side, "randurile nu sunt imagini 28x28");
    check(first + count <= M.rows, "prea putine randuri");

    size_t grid_rows = (count + per_row - 1) / per_row;
    size_t W = per_row * side;
    size_t H = grid_rows * side;

    std::vector<unsigned char> img(W * H, 0);

    for(size_t k = 0; k < count; ++k){
        size_t tr = k / per_row;
        size_t tc = k % per_row;
        for(size_t r = 0; r < side; ++r)
            for(size_t c = 0; c < side; ++c)
                img[(tr * side + r) * W + (tc * side + c)] = to_byte(M.data[(first + k) * M.cols + r * side + c]);
        std::FILE* f = std::fopen(in_results(path).c_str(), "wb");
        check(f != nullptr, "nu pot crea fisierul");
        std::fprintf(f, "P5\n%zu %zu\n255\n", W, H);
        std::fwrite(img.data(), 1, img.size(), f);
        std::fclose(f);
    }
}

inline void print_ascii(const Matrix& M, size_t row){
    const size_t side = 28;
    for(size_t r = 0; r < side; ++r){
        for(size_t c = 0; c < side; ++c){
            double v = M.data[row * M.cols + r * side + c];
            std::putchar(v > 0.5 ? '#' : (v > 0.1 ? '+' : '.'));
        }
        std::putchar('\n');
    }
}

inline void put_pixel(std::vector<unsigned char>& img, size_t S, long row, long col, unsigned char r, unsigned char g, unsigned char b) {
    if (row < 0 || col < 0 || row >= (long)S || col >= (long)S) return;
    size_t k = ((size_t)row * S + (size_t)col) * 3;
    img[k] = r;
    img[k + 1] = g;
    img[k + 2] = b;
}

inline void save_scatter_ppm(const char* path, const Matrix& Z, const std::vector<int>& labels, double lo, double hi, size_t S) {
    check(Z.cols == 2, "save_scatter_ppm: Z trebuie sa aiba 2 coloane");
    check(Z.rows == labels.size(), "save_scatter_ppm: numar diferit de puncte si etichete");

    const unsigned char colors[10][3] = {
        {230,  25,  75},   // 0 roșu
        { 60, 180,  75},   // 1 verde
        {230, 190,   0},   // 2 galben
        {  0, 130, 200},   // 3 albastru
        {245, 130,  48},   // 4 portocaliu
        {145,  30, 180},   // 5 mov
        { 70, 200, 200},   // 6 turcoaz
        {240,  50, 230},   // 7 roz
        {150,  90,  40},   // 8 maro
        {  0,   0,   0}    // 9 negru
    };

    std::vector<unsigned char> img(S * S * 3, 255);

    const double marks[3] = {-3.0, 0.0, 3.0};
    for(int m = 0; m < 3; ++m){
        long col = (long)((marks[m] - lo) / (hi - lo) * (S - 1) + 0.5);
        long row = (long)((hi - marks[m]) / (hi - lo) * (S - 1) + 0.5);
        for(long k = 0; k < (long)S; ++k){
            put_pixel(img, S, k, col, 200, 200, 200);
            put_pixel(img, S, row, k, 200, 200, 200);
        }
    }

    for(size_t r = 0; r < Z.rows; ++r){
        double x = Z.data[r * 2];
        double y = Z.data[r * 2 + 1];
        long col = (long)((x - lo) / (hi - lo) * (S - 1) + 0.5);
        long row = (long)((hi - y) / (hi - lo) * (S - 1) + 0.5);
        int d = labels[r];
        for(long dr = -1; dr <= 1; ++dr)
            for(long dc = -1; dc <= 1; ++dc)
                put_pixel(img, S, row + dr, col + dc, colors[d][0], colors[d][1], colors[d][2]);
    }

    std::FILE* f = std::fopen(in_results(path).c_str(), "wb");
    check(f != nullptr, "save_scatter_ppm: nu pot crea fisierul");
    std::fprintf(f, "P6\n%zu %zu\n255\n", S, S);
    std::fwrite(img.data(), 1, img.size(), f);
    std::fclose(f);
}