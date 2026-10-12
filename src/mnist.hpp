#pragma once
#include "matrix.hpp"
#include <vector>
#include <cstdio>

inline unsigned int read_u32_be(std::FILE* f){
    unsigned char b[4];
    size_t got = std::fread(b, 1, 4, f);
    check(got == 4, "read_u32_be: fisier trunchiat");
    return ((unsigned int)b[0] << 24) | ((unsigned int)b[1] << 16) | ((unsigned int)b[2] << 8)  |  (unsigned int)b[3];
}

inline Matrix load_mnist_images(const char* path){
    std::FILE* f = std::fopen(path, "rb");
    check(f != nullptr, "nu pot deschide fisierul");

    unsigned int magic = read_u32_be(f);
    check(magic == 2051, "nu e un fisier de imagini");
    unsigned int n = read_u32_be(f);
    unsigned int r = read_u32_be(f);
    unsigned int c = read_u32_be(f);

    size_t pixels = (size_t)r * c;
    std::vector<unsigned char> buf((size_t)n * pixels);
    size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    check(got == buf.size(), "fisier trunchiat");

    Matrix X(n, pixels);
    for(size_t t = 0; t < buf.size(); ++t)
        X.data[t] = buf[t] / 255.0;
    
    return X;
}

inline std::vector<int> load_mnist_labels(const char* path){
    std::FILE *f = std::fopen(path, "rb");
    check(f != nullptr, "nu pot deschide fisierul");

    unsigned int magic = read_u32_be(f);
    check(magic == 2049, "nu e fisier de etichete");
    unsigned int n = read_u32_be(f);

    std::vector<unsigned char> buf(n);
    size_t got = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    check(got == buf.size(), "fisier trunchiat");

    std::vector<int> labels(n);
    for(size_t t = 0; t < n; ++t)
        labels[t] = buf[t];
    
    return labels;
}