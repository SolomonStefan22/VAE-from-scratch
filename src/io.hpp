#pragma once
#include "matrix.hpp"
#include "linear.hpp"
#include "classifier.hpp"
#include "vae.hpp"
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <string>

inline std::string in_models(const char* name){
    std::filesystem::create_directories("models");
    return std::string("models/") + name;
}

inline void write_matrix(std::FILE* f, const Matrix& M){
    uint64_t r = M.rows, c = M.cols;
    std::fwrite(&r, sizeof(r), 1, f);
    std::fwrite(&c, sizeof(c), 1, f);
    std::fwrite(M.data.data(), sizeof(double), M.data.size(), f);
}

inline void read_matrix_into(std::FILE* f, Matrix& M){
    uint64_t r = 0, c = 0;
    check(std::fread(&r, sizeof(r), 1, f) == 1 && std::fread(&c, sizeof(c), 1, f) == 1, "fisier trunchiat");
    check(r == M.rows && c == M.cols, "dimensiunile din fisier difera de ale modelului");
    size_t got = std::fread(M.data.data(), sizeof(double), M.data.size(), f);
    check(got == M.data.size(), "fisier trunchiat");
}

inline void write_linear(std::FILE* f, const Linear& L){
    write_matrix(f, L.W);
    write_matrix(f, L.b);
}

inline void read_linear(std::FILE* f, Linear& L){
    read_matrix_into(f, L.W);
    read_matrix_into(f, L.b);
}

inline void save_classifier(const char* path, const Classifier& clf){
    std::FILE* f = std::fopen(in_models(path).c_str(), "wb");
    check(f != nullptr, "nu pot crea fisierul");
    write_linear(f, clf.l1);
    write_linear(f, clf.l2);
    std::fclose(f);
}

inline void load_classifier(const char* path, Classifier& clf){
    std::FILE* f = std::fopen(in_models(path).c_str(), "wb");
    check(f != nullptr, "nu gasesc fisierul");
    read_linear(f, clf.l1);
    read_linear(f, clf.l2);
    std::fclose(f);
}

inline void save_vae(const char* path, const VAE& vae){
    std::FILE* f = std::fopen(in_models(path).c_str(), "wb");
    check(f != nullptr, "nu pot crea fisierul");
    uint64_t D = vae.D;
    std::fwrite(&D, sizeof(D), 1, f);
    write_linear(f, vae.enc1);
    write_linear(f, vae.enc_mu);
    write_linear(f, vae.enc_v);
    write_linear(f, vae.dec1);
    write_linear(f, vae.dec2);
    std::fclose(f);
}

inline void load_vae(const char* path, VAE& vae){
    std::FILE* f = std::fopen(in_models(path).c_str(), "wb");
    check(f != nullptr, "nu gasesc fisierul");
    uint64_t D = 0;
    check(std::fread(&D, sizeof(D), 1, f) == 1, "fisier trunchiat");
    check(D == vae.D, "dimensiunea codului din fisier difera de a modelului");
    read_linear(f, vae.enc1);
    read_linear(f, vae.enc_mu);
    read_linear(f, vae.enc_v);
    read_linear(f, vae.dec1);
    read_linear(f, vae.dec2);
    std::fclose(f);
}
