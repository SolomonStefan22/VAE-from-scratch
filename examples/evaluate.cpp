#include "matrix.hpp"
#include "mnist.hpp"
#include "image.hpp"
#include "loss.hpp"
#include "activations.hpp"
#include "classifier.hpp"
#include "vae.hpp"
#include "io.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

void report(const char* title, const Matrix& P){
    size_t N = P.rows;
    double sum_conf = 0.0;
    size_t sure = 0;
    std::vector<size_t> counts(10, 0);

    for(size_t i = 0; i < N; ++i){
        size_t best = 0;
        for(size_t k = 1; k < 10; ++k)
            if(P.data[i * 10 + k] > P.data[i * 10 + best]) best = k;
        double conf = P.data[i * 10 + best];
        sum_conf += conf;
        if(conf > 0.9) sure++;
        counts[best]++;
    }

    double H = 0.0;
    for(size_t k = 0; k < 10; ++k){
        if(counts[k] == 0) continue;
        double q = (double)counts[k] / N;
        H -= q * std::log(q);
    }

    std::printf("%s\n", title);
    std::printf("  increderea medie:          %.3f\n", sum_conf / N);
    std::printf("  cifre clare (peste 0,9):   %.1f%%\n", 100.0 * sure / N);
    std::printf("  cate din fiecare cifra:   ");
    for(size_t k = 0; k < 10; ++k) 
        std::printf(" %zu:%zu", k, counts[k]);
    std::printf("\n  numarul efectiv de cifre:  %.2f (din 10)\n\n", std::exp(H));
}

int main(int argc, char** argv){
    size_t D = 2;
    if(argc > 1) D = (size_t)std::atoi(argv[1]);

    std::mt19937 rng(1);
    VAE vae(D, rng);
    char name[64];
    std::snprintf(name, sizeof(name), "vae_d%zu.bin", D);
    load_vae(name, vae);
    Classifier clf(rng);
    load_classifier("classifier.bin", clf);
    std::printf("Am incarcat %s si classifier.bin\n\n", name);

    const size_t N = 1000;

    Matrix X_test = load_mnist_images("data/t10k-images-idx3-ubyte");
    Matrix real(N, 784);
    for (size_t t = 0; t < N * 784; ++t) real.data[t] = X_test.data[t];
    report("Imagini reale (reperul):", softmax(clf.forward(real)));

    std::normal_distribution<double> normal(0.0, 1.0);
    Matrix Z(N, D);
    for (size_t t = 0; t < Z.data.size(); ++t) Z.data[t] = normal(rng);
    Matrix G = sigmoid(vae.decode(Z));
    report("Cifre generate de VAE:", softmax(clf.forward(G)));

    std::snprintf(name, sizeof(name), "eval_generate_d%zu.pgm", D);
    save_grid_pgm(name, G, 0, 64, 8);
    return 0;
}