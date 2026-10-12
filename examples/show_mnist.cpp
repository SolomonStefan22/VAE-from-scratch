#include "matrix.hpp"
#include "mnist.hpp"
#include "image.hpp"
#include <cstdio>

int main(){
    Matrix X = load_mnist_images("data/train-images-idx3-ubyte");
    std::vector<int> y = load_mnist_labels("data/train-labels-idx1-ubyte");
    check(X.rows == y.size(), "imagini si etichete in numar diferit");
    std::printf("Imagini: %zu, pixeli per imagine: %zu\n\n", X.rows, X.cols);

    std::printf("Prima imagine, eticheta %d:\n", y[0]);
    print_ascii(X, 0);

    std::printf("\nEtichetele primelor 32: ");
    for (int k = 0; k < 32; ++k) std::printf("%d ", y[k]);
    std::printf("\n");
    save_grid_pgm("first32.pgm", X, 0, 32, 8);
    std::printf("Am salvat first32.pgm\n\n");

    size_t near0 = 0, near1 = 0, middle = 0;
    for(size_t t = 0; t < X.data.size(); ++t){
        double v = X.data[t];
        if (v < 0.1) near0++;
        else if (v > 0.9) near1++;
        else middle++;
    }
    double total = (double)X.data.size();
    std::printf("Pixeli sub 0.1:  %.1f%%\n", 100.0 * near0 / total);
    std::printf("Pixeli peste 0.9: %.1f%%\n", 100.0 * near1 / total);
    std::printf("Pixeli intre:    %.1f%%\n", 100.0 * middle / total);

    return 0;
}