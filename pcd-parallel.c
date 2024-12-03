#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Definições de constantes
#define GRID_SIZE 2000  // Tamanho da grade
#define NUM_ITER 500    // Número de iterações no tempo
#define DIFF_COEFF 0.1  // Coeficiente de difusão
#define TIME_STEP 0.01
#define SPACE_STEP 1.0

// Função para calcular a equação de difusão
void compute_diffusion(double **current, double **next) {
    for (int t = 0; t < NUM_ITER; t++) {
        // Atualização paralela da grade com OpenMP
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < GRID_SIZE - 1; i++) {
            for (int j = 1; j < GRID_SIZE - 1; j++) {
                next[i][j] = current[i][j] + DIFF_COEFF * TIME_STEP * (
                    (current[i + 1][j] + current[i - 1][j] +
                     current[i][j + 1] + current[i][j - 1] - 4 * current[i][j])
                    / (SPACE_STEP * SPACE_STEP)
                );
            }
        }

        // Atualização das matrizes e cálculo da diferença média
        double avg_diff = 0.0;
        #pragma omp parallel for reduction(+:avg_diff) collapse(2)
        for (int i = 1; i < GRID_SIZE - 1; i++) {
            for (int j = 1; j < GRID_SIZE - 1; j++) {
                avg_diff += fabs(next[i][j] - current[i][j]);
                current[i][j] = next[i][j];
            }
        }

        // Exibição periódica das diferenças médias
        if (t % 100 == 0) {
            printf("Iteração %d - Diferença média: %g\n", t, avg_diff / ((GRID_SIZE - 2) * (GRID_SIZE - 2)));
        }
    }
}

// Função principal
int main() {
    // Alocar memória para as matrizes
    double **current = (double **)malloc(GRID_SIZE * sizeof(double *));
    double **next = (double **)malloc(GRID_SIZE * sizeof(double *));
    if (current == NULL || next == NULL) {
        fprintf(stderr, "Falha na alocação de memória\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < GRID_SIZE; i++) {
        current[i] = (double *)calloc(GRID_SIZE, sizeof(double));
        next[i] = (double *)calloc(GRID_SIZE, sizeof(double));
        if (current[i] == NULL || next[i] == NULL) {
            fprintf(stderr, "Falha na alocação de memória\n");
            return EXIT_FAILURE;
        }
    }

    // Inicializar concentração alta no centro
    current[GRID_SIZE / 2][GRID_SIZE / 2] = 1.0;

    // Definir o número de threads
    omp_set_num_threads(16);

    // Executar iterações e medir tempo
    double start_time = omp_get_wtime();
    compute_diffusion(current, next);
    double end_time = omp_get_wtime();

    // Exibir resultados
    printf("Concentração final no centro: %f\n", current[GRID_SIZE / 2][GRID_SIZE / 2]);
    printf("Tempo de execução (paralelo): %f segundos\n", end_time - start_time);

    // Liberar memória alocada
    for (int i = 0; i < GRID_SIZE; i++) {
        free(current[i]);
        free(next[i]);
    }
    free(current);
    free(next);

    return EXIT_SUCCESS;
}
