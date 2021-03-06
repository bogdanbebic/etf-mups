#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "timer.h"

char res_seq[100];
char res_par[100];

#define DIM 2 /* Two-dimensional system */
#define X 0   /* x-coordinate subscript */
#define Y 1   /* y-coordinate subscript */

const double G = 6.673e-11;

typedef double vect_t[DIM]; /* Vector type for position, etc. */

// vect_t forces_reduction[4999][5000];

struct particle_s
{
    double m; /* Mass     */
    vect_t s; /* Position */
    vect_t v; /* Velocity */
};

void Usage(char *prog_name);
void Get_args(int argc, char *argv[], int *n_p, int *n_steps_p,
              double *delta_t_p, int *output_freq_p, char *g_i_p);
void Get_init_cond(struct particle_s curr[], int n);
void Gen_init_cond(struct particle_s curr[], int n);
void Output_state(double time, struct particle_s curr[], int n);
void Compute_force(int part, vect_t forces[], struct particle_s curr[],
                   int n);
void Compute_force_parallel(int part, vect_t forces[], struct particle_s curr[],
                            int n);
void Update_part(int part, vect_t forces[], struct particle_s curr[],
                 int n, double delta_t);
void Compute_energy(struct particle_s curr[], int n, double *kin_en_p,
                    double *pot_en_p);

__global__ void kernel_compute_force_mat(struct particle_s *curr, vect_t *forces, int n)
{
    const unsigned long long int thread_id_i = threadIdx.y + (unsigned long long int)blockIdx.y * blockDim.y;
    const unsigned long long int thread_id_j = threadIdx.x + (unsigned long long int)blockIdx.x * blockDim.x;

    if (thread_id_j < n && thread_id_i < thread_id_j)
    {
        const struct particle_s curr_i = curr[thread_id_i];
        const struct particle_s curr_j = curr[thread_id_j];
        vect_t f_part_k;

        f_part_k[X] = curr_i.s[X] - curr_j.s[X];
        f_part_k[Y] = curr_i.s[Y] - curr_j.s[Y];

        const double len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
        const double len_3 = len * len * len;
        const double mg = -G * curr_i.m * curr_j.m;
        const double fact = mg / len_3;
        f_part_k[X] *= fact;
        f_part_k[Y] *= fact;

        forces[thread_id_j * n + thread_id_i][X] = f_part_k[X];
        forces[thread_id_j * n + thread_id_i][Y] = f_part_k[Y];

        forces[thread_id_i * n + thread_id_j][X] = -f_part_k[X];
        forces[thread_id_i * n + thread_id_j][Y] = -f_part_k[Y];
    }
}

__global__ void kernel_reduce_force_mat(vect_t *forces, int n)
{
    const unsigned long long int thread_id = threadIdx.x + (unsigned long long int)blockIdx.x * blockDim.x;
    if (thread_id < n)
    {
        double sum_x = 0.0;
        double sum_y = 0.0;
        for (unsigned long long int i = thread_id + n; i < n * n; i += n)
        {
            sum_x += forces[i][X];
            sum_y += forces[i][Y];
        }

        forces[thread_id][X] += sum_x;
        forces[thread_id][Y] += sum_y;
    }
}

__global__ void kernel_update_part(struct particle_s *curr, vect_t *forces, int n, double delta_time)
{
    const unsigned long long int thread_id = threadIdx.x + (unsigned long long int)blockIdx.x * blockDim.x;
    if (thread_id < n)
    {
        const double fact = delta_time / curr[thread_id].m;

        curr[thread_id].s[X] += delta_time * curr[thread_id].v[X];
        curr[thread_id].s[Y] += delta_time * curr[thread_id].v[Y];
        curr[thread_id].v[X] += fact * forces[thread_id][X];
        curr[thread_id].v[Y] += fact * forces[thread_id][Y];
    }
}

void sequential_solution(int argc, char *argv[])
{
    int n;                   /* Number of particles        */
    int n_steps;             /* Number of timesteps        */
    int step;                /* Current step               */
    int part;                /* Current particle           */
    int output_freq;         /* Frequency of output        */
    double delta_t;          /* Size of timestep           */
    double t;                /* Current Time               */
    struct particle_s *curr; /* Current state of system    */
    vect_t *forces;          /* Forces on each particle    */
    char g_i;                /*_G_en or _i_nput init conds */
    double kinetic_energy, potential_energy;
    double start, finish; /* For timings                */

    Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
    curr = (struct particle_s *)malloc(n * sizeof(struct particle_s));
    forces = (vect_t *)malloc(n * sizeof(vect_t));
    if (g_i == 'i')
        Get_init_cond(curr, n);
    else
        Gen_init_cond(curr, n);

    GET_TIME(start);
    Compute_energy(curr, n, &kinetic_energy, &potential_energy);
    printf("   PE = %e, KE = %e, Total Energy = %e\n",
           potential_energy, kinetic_energy, kinetic_energy + potential_energy);
    // Output_state(0, curr, n);
    for (step = 1; step <= n_steps; step++)
    {
        t = step * delta_t;
        memset(forces, 0, n * sizeof(vect_t));
        for (part = 0; part < n - 1; part++)
            Compute_force(part, forces, curr, n);
        for (part = 0; part < n; part++)
            Update_part(part, forces, curr, n, delta_t);
        Compute_energy(curr, n, &kinetic_energy, &potential_energy);
    }
    // Output_state(t, curr, n);

    printf("   PE = %e, KE = %e, Total Energy = %e\n",
           potential_energy, kinetic_energy, kinetic_energy + potential_energy);
    sprintf(res_seq, "   PE = %e, KE = %e, Total Energy = %e\n",
            potential_energy, kinetic_energy, kinetic_energy + potential_energy);
    GET_TIME(finish);
    printf("Elapsed time = %e seconds\n", finish - start);

    free(curr);
    free(forces);
} /* sequential_solution */

void parallel_solution(int argc, char *argv[])
{
    int n;                   /* Number of particles        */
    int n_steps;             /* Number of timesteps        */
    int output_freq;         /* Frequency of output        */
    double delta_time;       /* Size of timestep           */
    double t;                /* Current Time               */
    struct particle_s *curr; /* Current state of system    */
    char g_i;                /*_G_en or _i_nput init conds */
    double kinetic_energy, potential_energy;
    double start, finish; /* For timings                */

    Get_args(argc, argv, &n, &n_steps, &delta_time, &output_freq, &g_i);
    curr = (struct particle_s*)malloc(n * sizeof(struct particle_s));

    if (g_i == 'i')
        Get_init_cond(curr, n);
    else
        Gen_init_cond(curr, n);

    GET_TIME(start);
    Compute_energy(curr, n, &kinetic_energy, &potential_energy);
    printf("   PE = %e, KE = %e, Total Energy = %e\n",
           potential_energy, kinetic_energy, kinetic_energy + potential_energy);
    // Output_state(0, curr, n);

    const int block_size = 1024;
    const int grid_size = ceil(((double)n) / block_size);

    struct particle_s *curr_dev;
    vect_t *forces_mat;
    cudaMalloc(&curr_dev, n * sizeof(struct particle_s));
    cudaMalloc(&forces_mat, n * n * sizeof(vect_t));

    cudaMemcpy(curr_dev, curr, n * sizeof(struct particle_s), cudaMemcpyHostToDevice);

    const dim3 block_size_mat(32, 32, 1);
    const size_t grid_cols = (n + block_size_mat.x - 1) / block_size_mat.x;
    const size_t grid_rows = (n + block_size_mat.y - 1) / block_size_mat.y;
    const dim3 grid_size_mat(grid_cols, grid_rows, 1);

    for (int step = 1; step <= n_steps; step++)
    {
        cudaMemset(forces_mat, 0, n * n * sizeof(vect_t));

        kernel_compute_force_mat<<< grid_size_mat, block_size_mat >>>(curr_dev, forces_mat, n);
        kernel_reduce_force_mat<<< grid_size, block_size >>>(forces_mat, n);

        kernel_update_part<<< grid_size, block_size >>>(curr_dev, forces_mat, n, delta_time);
    }

    cudaMemcpy(curr, curr_dev, n * sizeof(struct particle_s), cudaMemcpyDeviceToHost);

    cudaFree(forces_mat);
    cudaFree(curr_dev);

    t = n_steps * delta_time;
    Compute_energy(curr, n, &kinetic_energy, &potential_energy);

    // Output_state(t, curr, n);

    printf("   PE = %e, KE = %e, Total Energy = %e\n",
           potential_energy, kinetic_energy, kinetic_energy + potential_energy);
    sprintf(res_par, "   PE = %e, KE = %e, Total Energy = %e\n",
        potential_energy, kinetic_energy, kinetic_energy + potential_energy);
    GET_TIME(finish);
    printf("Elapsed time = %e seconds\n", finish - start);

    free(curr);
} /* parallel_solution */

int compare_results(void)
{
    return !strcmp(res_seq, res_par);
}

int main(int argc, char *argv[])
{
    float elapsed_time_seq;
    cudaEvent_t start_time_seq, end_time_seq;
    cudaEventCreate(&start_time_seq);
    cudaEventCreate(&end_time_seq);
    float elapsed_time_parallel;
    cudaEvent_t start_time_parallel, end_time_parallel;
    cudaEventCreate(&start_time_parallel);
    cudaEventCreate(&end_time_parallel);

    printf("---------------------Sequential execution---------------------\n");
    cudaEventRecord(start_time_seq, 0);
    sequential_solution(argc, argv);
    cudaEventRecord(end_time_seq, 0);
    cudaEventSynchronize(end_time_seq);
    cudaEventElapsedTime(&elapsed_time_seq, start_time_seq, end_time_seq);

    printf("----------------------Parallel execution----------------------\n");
    cudaEventRecord(start_time_parallel, 0);
    parallel_solution(argc, argv);
    cudaEventRecord(end_time_parallel, 0);
    cudaEventSynchronize(end_time_parallel);
    cudaEventElapsedTime(&elapsed_time_parallel, start_time_parallel, end_time_parallel);

    printf("\nSequential elapsed time: %fs\n", elapsed_time_seq / 1000.0);
    printf("Parallel elapsed time: %fs\n", elapsed_time_parallel / 1000.0);

    if (compare_results())
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    return 0;
} /* main */

void Usage(char *prog_name)
{
    fprintf(stderr, "usage: %s <number of particles> <number of timesteps>\n",
            prog_name);
    fprintf(stderr, "   <size of timestep> <output frequency>\n");
    fprintf(stderr, "   <g|i>\n");
    fprintf(stderr, "   'g': program should generate init conds\n");
    fprintf(stderr, "   'i': program should get init conds from stdin\n");

    exit(0);
} /* Usage */

void Get_args(int argc, char *argv[], int *n_p, int *n_steps_p,
              double *delta_t_p, int *output_freq_p, char *g_i_p)
{
    if (argc != 6)
        Usage(argv[0]);
    *n_p = strtol(argv[1], NULL, 10);
    *n_steps_p = strtol(argv[2], NULL, 10);
    *delta_t_p = strtod(argv[3], NULL);
    *output_freq_p = strtol(argv[4], NULL, 10);
    *g_i_p = argv[5][0];

    if (*n_p <= 0 || *n_steps_p < 0 || *delta_t_p <= 0)
        Usage(argv[0]);
    if (*g_i_p != 'g' && *g_i_p != 'i')
        Usage(argv[0]);

} /* Get_args */

void Get_init_cond(struct particle_s curr[], int n)
{
    int part;

    printf("For each particle, enter (in order):\n");
    printf("   its mass, its x-coord, its y-coord, ");
    printf("its x-velocity, its y-velocity\n");
    for (part = 0; part < n; part++)
    {
        scanf("%lf", &curr[part].m);
        scanf("%lf", &curr[part].s[X]);
        scanf("%lf", &curr[part].s[Y]);
        scanf("%lf", &curr[part].v[X]);
        scanf("%lf", &curr[part].v[Y]);
    }
} /* Get_init_cond */

void Gen_init_cond(struct particle_s curr[], int n)
{
    int part;
    double mass = 5.0e24;
    double gap = 1.0e5;
    double speed = 3.0e4;

    srandom(1);
    for (part = 0; part < n; part++)
    {
        curr[part].m = mass;
        curr[part].s[X] = part * gap;
        curr[part].s[Y] = 0.0;
        curr[part].v[X] = 0.0;
        if (part % 2 == 0)
            curr[part].v[Y] = speed;
        else
            curr[part].v[Y] = -speed;
    }
} /* Gen_init_cond */

void Output_state(double time, struct particle_s curr[], int n)
{
    int part;
    printf("%.2f\n", time);
    for (part = 0; part < n; part++)
    {
        printf("%3d %10.3e ", part, curr[part].s[X]);
        printf("  %10.3e ", curr[part].s[Y]);
        printf("  %10.3e ", curr[part].v[X]);
        printf("  %10.3e\n", curr[part].v[Y]);
    }
    printf("\n");
} /* Output_state */

void Compute_force(int part, vect_t forces[], struct particle_s curr[],
                   int n)
{
    int k;
    double mg;
    vect_t f_part_k;
    double len, len_3, fact;
    for (k = part + 1; k < n; k++)
    {
        f_part_k[X] = curr[part].s[X] - curr[k].s[X];
        f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
        len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
        len_3 = len * len * len;
        mg = -G * curr[part].m * curr[k].m;
        fact = mg / len_3;
        f_part_k[X] *= fact;
        f_part_k[Y] *= fact;

        forces[part][X] += f_part_k[X];
        forces[part][Y] += f_part_k[Y];
        forces[k][X] -= f_part_k[X];
        forces[k][Y] -= f_part_k[Y];
    }
} /* Compute_force */

void Update_part(int part, vect_t forces[], struct particle_s curr[],
                 int n, double delta_t)
{
    double fact = delta_t / curr[part].m;

    curr[part].s[X] += delta_t * curr[part].v[X];
    curr[part].s[Y] += delta_t * curr[part].v[Y];
    curr[part].v[X] += fact * forces[part][X];
    curr[part].v[Y] += fact * forces[part][Y];
} /* Update_part */

void Compute_energy(struct particle_s curr[], int n, double *kin_en_p,
                    double *pot_en_p)
{
    int i, j;
    vect_t diff;
    double pe = 0.0, ke = 0.0;
    double dist, speed_sqr;

    for (i = 0; i < n; i++)
    {
        speed_sqr = curr[i].v[X] * curr[i].v[X] + curr[i].v[Y] * curr[i].v[Y];
        ke += curr[i].m * speed_sqr;
    }
    ke *= 0.5;

    for (i = 0; i < n - 1; i++)
    {
        for (j = i + 1; j < n; j++)
        {
            diff[X] = curr[i].s[X] - curr[j].s[X];
            diff[Y] = curr[i].s[Y] - curr[j].s[Y];
            dist = sqrt(diff[X] * diff[X] + diff[Y] * diff[Y]);
            pe += -G * curr[i].m * curr[j].m / dist;
        }
    }

    *kin_en_p = ke;
    *pot_en_p = pe;
} /* Compute_energy */
