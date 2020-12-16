#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "timer.h"

#define DIM 2 /* Two-dimensional system */
#define X 0   /* x-coordinate subscript */
#define Y 1   /* y-coordinate subscript */

const double G = 6.673e-11;

typedef double vect_t[DIM]; /* Vector type for position, etc. */

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
void Update_part(int part, vect_t forces[], struct particle_s curr[],
                 int n, double delta_t);
void Compute_energy(struct particle_s curr[], int n, double *kin_en_p,
                    double *pot_en_p);

int rank, size;
MPI_Datatype particle_s_type;
MPI_Datatype vect_t_type;

enum Tags
{
    TAG_CURR = 1000,
    TAG_FORCES_SEND,
    TAG_FORCES,
};

int main(int argc, char *argv[])
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

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    curr = malloc(n * sizeof(struct particle_s));
    forces = malloc(n * sizeof(vect_t));

    // define types for communication
    MPI_Type_contiguous(sizeof(vect_t) / sizeof(double),
                        MPI_DOUBLE, &vect_t_type);
    MPI_Type_commit(&vect_t_type);
    MPI_Type_contiguous(sizeof(struct particle_s) / sizeof(double),
                        MPI_DOUBLE, &particle_s_type);
    MPI_Type_commit(&particle_s_type);

    if (rank == 0)
    {
        Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
        if (g_i == 'i')
            Get_init_cond(curr, n);
        else
            Gen_init_cond(curr, n);

        GET_TIME(start);
        Compute_energy(curr, n, &kinetic_energy, &potential_energy);
        printf("   PE = %e, KE = %e, Total Energy = %e\n",
               potential_energy, kinetic_energy, kinetic_energy + potential_energy);
        Output_state(0, curr, n);
    }

    MPI_Bcast(&n_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&delta_t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (step = 1; step <= n_steps; step++)
    {
        memset(forces, 0, n * sizeof(vect_t));
        for (part = 0; part < n - 1; part++)
            Compute_force(part, forces, curr, n);
        if (rank == 0)
            for (part = 0; part < n; part++)
                Update_part(part, forces, curr, n, delta_t);
    }

    if (rank == 0)
    {
        t = step * delta_t;
        Compute_energy(curr, n, &kinetic_energy, &potential_energy);
        Output_state(t, curr, n);

        printf("   PE = %e, KE = %e, Total Energy = %e\n",
               potential_energy, kinetic_energy, kinetic_energy + potential_energy);

        GET_TIME(finish);
        printf("Elapsed time = %e seconds\n", finish - start);
    }

    free(curr);
    free(forces);

    MPI_Type_free(&vect_t_type);
    MPI_Type_free(&particle_s_type);
    MPI_Finalize();

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

    double forces_part_x = 0.0, forces_part_y = 0.0;
    double forces_part_x_reduced = 0.0, forces_part_y_reduced = 0.0;

    int k_start, k_end;
    int chunk = (n - (part + 1)) / size;

    k_start = part + 1 + rank * chunk;
    k_end = k_start + chunk;
    if (size > 1 && rank == size - 1)
    {
        k_end = n;
        chunk = k_end - k_start;
    }

    if (rank == 0)
    {
        for (int i = 1; i < size - 1; i++)
        {
            MPI_Send(curr + part + 1 + i * chunk, chunk, particle_s_type, i, TAG_CURR, MPI_COMM_WORLD);
            MPI_Send(forces + part + 1 + i * chunk, chunk, vect_t_type, i, TAG_FORCES_SEND, MPI_COMM_WORLD);
        }
        if (size > 1)
        {
            int buff_size = n - (part + 1 + (size - 1) * chunk);
            MPI_Send(curr + part + 1 + (size - 1) * chunk, buff_size, particle_s_type, (size - 1), TAG_CURR, MPI_COMM_WORLD);
            MPI_Send(forces + part + 1 + (size - 1) * chunk, buff_size, vect_t_type, (size - 1), TAG_FORCES_SEND, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(curr + k_start, chunk, particle_s_type, 0, TAG_CURR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(forces + k_start, chunk, vect_t_type, 0, TAG_FORCES_SEND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (k = k_start; k < k_end; k++)
    {
        f_part_k[X] = curr[part].s[X] - curr[k].s[X];
        f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
        len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
        len_3 = len * len * len;
        mg = -G * curr[part].m * curr[k].m;
        fact = mg / len_3;
        f_part_k[X] *= fact;
        f_part_k[Y] *= fact;

        // forces[part][X] += f_part_k[X];
        // forces[part][Y] += f_part_k[Y];
        forces_part_x += f_part_k[X];
        forces_part_y += f_part_k[Y];
        forces[k][X] -= f_part_k[X];
        forces[k][Y] -= f_part_k[Y];
    }

    MPI_Reduce(&forces_part_x, &forces_part_x_reduced, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&forces_part_y, &forces_part_y_reduced, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 1; i < size - 1; i++)
        {
            MPI_Recv(forces + part + 1 + i * chunk, chunk, vect_t_type, i, TAG_FORCES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (size > 1)
        {
            int buff_size = n - (part + 1 + (size - 1) * chunk);
            MPI_Recv(forces + part + 1 + (size - 1) * chunk, buff_size, vect_t_type, (size - 1), TAG_FORCES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        forces[part][X] += forces_part_x_reduced;
        forces[part][Y] += forces_part_y_reduced;
    }
    else
    {
        MPI_Send(forces + k_start, chunk, vect_t_type, 0, TAG_FORCES, MPI_COMM_WORLD);
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
