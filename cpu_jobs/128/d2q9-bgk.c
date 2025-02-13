/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <sys/resource.h>
#include <string.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
const int TEST = 1;
const int ASYNC_HALOS = 0;
const int SPREAD_COLS_EVENLY = 1;
const int MERGE_TIMESTEP = 1;
const int REDUCE_HALO_SPEED_ECHANGE = 1;

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

typedef struct
{
  float* restrict speeds[NSPEEDS];
} t_speed_arrays;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_arrays** cells_ptr, t_speed_arrays** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_speed_arrays** cells, t_speed_arrays** tmp_cells, int* obstacles, int flag);
float timestep_async(const t_param params, t_speed_arrays** cells, t_speed_arrays** tmp_cells, int* obstacles, int flag,
                                           t_speed_arrays *tmp_cells2, int total_requests, MPI_Request ** requests);
int accelerate_flow(const t_param params, t_speed_arrays* cells, int* obstacles, int flag);
int propagate(const t_param params, t_speed_arrays* cells, t_speed_arrays* tmp_cells, int flag);
int rebound(const t_param params, t_speed_arrays* cells, t_speed_arrays* tmp_cells, int* obstacles, int flag);
int collision(const t_param params, t_speed_arrays* cells, t_speed_arrays* tmp_cells, int* obstacles, int flag);
float merged_timestep_ops(const t_param params, t_speed_arrays* cells, t_speed_arrays* tmp_cells, int* obstacles, int flag);

int write_values(const t_param params, t_speed_arrays* cells, int* obstacles, float* av_vels);
void initialise_params_from_file(const char* paramfile, t_param* params);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed_arrays** cells_ptr, t_speed_arrays** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed_arrays* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed_arrays* cells, int* obstacles, int flag);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed_arrays* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);
int calc_ncols_from_rank(int rank, int size, int nx);
void output_state(const char* output_file, int step, t_speed_arrays *cells, int *obstacles, int nx, int ny);
void test_vels(const char* output_file, float *vels, int steps);
void exchange_obstacles(int rank, int size, t_param child_params, int *child_obstacles,
                      int* sbuffer_obstacles, int* rbuffer_obstacles);
void exchange_halos(int rank, int size, t_param child_params, t_speed_arrays *child_cells,
                      float* sbuffer_cells, float* rbuffer_cells);
void exchange_halos_async(MPI_Request** requests, int rank, int size, t_param child_params, t_speed_arrays *child_cells,
                      float* sbuffer_cells1, float* rbuffer_cells1,
                      float* sbuffer_cells2, float* rbuffer_cells2);
void swap_floats(float *var1, float *var2);
void swap_cells(t_speed *var1, t_speed *var2);
void swap_cells_arrays(t_speed_arrays *var1, t_speed_arrays *var2, int coord1, int coord2);
int start_process_grid_from(int size, int rank, int n);
int min(int a, int b);
t_speed_arrays* create_t_speed_arrays(t_param params);
void free_t_speed_arrays(t_speed_arrays* obj);


/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed_arrays* cells     = NULL;    /* grid containing fluid densities */
  t_speed_arrays* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  //MPI related
  int rank;               /* 'rank' of process among it's cohort */
  int size;               /* size of cohort, i.e. num processes started */
  int flag;               /* for checking whether MPI_Init() has been called */
  int strlen;             /* length of a character array */
  char hostname[MPI_MAX_PROCESSOR_NAME];  /* character array to hold hostname running process */
  t_speed_arrays *child_cells;
  t_speed_arrays *child_tmp_cells;
  int *child_obstacles;
  float *child_vels;
  float *rbuffer_vels;
  float *sbuffer_cells1;
  float *rbuffer_cells1;
  int *sbuffer_obstacles1;
  int *rbuffer_obstacles1;
  float *sbuffer_cells2;
  float *rbuffer_cells2;
  t_speed_arrays *old_cell_vals;
  MPI_Request** requests;

  /* initialise our MPI environment */
  MPI_Init( &argc, &argv );

  /* check whether the initialisation was successful */
  MPI_Initialized(&flag);
  if ( flag != 1 ) {
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  /* determine the hostname */
  MPI_Get_processor_name(hostname,&strlen);

  /*
  ** determine the SIZE of the group of processes associated with
  ** the 'communicator'.  MPI_COMM_WORLD is the default communicator
  ** consisting of all the processes in the launched MPI 'job'
  */
  MPI_Comm_size( MPI_COMM_WORLD, &size );

  /* determine the RANK of the current process [0:SIZE-1] */
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  initialise_params_from_file(paramfile, &params);
  t_param child_params;
  child_params = params;

  //Work out child params
  int child_cols = calc_ncols_from_rank(rank, size, params.nx);
  child_params.nx = child_cols + 2; // add 2 halo cols
  //Initialise child memory
  rbuffer_vels = (float*) calloc(params.maxIters, sizeof(float));
  child_cells = create_t_speed_arrays(child_params);
  child_tmp_cells = create_t_speed_arrays(child_params);
  child_obstacles = (int*) calloc((child_params.ny * child_params.nx), sizeof(int));
  child_vels = (float*) calloc(params.maxIters, sizeof(float));
  sbuffer_cells1 = (float*) calloc(params.ny * NSPEEDS, sizeof(float));
  rbuffer_cells1 = (float*) calloc(params.ny * NSPEEDS, sizeof(float));
  sbuffer_obstacles1 = (int *) calloc(params.ny, sizeof(int));
  rbuffer_obstacles1 = (int *) calloc(params.ny, sizeof(int));
  sbuffer_cells2 = (float*) calloc(params.ny * NSPEEDS, sizeof(float));
  rbuffer_cells2 = (float*) calloc(params.ny * NSPEEDS, sizeof(float));
  old_cell_vals = create_t_speed_arrays(child_params);
  requests = (MPI_Request **) malloc(4*sizeof(MPI_Request*));  // for async halo exchange

  if(rank == 0) {
    printf("Number of processes: %d\n", size);
    if(ASYNC_HALOS) printf("Asynchronous halo exchange.\n");
    if(SPREAD_COLS_EVENLY) printf("Spreading remainder cols evenly.\n");
    if(MERGE_TIMESTEP) printf("Merging propagate, rebound, collision and av_velocity.\n");
    if(REDUCE_HALO_SPEED_ECHANGE) printf("Using reduced halo exchange.\n");
    /* initialise our data structures and load values from file */
    initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

    //allocate memory for async send buffs
    float** send_buffer_cells = (float**) malloc(size * sizeof(float*));
    int** send_buffer_obstacles = (int**) malloc(size * sizeof(int*));
    for(int process = 0; process < size; ++process) {
      int process_cols = calc_ncols_from_rank(process, size, params.nx);
      send_buffer_cells[process] = (float*) malloc(params.ny * process_cols * NSPEEDS * sizeof(float));
      send_buffer_obstacles[process] = (int*) malloc(params.ny * process_cols * sizeof(int));
    }

    //Send data to children and itself
    for(int process = 0; process < size; ++process) {
      int current_child_cols = calc_ncols_from_rank(process, size, params.nx);
      int start_from = start_process_grid_from(size, process, params.nx);
      //printf("rank: %d, start from: %d, cols: %d\n", process, start_from, current_child_cols);
      for(int col = start_from, child_col = 0; col < start_from + current_child_cols; ++col, ++child_col) {
        //Fill send buffers
        for(int row = 0; row < params.ny; ++row) {
          send_buffer_obstacles[process][child_col*params.ny + row] = obstacles[row*params.nx + col];
          for(int speed = 0; speed < NSPEEDS; ++speed) {
            send_buffer_cells[process][child_col*params.ny*NSPEEDS + row*NSPEEDS + speed] = cells->speeds[speed][row*params.nx + col];
          }
        }
        //Send data
        MPI_Request send_request;
        MPI_Isend(&send_buffer_cells[process][child_col*params.ny*NSPEEDS], params.ny*NSPEEDS, MPI_FLOAT, process, 0, MPI_COMM_WORLD, &send_request);
        MPI_Isend(&send_buffer_obstacles[process][child_col*params.ny], params.ny, MPI_INT, process, 1, MPI_COMM_WORLD, &send_request);
      }
    }
    // Done sending stuff
  }
  //Receive data from master
  for(int col = 1; col < child_params.nx-1; ++col) {
    MPI_Recv(rbuffer_cells1, child_params.ny*NSPEEDS, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(rbuffer_obstacles1, child_params.ny, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for(int row = 0; row < child_params.ny; ++row) {
      child_obstacles[row*child_params.nx + col] = rbuffer_obstacles1[row];
      for(int speed = 0; speed < NSPEEDS; ++speed) {
        child_cells->speeds[speed][row*child_params.nx + col] = rbuffer_cells1[row*NSPEEDS + speed];
      }
    }
  }
  //obstacles don't ever change values, so send here for halos once
  exchange_obstacles(rank, size, child_params, child_obstacles, sbuffer_obstacles1, rbuffer_obstacles1);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  char output_file[1024];
  sprintf(output_file, "final_state_size_%d.txt", size);
  fclose(fopen(output_file, "w"));

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    //output_state(file_name, tt, process_cells, process_obstacles, process_params.nx, process_params.ny);
    if(rank == 0 && tt % 500 == 0) printf("iteration: %d\n", tt);

    if(!ASYNC_HALOS) {
      if(rank == 0 && tt == 0) printf("Flag: 2\n");
      //Exchange halos
      if(size > 1) {
        exchange_halos(rank, size, child_params, child_cells, sbuffer_cells1, rbuffer_cells1);
      }

      //now do computations
      //timestep(child_params, &child_cells, &child_tmp_cells, child_obstacles, 2);
      timestep_async(child_params, &child_cells, &child_tmp_cells, child_obstacles, 2, old_cell_vals, 0, 0);
      child_vels[tt] = av_velocity(child_params, child_cells, child_obstacles, 2);
    } else {
      int total_requests = 4;  // total async halo exchange requests
      for(int i = 0; i < total_requests; ++i) {
        requests[i] = (MPI_Request*) malloc(sizeof(MPI_Request));
      }
      exchange_halos_async(requests, rank, size, child_params, child_cells,
                                                      sbuffer_cells1, rbuffer_cells1,
                                                      sbuffer_cells2, rbuffer_cells2);

      //now do computations
      //timestep(child_params, &child_cells, &child_tmp_cells, child_obstacles, 0);
      if(MERGE_TIMESTEP) {
        child_vels[tt] = timestep_async(child_params, &child_cells, &child_tmp_cells, child_obstacles, 0, old_cell_vals, total_requests, requests);
      } else {
        timestep_async(child_params, &child_cells, &child_tmp_cells, child_obstacles, 0, old_cell_vals, total_requests, requests);
        child_vels[tt] = av_velocity(child_params, child_cells, child_obstacles, 0);
      }

      //synchronise
      for(int i = 0; i < total_requests; ++i) {
        MPI_Request* current_request = requests[i];
        MPI_Wait(current_request, MPI_STATUS_IGNORE);
      }
      int speeds_to_recv = REDUCE_HALO_SPEED_ECHANGE ? 3 : NSPEEDS;
      //populate left col
      for(int row = 0; row < child_params.ny; ++row) {
        //t_speed speeds;
        if(REDUCE_HALO_SPEED_ECHANGE) {
          if(MERGE_TIMESTEP) {
            child_tmp_cells->speeds[1][row*child_params.nx] = rbuffer_cells2[row*speeds_to_recv + 0];
            child_tmp_cells->speeds[5][row*child_params.nx] = rbuffer_cells2[row*speeds_to_recv + 1];
            child_tmp_cells->speeds[8][row*child_params.nx] = rbuffer_cells2[row*speeds_to_recv + 2];
          } else {
            child_cells->speeds[1][row*child_params.nx] = rbuffer_cells2[row*speeds_to_recv + 0];
            child_cells->speeds[5][row*child_params.nx] = rbuffer_cells2[row*speeds_to_recv + 1];
            child_cells->speeds[8][row*child_params.nx] = rbuffer_cells2[row*speeds_to_recv + 2];
          }
        } else {
            for(int speed = 0; speed < NSPEEDS; ++speed) {
              //speeds.speeds[speed] = rbuffer_cells2[row*NSPEEDS + speed];
              if(MERGE_TIMESTEP) {
                  child_tmp_cells->speeds[speed][row*child_params.nx] = rbuffer_cells2[row*speeds_to_recv + speed];
              } else {
                  child_cells->speeds[speed][row*child_params.nx] = rbuffer_cells2[row*speeds_to_recv + speed];
              }
            }
        }
      }
      //populate right col
      for(int row = 0; row < child_params.ny; ++row) {
        //t_speed speeds;
        if(REDUCE_HALO_SPEED_ECHANGE) {
          if(MERGE_TIMESTEP) {
            child_tmp_cells->speeds[3][row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells1[row*speeds_to_recv + 0];
            child_tmp_cells->speeds[6][row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells1[row*speeds_to_recv + 1];
            child_tmp_cells->speeds[7][row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells1[row*speeds_to_recv + 2];
          } else {
            child_cells->speeds[3][row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells1[row*speeds_to_recv + 0];
            child_cells->speeds[6][row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells1[row*speeds_to_recv + 1];
            child_cells->speeds[7][row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells1[row*speeds_to_recv + 2];
          }
        } else {
          for(int speed = 0; speed < NSPEEDS; ++speed) {
            //speeds.speeds[speed] = rbuffer_cells1[row*NSPEEDS + speed];
            if(MERGE_TIMESTEP) {
                child_tmp_cells->speeds[speed][row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells1[row*speeds_to_recv + speed];
            } else {
                child_cells->speeds[speed][row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells1[row*speeds_to_recv + speed];
            }
          }
        }
      }

      //now do computations
      //timestep(child_params, &child_cells, &child_tmp_cells, child_obstacles, 1);
      if(MERGE_TIMESTEP) {
        child_vels[tt] += timestep_async(child_params, &child_cells, &child_tmp_cells, child_obstacles, 1, old_cell_vals, total_requests, requests);
      } else {
        timestep_async(child_params, &child_cells, &child_tmp_cells, child_obstacles, 1, old_cell_vals, total_requests, requests);
        child_vels[tt] += av_velocity(child_params, child_cells, child_obstacles, 1);
      }
    }

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
    if(TEST && rank == 0 && (tt < 20 || tt % 500 == 0)) {
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", child_vels[tt]);
    }
  }

  //Handle average velocity computations
  if(rank == 0) {
    for(int process = 1; process < size; ++process) {
      MPI_Recv(rbuffer_vels, child_params.maxIters, MPI_FLOAT, process, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for(int tt = 0; tt < child_params.maxIters; ++tt) {
        child_vels[tt] += rbuffer_vels[tt];
      }
    }
    //compute average velocity
    int tot_u = 0;
    for(int row = 0; row < params.ny; ++row) {
      for(int col = 0; col < params.nx; ++col) {
        if(!obstacles[row*params.nx + col]) {
          ++tot_u;
        }
      }
    }
    for(int tt = 0; tt < child_params.maxIters; ++tt) {
      av_vels[tt] = child_vels[tt] / tot_u;
    }
  } else {
    MPI_Send(child_vels, child_params.maxIters, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
  }

  if(rank == 0) {
    //char output_file[1024];
    //sprintf(output_file, "velocities_tot_u_size_%d.txt", size);
    //test_vels(output_file, av_vels, child_params.maxIters);
    gettimeofday(&timstr, NULL);
    toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr = ru.ru_utime;
    usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    timstr = ru.ru_stime;
    systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    printf("==done==\n");
    //printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  }

//DONT TIME THIS!!!! {{{
  //Send data from child process to master
  float *send_child_buffer_cells = (float*) malloc(child_params.nx * child_params.ny * NSPEEDS * sizeof(float));
  int *send_child_buffer_obstacles = (int*) malloc(child_params.nx * child_params.ny * sizeof(int));
  for(int col = 1; col < child_params.nx-1; ++col) {
    for(int row = 0; row < child_params.ny; ++row) {
      send_child_buffer_obstacles[col*child_params.ny + row] = child_obstacles[row*child_params.nx + col];
      for(int speed = 0; speed < NSPEEDS; ++speed) {
        send_child_buffer_cells[col*child_params.ny*NSPEEDS + row*NSPEEDS + speed] = child_cells->speeds[speed][row*child_params.nx + col];
      }
    }
    MPI_Request send_request;
    MPI_Isend(&send_child_buffer_cells[col*child_params.ny*NSPEEDS], child_params.ny*NSPEEDS, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &send_request);
    MPI_Isend(&send_child_buffer_obstacles[col*child_params.ny], child_params.ny, MPI_INT, 0, 1, MPI_COMM_WORLD, &send_request);
  }

  if(rank == 0) {
    //Receive data from children
    for(int process = 0; process < size; ++process) {
      int current_child_cols = calc_ncols_from_rank(process, size, params.nx);
      int start_from = start_process_grid_from(size, process, params.nx);
      for(int col = start_from, child_col = 0; col < start_from + current_child_cols; ++col, ++child_col) {
        MPI_Recv(rbuffer_cells1, child_params.ny*NSPEEDS, MPI_FLOAT, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(rbuffer_obstacles1, child_params.ny, MPI_INT, process, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //Fill send buffers
        for(int row = 0; row < params.ny; ++row) {
          obstacles[row*params.nx + col] = rbuffer_obstacles1[row];
          //t_speed speeds;
          for(int speed = 0; speed < NSPEEDS; ++speed) {
            //speeds.speeds[speed] = rbuffer_cells1[row*NSPEEDS + speed];
            cells->speeds[speed][row*params.nx + col] = rbuffer_cells1[row*NSPEEDS + speed];
          }
        }
      }
    }
  }

  //}}}

  if(rank == 0) {
    write_values(params, cells, obstacles, av_vels);
    finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  }

  /* finialise the MPI enviroment */
  MPI_Finalize();
  free(child_cells);
  free(child_tmp_cells);
  free(child_obstacles);
  free(sbuffer_cells1);
  free(rbuffer_cells1);
  free(sbuffer_obstacles1);
  free(rbuffer_obstacles1);
  free(sbuffer_cells2);
  free(rbuffer_cells2);

  return EXIT_SUCCESS;
}

void swap_floats(float *var1, float *var2) {
  float temp = *var1;
  *var1 = *var2;
  *var2 = temp;
}

void swap_cells(t_speed *var1, t_speed *var2) {
  t_speed temp = *var1;
  *var1 = *var2;
  *var2 = temp;
}

void swap_cells_arrays(t_speed_arrays *var1, t_speed_arrays *var2, int coord1, int coord2) {
  for(int kk = 0; kk < NSPEEDS; ++kk) {
    swap_floats(&var1->speeds[kk][coord1], &var2->speeds[kk][coord2]);
  }
}

int start_process_grid_from(int size, int rank, int n) {
  int start_from = -1;
  int per_process = n / size;
  if(SPREAD_COLS_EVENLY) {
    int remainder_cols = n % size;
    int spreaded_remainder_before_process = min(rank, remainder_cols);
    start_from = per_process*rank + spreaded_remainder_before_process;
  } else {
    start_from = per_process*rank;
  }

  return start_from;
}

int min(int a, int b) {
  return (a < b) ? a : b;
}

void exchange_obstacles(int rank, int size, t_param child_params, int *child_obstacles,
                      int* sbuffer_obstacles, int* rbuffer_obstacles) {
  int left = (rank == 0) ? (rank + size - 1) : (rank - 1); // left is bottom, right is top equiv
  int right = (rank + 1) % size;
  //send to the left, receive from right
  //fill with left col
  for(int row = 0; row < child_params.ny; ++row) {
    sbuffer_obstacles[row] = child_obstacles[row*child_params.nx + 1];
  }
  MPI_Sendrecv(sbuffer_obstacles, child_params.ny, MPI_INT, left, 1, rbuffer_obstacles,
              child_params.ny, MPI_INT, right, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //populate right col
  for(int row = 0; row < child_params.ny; ++row) {
    child_obstacles[row*child_params.nx + (child_params.nx - 1)] = rbuffer_obstacles[row];
  }
  //send to right, receive from left
  //fill with right col
  for(int row = 0; row < child_params.ny; ++row) {
    sbuffer_obstacles[row] = child_obstacles[row*child_params.nx + (child_params.nx - 2)];
  }
  MPI_Sendrecv(sbuffer_obstacles, child_params.ny, MPI_INT, right, 1, rbuffer_obstacles,
              child_params.ny, MPI_INT, left, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //populate left col
  for(int row = 0; row < child_params.ny; ++row) {
    child_obstacles[row*child_params.nx] = rbuffer_obstacles[row];
  }
}


void exchange_halos_async(MPI_Request** requests, int rank, int size, t_param child_params, t_speed_arrays *child_cells,
                      float* sbuffer_cells1, float* rbuffer_cells1,
                      float* sbuffer_cells2, float* rbuffer_cells2) {
  int left = (rank == 0) ? (rank + size - 1) : (rank - 1); // left is bottom, right is top equiv
  int right = (rank + 1) % size;
  int speeds_to_send = REDUCE_HALO_SPEED_ECHANGE ? 3 : NSPEEDS;
  MPI_Irecv(rbuffer_cells1, child_params.ny*speeds_to_send, MPI_FLOAT, right, 0, MPI_COMM_WORLD, requests[1]);
  MPI_Irecv(rbuffer_cells2, child_params.ny*speeds_to_send, MPI_FLOAT, left, 0, MPI_COMM_WORLD, requests[3]);
  //send to the left, receive from right
  //fill with left col
  for(int row = 0; row < child_params.ny; ++row) {
    if(REDUCE_HALO_SPEED_ECHANGE) {
        sbuffer_cells1[row*speeds_to_send + 0] = child_cells->speeds[3][row*child_params.nx + 1];
        sbuffer_cells1[row*speeds_to_send + 1] = child_cells->speeds[6][row*child_params.nx + 1];
        sbuffer_cells1[row*speeds_to_send + 2] = child_cells->speeds[7][row*child_params.nx + 1];
    } else {
      for(int speed = 0; speed < NSPEEDS; ++speed) {
        sbuffer_cells1[row*speeds_to_send + speed] = child_cells->speeds[speed][row*child_params.nx + 1];
      }
    }

  }
  MPI_Isend(sbuffer_cells1, child_params.ny*speeds_to_send, MPI_FLOAT, left, 0, MPI_COMM_WORLD, requests[0]);

  //send to right, receive from left
  //fill with right col
  for(int row = 0; row < child_params.ny; ++row) {
    if(REDUCE_HALO_SPEED_ECHANGE) {
      sbuffer_cells2[row*speeds_to_send + 0] = child_cells->speeds[1][row*child_params.nx + (child_params.nx - 2)];
      sbuffer_cells2[row*speeds_to_send + 1] = child_cells->speeds[5][row*child_params.nx + (child_params.nx - 2)];
      sbuffer_cells2[row*speeds_to_send + 2] = child_cells->speeds[8][row*child_params.nx + (child_params.nx - 2)];
    } else {
      for(int speed = 0; speed < NSPEEDS; ++speed) {
        sbuffer_cells2[row*speeds_to_send + speed] = child_cells->speeds[speed][row*child_params.nx + (child_params.nx - 2)];
      }
    }
  }
  MPI_Isend(sbuffer_cells2, child_params.ny*speeds_to_send, MPI_FLOAT, right, 0, MPI_COMM_WORLD, requests[2]);

}

t_speed_arrays* create_t_speed_arrays(t_param params) {
  t_speed_arrays* object_ptr = (t_speed_arrays*) calloc(1, sizeof(t_speed_arrays));
  for(int kk = 0; kk < NSPEEDS; ++kk) {
    object_ptr->speeds[kk] = (float*) calloc(params.nx*params.ny, sizeof(float));
  }
  return object_ptr;
}

void free_t_speed_arrays(t_speed_arrays* obj) {
  for(int kk = 0; kk < NSPEEDS; ++kk) {
    free(obj->speeds[kk]);
  }
  free(obj);
}

void exchange_halos(int rank, int size, t_param child_params, t_speed_arrays *child_cells,
                      float* sbuffer_cells, float* rbuffer_cells) {
  int left = (rank == 0) ? (rank + size - 1) : (rank - 1); // left is bottom, right is top equiv
  int right = (rank + 1) % size;
  //send to the left, receive from right
  //fill with left col
  for(int row = 0; row < child_params.ny; ++row) {
    for(int speed = 0; speed < NSPEEDS; ++speed) {
      sbuffer_cells[row*NSPEEDS + speed] = child_cells->speeds[speed][row*child_params.nx + 1];
    }
  }

  MPI_Sendrecv(sbuffer_cells, child_params.ny*NSPEEDS, MPI_FLOAT, left, 0, rbuffer_cells,
              child_params.ny*NSPEEDS, MPI_FLOAT, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //populate right col
  for(int row = 0; row < child_params.ny; ++row) {
    //t_speed speeds;
    for(int speed = 0; speed < NSPEEDS; ++speed) {
      child_cells->speeds[speed][row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells[row*NSPEEDS + speed];
      //speeds.speeds[speed] = rbuffer_cells[row*NSPEEDS + speed];
    }

  }
  //send to right, receive from left
  //fill with right col
  for(int row = 0; row < child_params.ny; ++row) {
    for(int speed = 0; speed < NSPEEDS; ++speed) {
      sbuffer_cells[row*NSPEEDS + speed] = child_cells->speeds[speed][row*child_params.nx + (child_params.nx - 2)];
    }
  }
  MPI_Sendrecv(sbuffer_cells, child_params.ny*NSPEEDS, MPI_FLOAT, right, 0, rbuffer_cells,
              child_params.ny*NSPEEDS, MPI_FLOAT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //populate left col
  for(int row = 0; row < child_params.ny; ++row) {
    //t_speed speeds;
    for(int speed = 0; speed < NSPEEDS; ++speed) {
      //speeds.speeds[speed] = rbuffer_cells[row*NSPEEDS + speed];
      child_cells->speeds[speed][row*child_params.nx] = rbuffer_cells[row*NSPEEDS + speed];
    }

  }
}

void output_state(const char* output_file, int step, t_speed_arrays *cells, int *obstacles, int nx, int ny) {
  FILE* fp = fopen(output_file, "a");
  if (fp == NULL)
  {
    printf("could not open input parameter file: %s", output_file);
    return;
  }
  fprintf(fp, "Step %d:\n", step);
  for(int i = 0; i < ny; ++i) {
    for(int j = 0; j < nx; ++j) {
      for(int z = 0; z < 9; ++z) {
        fprintf(fp, "%f ", cells->speeds[z][i*nx + j]);
      }
      fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
  }
  for(int i = 0; i < ny; ++i) {
    for(int j = 0; j < nx; ++j) {
      fprintf(fp, "%d ", obstacles[i*nx + j]);
    }
    fprintf(fp, "\n");
  }
  fprintf(fp, "\n\n");
  fclose(fp);
}

int calc_ncols_from_rank(int rank, int size, int nx)
{
  int ncols;

  if(!SPREAD_COLS_EVENLY) {
    ncols = nx / size;       /* integer division */
    if ((nx % size) != 0) {  /* if there is a remainder */
      if (rank == size - 1)
        ncols += nx % size;  /* add remainder to last rank */
    }
  } else {
    int per_process = nx / size;
    int remainder_cols = nx % size;
    ncols = per_process;
    if(rank < remainder_cols) {
      ++ncols;
    }
  }

  return ncols;
}

void initialise_params_from_file(const char* paramfile, t_param* params) {
  char   message[1024];  /* message buffer */
  int    retval;         /* to hold return value for checking */
  FILE* fp;

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);
}

int timestep(const t_param params, t_speed_arrays** cells, t_speed_arrays** tmp_cells, int* obstacles, int flag)
{
  accelerate_flow(params, *cells, obstacles, flag);

  propagate(params, *cells, *tmp_cells, flag);
  rebound(params, *cells, *tmp_cells, obstacles, flag);
  collision(params, *cells, *tmp_cells, obstacles, flag);

  //printf("ALBADASKFEWF\n");
  //printf("flag: %d\n", flag);

  //merged_timestep_ops(params, *cells, *tmp_cells, obstacles, flag);
  //t_speed *cells_ptr = *cells;
  //*cells = *tmp_cells;
  //*tmp_cells = cells_ptr;

  return EXIT_SUCCESS;
}

float timestep_async(const t_param params, t_speed_arrays** cells, t_speed_arrays** tmp_cells, int* obstacles, int flag, t_speed_arrays *tmp_cells2, int total_requests, MPI_Request **requests)
{
  float res = -1;
  if(flag == 0) {
    accelerate_flow(params, *cells, obstacles, 0);

    if(MERGE_TIMESTEP) {
      res = merged_timestep_ops(params, *cells, *tmp_cells, obstacles, 0);
      t_speed_arrays *cells_ptr = *cells;
      *cells = *tmp_cells;
      *tmp_cells = cells_ptr;
    } else {

      //retain cols 2 and params.nx-3
      for(int i = 0; i < params.ny; ++i) {
        for(int kk = 0; kk < NSPEEDS; ++kk) {
          tmp_cells2->speeds[kk][i] = (*cells)->speeds[kk][i*params.nx + 2];
          tmp_cells2->speeds[kk][params.ny + i] = (*cells)->speeds[kk][i*params.nx + (params.nx - 3)];
        }
      }

      propagate(params, *cells, *tmp_cells, 0);
      rebound(params, *cells, *tmp_cells, obstacles, 0);
      collision(params, *cells, *tmp_cells, obstacles, 0);
    }
  } else if(flag == 1) {

    if(MERGE_TIMESTEP) {
      accelerate_flow(params, *tmp_cells, obstacles, 1);
      res = merged_timestep_ops(params, *tmp_cells, *cells, obstacles, 1);

    } else {
      //swap vals
      for(int i = 0; i < params.ny; ++i) {
        swap_cells_arrays(tmp_cells2, *cells, i, i*params.nx + 2);
        swap_cells_arrays(tmp_cells2, *cells, params.ny + i, i*params.nx + (params.nx - 3));
      }

      accelerate_flow(params, *cells, obstacles, 1);
      propagate(params, *cells, *tmp_cells, 1);
      //MPI implementations are lazy, so check for status to encourage exchange
      for(int i = 0; i < total_requests; ++i) {
          int res = 0;
          MPI_Test(requests[i], &res, MPI_STATUS_IGNORE);
      }
      rebound(params, *cells, *tmp_cells, obstacles, 1);
      collision(params, *cells, *tmp_cells, obstacles, 1);
      for(int i = 0; i < params.ny; ++i) {
        for(int kk = 0; kk < NSPEEDS; ++kk) {
          (*cells)->speeds[kk][i*params.nx + 2] = tmp_cells2->speeds[kk][i];
          (*cells)->speeds[kk][i*params.nx + (params.nx - 3)] = tmp_cells2->speeds[kk][params.ny + i];
        }
      }
    }

  } else {
    accelerate_flow(params, *cells, obstacles, flag);
    res = merged_timestep_ops(params, *cells, *tmp_cells, obstacles, flag);
    t_speed_arrays *cells_ptr = *cells;
    *cells = *tmp_cells;
    *tmp_cells = cells_ptr;
    //propagate(params, cells, tmp_cells, flag);
    //rebound(params, cells, tmp_cells, obstacles, flag);
    //collision(params, cells, tmp_cells, obstacles, flag);
  }
  return res;
}

int accelerate_flow(const t_param params, t_speed_arrays* cells, int* obstacles, int flag)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  int start, end, increment;
  if(flag == 0) {
    start = 1;
    end = params.nx-1;
    increment = 1;
  } else if(flag == 1) {
    start = 0;
    end = params.nx;
    increment = params.nx - 1;
  } else if(flag == 2) {
    start = 0;
    end = params.nx;
    increment = 1;
  } else {
    start = 0;
    end = params.nx;
    increment = 1;
  }

  for (int ii = start; ii < end; ++ii)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells->speeds[3][ii + jj*params.nx] - w1) > 0.f
        && (cells->speeds[6][ii + jj*params.nx] - w2) > 0.f
        && (cells->speeds[7][ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speeds[1][ii + jj*params.nx] += w1;
      cells->speeds[5][ii + jj*params.nx] += w2;
      cells->speeds[8][ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speeds[3][ii + jj*params.nx] -= w1;
      cells->speeds[6][ii + jj*params.nx] -= w2;
      cells->speeds[7][ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed_arrays* cells, t_speed_arrays* tmp_cells, int flag)
{
  int start, end, increment;
  if(flag == 0) {
    start = 2;
    end = params.nx-2;
    increment = 1;
  } else if(flag == 1) {
    start = 0;
    end = params.nx;
    increment = 1;
  } else {
    start = 0;
    end = params.nx;
    increment = 1;
  }
  /* loop over _all_ cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = start; ii < end; ii += increment)
    {
      if(flag == 1 && ii == 2) {
        ii = params.nx - 2;
      }
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells->speeds[0][ii + jj*params.nx] = cells->speeds[0][ii + jj*params.nx]; /* central cell, no movement */
      tmp_cells->speeds[1][ii + jj*params.nx] = cells->speeds[1][x_w + jj*params.nx]; /* east */
      tmp_cells->speeds[2][ii + jj*params.nx] = cells->speeds[2][ii + y_s*params.nx]; /* north */
      tmp_cells->speeds[3][ii + jj*params.nx] = cells->speeds[3][x_e + jj*params.nx]; /* west */
      tmp_cells->speeds[4][ii + jj*params.nx] = cells->speeds[4][ii + y_n*params.nx]; /* south */
      tmp_cells->speeds[5][ii + jj*params.nx] = cells->speeds[5][x_w + y_s*params.nx]; /* north-east */
      tmp_cells->speeds[6][ii + jj*params.nx] = cells->speeds[6][x_e + y_s*params.nx]; /* north-west */
      tmp_cells->speeds[7][ii + jj*params.nx] = cells->speeds[7][x_e + y_n*params.nx]; /* south-west */
      tmp_cells->speeds[8][ii + jj*params.nx] = cells->speeds[8][x_w + y_n*params.nx]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

float merged_timestep_ops(const t_param params, t_speed_arrays*restrict cells, t_speed_arrays*restrict tmp_cells, int*restrict obstacles, int flag) {
  // merge propagate, rebound, collision and av_velocity
  int start, end, increment;
  if(flag == 0) {
    start = 2;
    end = params.nx-2;
    increment = 1;
  } else if(flag == 1) {
    start = 0;
    end = params.nx;
    increment = 1;
  } else {
    start = 0;
    end = params.nx;
    increment = 1;
  }

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  float tot_u = 0.f;         /* accumulated magnitudes of velocity for each cell */
  /* loop over _all_ cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = start; ii < end; ++ii)
    {


      /*
      t_speed currentVal = cells[jj*params.nx + ii];
      printf("BEFORE: speed1: %d, speed2: %d, speed6: %d\n", currentVal.speed[1],
                                      currentVal.speed[2], currentVal.speed[6]);
      */

      // PROPAGATE STUFF
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells->speeds[0][ii + jj*params.nx] = cells->speeds[0][ii + jj*params.nx]; /* central cell, no movement */
      tmp_cells->speeds[1][ii + jj*params.nx] = cells->speeds[1][x_w + jj*params.nx]; /* east */
      tmp_cells->speeds[2][ii + jj*params.nx] = cells->speeds[2][ii + y_s*params.nx]; /* north */
      tmp_cells->speeds[3][ii + jj*params.nx] = cells->speeds[3][x_e + jj*params.nx]; /* west */
      tmp_cells->speeds[4][ii + jj*params.nx] = cells->speeds[4][ii + y_n*params.nx]; /* south */
      tmp_cells->speeds[5][ii + jj*params.nx] = cells->speeds[5][x_w + y_s*params.nx]; /* north-east */
      tmp_cells->speeds[6][ii + jj*params.nx] = cells->speeds[6][x_e + y_s*params.nx]; /* north-west */
      tmp_cells->speeds[7][ii + jj*params.nx] = cells->speeds[7][x_e + y_n*params.nx]; /* south-west */
      tmp_cells->speeds[8][ii + jj*params.nx] = cells->speeds[8][x_w + y_n*params.nx]; /* south-east */

      // PROPAGATION DONE

      // REBOUND STUFF
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        //t_speed current_cell = tmp_cells[ii + jj*params.nx];
        float current_cell[NSPEEDS];
        for(int kk = 0; kk < NSPEEDS; ++kk) {
          current_cell[kk] = tmp_cells->speeds[kk][ii + jj*params.nx];
        }
        tmp_cells->speeds[1][ii + jj*params.nx] = current_cell[3];
        tmp_cells->speeds[2][ii + jj*params.nx] = current_cell[4];
        tmp_cells->speeds[3][ii + jj*params.nx] = current_cell[1];
        tmp_cells->speeds[4][ii + jj*params.nx] = current_cell[2];
        tmp_cells->speeds[5][ii + jj*params.nx] = current_cell[7];
        tmp_cells->speeds[6][ii + jj*params.nx] = current_cell[8];
        tmp_cells->speeds[7][ii + jj*params.nx] = current_cell[5];
        tmp_cells->speeds[8][ii + jj*params.nx] = current_cell[6];
      }
      // REBOUND DONE

      // COLLISION STUFF
      /* don't consider occupied cells */
      else
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells->speeds[kk][ii + jj*params.nx];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells->speeds[1][ii + jj*params.nx]
                      + tmp_cells->speeds[5][ii + jj*params.nx]
                      + tmp_cells->speeds[8][ii + jj*params.nx]
                      - (tmp_cells->speeds[3][ii + jj*params.nx]
                         + tmp_cells->speeds[6][ii + jj*params.nx]
                         + tmp_cells->speeds[7][ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells->speeds[2][ii + jj*params.nx]
                      + tmp_cells->speeds[5][ii + jj*params.nx]
                      + tmp_cells->speeds[6][ii + jj*params.nx]
                      - (tmp_cells->speeds[4][ii + jj*params.nx]
                         + tmp_cells->speeds[7][ii + jj*params.nx]
                         + tmp_cells->speeds[8][ii + jj*params.nx]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          tmp_cells->speeds[kk][ii + jj*params.nx] = tmp_cells->speeds[kk][ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_cells->speeds[kk][ii + jj*params.nx]);
        }

        //AV VELOCITY CODE
        /* accumulate the norm of x- and y- velocity components */
        if(ii != 0 && ii != params.nx-1) {
          u_x = (tmp_cells->speeds[1][ii + jj*params.nx]
                        + tmp_cells->speeds[5][ii + jj*params.nx]
                        + tmp_cells->speeds[8][ii + jj*params.nx]
                        - (tmp_cells->speeds[3][ii + jj*params.nx]
                           + tmp_cells->speeds[6][ii + jj*params.nx]
                           + tmp_cells->speeds[7][ii + jj*params.nx]))
                       / local_density;
          /* compute y velocity component */
          u_y = (tmp_cells->speeds[2][ii + jj*params.nx]
                        + tmp_cells->speeds[5][ii + jj*params.nx]
                        + tmp_cells->speeds[6][ii + jj*params.nx]
                        - (tmp_cells->speeds[4][ii + jj*params.nx]
                           + tmp_cells->speeds[7][ii + jj*params.nx]
                           + tmp_cells->speeds[8][ii + jj*params.nx]))
                       / local_density;

          tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        }
        //DONE AV_VELOCITY
      }
      // COLLISION DONE

      /*
      currentVal = tmp_cells[jj*params.nx + ii];
      printf("AFTER: speed1: %d, speed2: %d, speed6: %d\n", currentVal.speed[1],
                                      currentVal.speed[2], currentVal.speed[6]);
      */

    }
  }

  return tot_u;
}

int rebound(const t_param params, t_speed_arrays* cells, t_speed_arrays* tmp_cells, int* obstacles, int flag)
{
  int start, end, increment;
  if(flag == 0) {
    start = 2;
    end = params.nx-2;
    increment = 1;
  } else if(flag == 1) {
    start = 0;
    end = params.nx;
    increment = 1;
  } else {
    start = 0;
    end = params.nx;
    increment = 1;
  }
  /* loop over the cells in the grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = start; ii < end; ii += increment)
    {
      if(flag == 1 && ii == 2) {
        ii = params.nx - 2;
      }
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        cells->speeds[1][ii + jj*params.nx] = tmp_cells->speeds[3][ii + jj*params.nx];
        cells->speeds[2][ii + jj*params.nx] = tmp_cells->speeds[4][ii + jj*params.nx];
        cells->speeds[3][ii + jj*params.nx] = tmp_cells->speeds[1][ii + jj*params.nx];
        cells->speeds[4][ii + jj*params.nx] = tmp_cells->speeds[2][ii + jj*params.nx];
        cells->speeds[5][ii + jj*params.nx] = tmp_cells->speeds[7][ii + jj*params.nx];
        cells->speeds[6][ii + jj*params.nx] = tmp_cells->speeds[8][ii + jj*params.nx];
        cells->speeds[7][ii + jj*params.nx] = tmp_cells->speeds[5][ii + jj*params.nx];
        cells->speeds[8][ii + jj*params.nx] = tmp_cells->speeds[6][ii + jj*params.nx];
      }
    }
  }

  return EXIT_SUCCESS;
}

void test_vels(const char* output_file, float *vels, int steps) {
  FILE* fp = fopen(output_file, "w");
  for(int i = 0; i < steps; ++i) {
    float vel = vels[i];
    fprintf(fp, "%.12lf\n", vel);
  }

  fclose(fp);
}

int collision(const t_param params, t_speed_arrays* cells, t_speed_arrays* tmp_cells, int* obstacles, int flag)
{
  int start, end, increment;
  if(flag == 0) {
    start = 2;
    end = params.nx-2;
    increment = 1;
  } else if(flag == 1) {
    start = 0;
    end = params.nx;
    increment = 1;
  } else {
    start = 0;
    end = params.nx;
    increment = 1;
  }
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = start; ii < end; ii += increment)
    {
      if(flag == 1 && ii == 2) {
        ii = params.nx - 2;
      }
      /* don't consider occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells->speeds[kk][ii + jj*params.nx];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells->speeds[1][ii + jj*params.nx]
                      + tmp_cells->speeds[5][ii + jj*params.nx]
                      + tmp_cells->speeds[8][ii + jj*params.nx]
                      - (tmp_cells->speeds[3][ii + jj*params.nx]
                         + tmp_cells->speeds[6][ii + jj*params.nx]
                         + tmp_cells->speeds[7][ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells->speeds[2][ii + jj*params.nx]
                      + tmp_cells->speeds[5][ii + jj*params.nx]
                      + tmp_cells->speeds[6][ii + jj*params.nx]
                      - (tmp_cells->speeds[4][ii + jj*params.nx]
                         + tmp_cells->speeds[7][ii + jj*params.nx]
                         + tmp_cells->speeds[8][ii + jj*params.nx]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells->speeds[kk][ii + jj*params.nx] = tmp_cells->speeds[kk][ii + jj*params.nx]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_cells->speeds[kk][ii + jj*params.nx]);
        }
      }
    }
  }

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed_arrays* cells, int* obstacles, int flag)
{
  int start, end, increment;
  if(flag == 0) {
    start = 2;
    end = params.nx-2;
    increment = 1;
  } else if(flag == 1) {
    start = 1;
    end = params.nx-1;
    increment = params.nx-3;
  } else {
    start = 1;
    end = params.nx-1;
    increment = 1;
  }

  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = start; ii < end; ++ii)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells->speeds[kk][ii + jj*params.nx];
        }

        /* x-component of velocity */
        float u_x = (cells->speeds[1][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[8][ii + jj*params.nx]
                      - (cells->speeds[3][ii + jj*params.nx]
                         + cells->speeds[6][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speeds[2][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[6][ii + jj*params.nx]
                      - (cells->speeds[4][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]
                         + cells->speeds[8][ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  //return tot_u / (float)tot_cells;
  return tot_u;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_arrays** cells_ptr, t_speed_arrays** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = create_t_speed_arrays(*params);

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = create_t_speed_arrays(*params);

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)->speeds[0][ii + jj*params->nx] = w0;
      /* axis directions */
      (*cells_ptr)->speeds[1][ii + jj*params->nx] = w1;
      (*cells_ptr)->speeds[2][ii + jj*params->nx] = w1;
      (*cells_ptr)->speeds[3][ii + jj*params->nx] = w1;
      (*cells_ptr)->speeds[4][ii + jj*params->nx] = w1;
      /* diagonals */
      (*cells_ptr)->speeds[5][ii + jj*params->nx] = w2;
      (*cells_ptr)->speeds[6][ii + jj*params->nx] = w2;
      (*cells_ptr)->speeds[7][ii + jj*params->nx] = w2;
      (*cells_ptr)->speeds[8][ii + jj*params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed_arrays** cells_ptr, t_speed_arrays** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free_t_speed_arrays(*cells_ptr);
  *cells_ptr = NULL;

  free_t_speed_arrays(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed_arrays* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  float value = 0;
  value += av_velocity(params, cells, obstacles, 0) * params.reynolds_dim / viscosity;
  value += av_velocity(params, cells, obstacles, 1) * params.reynolds_dim / viscosity;

  return value;
}

float total_density(const t_param params, t_speed_arrays* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells->speeds[kk][ii + jj*params.nx];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed_arrays* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells->speeds[kk][ii + jj*params.nx];
        }

        /* compute x velocity component */
        u_x = (cells->speeds[1][ii + jj*params.nx]
               + cells->speeds[5][ii + jj*params.nx]
               + cells->speeds[8][ii + jj*params.nx]
               - (cells->speeds[3][ii + jj*params.nx]
                  + cells->speeds[6][ii + jj*params.nx]
                  + cells->speeds[7][ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells->speeds[2][ii + jj*params.nx]
               + cells->speeds[5][ii + jj*params.nx]
               + cells->speeds[6][ii + jj*params.nx]
               - (cells->speeds[4][ii + jj*params.nx]
                  + cells->speeds[7][ii + jj*params.nx]
                  + cells->speeds[8][ii + jj*params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
