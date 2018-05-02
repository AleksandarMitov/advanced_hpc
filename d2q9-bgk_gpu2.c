
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
//const int ASYNC_HALOS = 0;
//const int SPREAD_COLS_EVENLY = 1;
//const int MERGE_TIMESTEP = 1;
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
//#pragma omp declare target
int timestep(const t_param params, int* obstacles, int flag,
  float *a,float *b,float *c,float *d,float *e,float *f,float *g,float *h,float *i,
  float *at,float *bt,float *ct,float *dt,float *et,float *ft,float *gt,float *ht,float *it);
int accelerate_flow(const t_param params, int* obstacles, int flag, float *a,
float *b,float *c,float *d,float *e,float *f,float *g,float *h,float *i);
float merged_timestep_ops(const t_param params, int*restrict obstacles, int flag,
    float *a, float *b,float *c,float *d,float *e,float *f,float *g,float *h,float *i,
    float *at, float *bt,float *ct,float *dt,float *et,float *ft,float *gt,float *ht,float *it);
    /* compute average velocity */
    float av_velocity(const t_param params, int* obstacles, int flag, float* tot_u_buffer,
        float* a,float* b,float* c,float* d,float* e,float* f,float* g,float* h,float* i);
        /*void exchange_halos(int rank, int size, t_param child_params,
              float* sbuffer_cells, float* rbuffer_cells,
              float* a,float* b,float* c,float* d,float* e,float* f,float* g,float* h,float* i);
              extern int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                          int dest, int sendtag,
                          void *recvbuf, int recvcount, MPI_Datatype recvtype,
                          int source, int recvtag,
                          MPI_Comm comm, MPI_Status *status); */

//#pragma omp end declare target

int write_values(const t_param params, t_speed_arrays* cells, int* obstacles, float* av_vels);
void initialise_params_from_file(const char* paramfile, t_param* params);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed_arrays** cells_ptr, t_speed_arrays** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed_arrays* cells);



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
  //t_speed_arrays *old_cell_vals;
//  MPI_Request** requests;

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
  //old_cell_vals = create_t_speed_arrays(child_params);
  //requests = (MPI_Request **) malloc(4*sizeof(MPI_Request*));  // for async halo exchange

  if(rank == 0) {
    printf("Number of processes: %d\n", size);
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

  float* tmp;
  float *a, *b, *c, *d, *e, *f, *g, *h, *i;
  float *at, *bt, *ct, *dt, *et, *ft, *gt, *ht, *it;

  a = child_cells->speeds[0];
  b = child_cells->speeds[1];
  c = child_cells->speeds[2];
  d = child_cells->speeds[3];
  e = child_cells->speeds[4];
  f = child_cells->speeds[5];
  g = child_cells->speeds[6];
  h = child_cells->speeds[7];
  i = child_cells->speeds[8];

  at = child_tmp_cells->speeds[0];
  bt = child_tmp_cells->speeds[1];
  ct = child_tmp_cells->speeds[2];
  dt = child_tmp_cells->speeds[3];
  et = child_tmp_cells->speeds[4];
  ft = child_tmp_cells->speeds[5];
  gt = child_tmp_cells->speeds[6];
  ht = child_tmp_cells->speeds[7];
  it = child_tmp_cells->speeds[8];
  int N = child_params.nx * child_params.ny;
  int iters = child_params.maxIters;
  float *tot_u_buffer = (float*) malloc(sizeof(float));
#pragma omp target enter data map(to:  child_obstacles[0:N], \
   tot_u_buffer[0:1], \
   a[0:N], b[0:N],c[0:N],d[0:N],e[0:N],f[0:N],g[0:N], \
   h[0:N],i[0:N],at[0:N],bt[0:N],ct[0:N],dt[0:N],et[0:N],ft[0:N],gt[0:N],ht[0:N],it[0:N])
{}

  //#pragma omp target
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    //output_state(file_name, tt, process_cells, process_obstacles, process_params.nx, process_params.ny);
    if(rank == 0 && tt % 500 == 0) printf("iteration: %d\n", tt);


      if(rank == 0 && tt == 0) printf("Flag: 2\n");
      //Exchange halos
      //exchange_halos(rank, size, child_params, sbuffer_cells1, rbuffer_cells1,
      //                                                        a,b,c,d,e,f,g,h,i);
      //now do computations
      //timestep(child_params, &child_cells, &child_tmp_cells, child_obstacles, 2);
      timestep(child_params, child_obstacles, 2,
              a,b,c,d,e,f,g,h,i, at,bt,ct,dt,et,ft,gt,ht,it);
      child_vels[tt] = av_velocity(child_params, child_obstacles, 2, tot_u_buffer,
              at,bt,ct,dt,et,ft,gt,ht,it);

      tmp = a;
      a = at;
      at = tmp;

      tmp = b;
      b = bt;
      bt = tmp;

      tmp = c;
      c = ct;
      ct = tmp;

      tmp = d;
      d = dt;
      dt = tmp;

      tmp = e;
      e = et;
      et = tmp;

      tmp = f;
      f = ft;
      ft = tmp;

      tmp = g;
      g = gt;
      gt = tmp;

      tmp = h;
      h = ht;
      ht = tmp;

      tmp = i;
      i = it;
      it = tmp;

      /*
      t_speed_arrays* tmp = child_cells;
      child_cells = child_tmp_cells;
      child_tmp_cells = tmp;
      */

#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
    if(TEST && rank == 0 && tt % 500 == 0) {
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", child_vels[tt]);
    }
  }

  #pragma omp target exit data map(from: \
     a[0:N], b[0:N],c[0:N],d[0:N],e[0:N],f[0:N],g[0:N], h[0:N], i[0:N])
  {}

     child_cells->speeds[0] = a;
     child_cells->speeds[1] = b;
     child_cells->speeds[2] = c;
     child_cells->speeds[3] = d;
     child_cells->speeds[4] = e;
     child_cells->speeds[5] = f;
     child_cells->speeds[6] = g;
     child_cells->speeds[7] = h;
     child_cells->speeds[8] = i;

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

    int remainder_cols = n % size;
    int spreaded_remainder_before_process = min(rank, remainder_cols);
    start_from = per_process*rank + spreaded_remainder_before_process;

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
        sbuffer_cells1[row*speeds_to_send + 0] = child_cells->speeds[3][row*child_params.nx + 1];
        sbuffer_cells1[row*speeds_to_send + 1] = child_cells->speeds[6][row*child_params.nx + 1];
        sbuffer_cells1[row*speeds_to_send + 2] = child_cells->speeds[7][row*child_params.nx + 1];
  }
  MPI_Isend(sbuffer_cells1, child_params.ny*speeds_to_send, MPI_FLOAT, left, 0, MPI_COMM_WORLD, requests[0]);

  //send to right, receive from left
  //fill with right col
  for(int row = 0; row < child_params.ny; ++row) {
      sbuffer_cells2[row*speeds_to_send + 0] = child_cells->speeds[1][row*child_params.nx + (child_params.nx - 2)];
      sbuffer_cells2[row*speeds_to_send + 1] = child_cells->speeds[5][row*child_params.nx + (child_params.nx - 2)];
      sbuffer_cells2[row*speeds_to_send + 2] = child_cells->speeds[8][row*child_params.nx + (child_params.nx - 2)];
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

void exchange_halos(int rank, int size, t_param child_params,
                      float* sbuffer_cells, float* rbuffer_cells,
    float* a,float* b,float* c,float* d,float* e,float* f,float* g,float* h,float* i) {
/*
  float *a = child_cells->speeds[0];
  float *b = child_cells->speeds[1];
  float *c = child_cells->speeds[2];
  float *d = child_cells->speeds[3];
  float *e = child_cells->speeds[4];
  float *f = child_cells->speeds[5];
  float *g = child_cells->speeds[6];
  float *h = child_cells->speeds[7];
  float *i = child_cells->speeds[8];
  */

  int left = (rank == 0) ? (rank + size - 1) : (rank - 1); // left is bottom, right is top equiv
  int right = (rank + 1) % size;
  //send to the left, receive from right
  //fill with left col
  for(int row = 0; row < child_params.ny; ++row) {
    sbuffer_cells[row*NSPEEDS + 0] = a[row*child_params.nx + 1];
    sbuffer_cells[row*NSPEEDS + 1] = b[row*child_params.nx + 1];
    sbuffer_cells[row*NSPEEDS + 2] = c[row*child_params.nx + 1];
    sbuffer_cells[row*NSPEEDS + 3] = d[row*child_params.nx + 1];
    sbuffer_cells[row*NSPEEDS + 4] = e[row*child_params.nx + 1];
    sbuffer_cells[row*NSPEEDS + 5] = f[row*child_params.nx + 1];
    sbuffer_cells[row*NSPEEDS + 6] = g[row*child_params.nx + 1];
    sbuffer_cells[row*NSPEEDS + 7] = h[row*child_params.nx + 1];
    sbuffer_cells[row*NSPEEDS + 8] = i[row*child_params.nx + 1];
    /*
    for(int speed = 0; speed < NSPEEDS; ++speed) {
      sbuffer_cells[row*NSPEEDS + speed] = child_cells->speeds[speed][row*child_params.nx + 1];
    }*/
  }

  MPI_Sendrecv(sbuffer_cells, child_params.ny*NSPEEDS, MPI_FLOAT, left, 0, rbuffer_cells,
              child_params.ny*NSPEEDS, MPI_FLOAT, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //populate right col
  for(int row = 0; row < child_params.ny; ++row) {
    //t_speed speeds;
    a[row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells[row*NSPEEDS + 0];
    b[row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells[row*NSPEEDS + 1];
    c[row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells[row*NSPEEDS + 2];
    d[row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells[row*NSPEEDS + 3];
    e[row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells[row*NSPEEDS + 4];
    f[row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells[row*NSPEEDS + 5];
    g[row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells[row*NSPEEDS + 6];
    h[row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells[row*NSPEEDS + 7];
    i[row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells[row*NSPEEDS + 8];
    /*
    for(int speed = 0; speed < NSPEEDS; ++speed) {
      child_cells->speeds[speed][row*child_params.nx + (child_params.nx - 1)] = rbuffer_cells[row*NSPEEDS + speed];
      //speeds.speeds[speed] = rbuffer_cells[row*NSPEEDS + speed];
    }*/

  }
  //send to right, receive from left
  //fill with right col
  for(int row = 0; row < child_params.ny; ++row) {
    sbuffer_cells[row*NSPEEDS + 0] = a[row*child_params.nx + (child_params.nx - 2)];
    sbuffer_cells[row*NSPEEDS + 1] = b[row*child_params.nx + (child_params.nx - 2)];
    sbuffer_cells[row*NSPEEDS + 2] = c[row*child_params.nx + (child_params.nx - 2)];
    sbuffer_cells[row*NSPEEDS + 3] = d[row*child_params.nx + (child_params.nx - 2)];
    sbuffer_cells[row*NSPEEDS + 4] = e[row*child_params.nx + (child_params.nx - 2)];
    sbuffer_cells[row*NSPEEDS + 5] = f[row*child_params.nx + (child_params.nx - 2)];
    sbuffer_cells[row*NSPEEDS + 6] = g[row*child_params.nx + (child_params.nx - 2)];
    sbuffer_cells[row*NSPEEDS + 7] = h[row*child_params.nx + (child_params.nx - 2)];
    sbuffer_cells[row*NSPEEDS + 8] = i[row*child_params.nx + (child_params.nx - 2)];

    /*
    for(int speed = 0; speed < NSPEEDS; ++speed) {
      sbuffer_cells[row*NSPEEDS + speed] = child_cells->speeds[speed][row*child_params.nx + (child_params.nx - 2)];
    }*/
  }
  MPI_Sendrecv(sbuffer_cells, child_params.ny*NSPEEDS, MPI_FLOAT, right, 0, rbuffer_cells,
              child_params.ny*NSPEEDS, MPI_FLOAT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //populate left col
  for(int row = 0; row < child_params.ny; ++row) {
    //t_speed speeds;
    a[row*child_params.nx] = rbuffer_cells[row*NSPEEDS + 0];
    b[row*child_params.nx] = rbuffer_cells[row*NSPEEDS + 1];
    c[row*child_params.nx] = rbuffer_cells[row*NSPEEDS + 2];
    d[row*child_params.nx] = rbuffer_cells[row*NSPEEDS + 3];
    e[row*child_params.nx] = rbuffer_cells[row*NSPEEDS + 4];
    f[row*child_params.nx] = rbuffer_cells[row*NSPEEDS + 5];
    g[row*child_params.nx] = rbuffer_cells[row*NSPEEDS + 6];
    h[row*child_params.nx] = rbuffer_cells[row*NSPEEDS + 7];
    i[row*child_params.nx] = rbuffer_cells[row*NSPEEDS + 8];
    /*
    for(int speed = 0; speed < NSPEEDS; ++speed) {
      //speeds.speeds[speed] = rbuffer_cells[row*NSPEEDS + speed];
      child_cells->speeds[speed][row*child_params.nx] = rbuffer_cells[row*NSPEEDS + speed];
    }
    */
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

    int per_process = nx / size;
    int remainder_cols = nx % size;
    ncols = per_process;
    if(rank < remainder_cols) {
      ++ncols;
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


int timestep(const t_param params, int* obstacles, int flag,
  float *a,float *b,float *c,float *d,float *e,float *f,float *g,float *h,float *i,
  float *at,float *bt,float *ct,float *dt,float *et,float *ft,float *gt,float *ht,float *it)
{
  /*
  float *a = (*cells)->speeds[0];
  float *b = (*cells)->speeds[1];
  float *c = (*cells)->speeds[2];
  float *d = (*cells)->speeds[3];
  float *e = (*cells)->speeds[4];
  float *f = (*cells)->speeds[5];
  float *g = (*cells)->speeds[6];
  float *h = (*cells)->speeds[7];
  float *i = (*cells)->speeds[8];

  float *at = (*tmp_cells)->speeds[0];
  float *bt = (*tmp_cells)->speeds[1];
  float *ct = (*tmp_cells)->speeds[2];
  float *dt = (*tmp_cells)->speeds[3];
  float *et = (*tmp_cells)->speeds[4];
  float *ft = (*tmp_cells)->speeds[5];
  float *gt = (*tmp_cells)->speeds[6];
  float *ht = (*tmp_cells)->speeds[7];
  float *it = (*tmp_cells)->speeds[8];
  */


  accelerate_flow(params, obstacles, flag, a,b,c,d,e,f,g,h,i);
  merged_timestep_ops(params, obstacles, flag,a,b,c,d,e,f,g,h,i,at,bt,ct,dt,et,ft,gt,ht,it);
  /*
  t_speed_arrays *cells_ptr = *cells;
  *cells = *tmp_cells;
  *tmp_cells = cells_ptr;
  */
  //propagate(params, cells, tmp_cells, flag);
  //rebound(params, cells, tmp_cells, obstacles, flag);
  //collision(params, cells, tmp_cells, obstacles, flag);

return 0;
}

/*
float timestep_async(const t_param params, t_speed_arrays** cells, t_speed_arrays** tmp_cells, int* obstacles, int flag, t_speed_arrays *tmp_cells2, int total_requests, MPI_Request **requests)
{
  float res = -1;
    accelerate_flow(params, *cells, obstacles, flag);
    res = merged_timestep_ops(params, *cells, *tmp_cells, obstacles, flag);
    t_speed_arrays *cells_ptr = *cells;
    *cells = *tmp_cells;
    *tmp_cells = cells_ptr;
    //propagate(params, cells, tmp_cells, flag);
    //rebound(params, cells, tmp_cells, obstacles, flag);
    //collision(params, cells, tmp_cells, obstacles, flag);

  return res;
}
*/

int accelerate_flow(const t_param params, int* obstacles, int flag, float *a,
    float *b,float *c,float *d,float *e,float *f,float *g,float *h,float *i)
{
  /*
  float *a = cells->speeds[0];
  float *b = cells->speeds[1];
  float *c = cells->speeds[2];
  float *d = cells->speeds[3];
  float *e = cells->speeds[4];
  float *f = cells->speeds[5];
  float *g = cells->speeds[6];
  float *h = cells->speeds[7];
  float *i = cells->speeds[8];
  */

  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;
  int nx = params.nx;

#pragma omp target teams distribute parallel for simd
    for (int ii = 0; ii < nx; ++ii)
    {
      /* if the cell is not occupied and
      ** we don't send a negative density */
      if (!obstacles[ii + jj*nx]
          && (d[ii + jj*nx] - w1) > 0.f
          && (g[ii + jj*nx] - w2) > 0.f
          && (h[ii + jj*nx] - w2) > 0.f)
      {
        /* increase 'east-side' densities */
        b[ii + jj*nx] += w1;
        f[ii + jj*nx] += w2;
        i[ii + jj*nx] += w2;
        /* decrease 'west-side' densities */
        d[ii + jj*nx] -= w1;
        g[ii + jj*nx] -= w2;
        h[ii + jj*nx] -= w2;
      }
    }

  return EXIT_SUCCESS;
}

float merged_timestep_ops(const t_param params, int*restrict obstacles, int flag,
    float *a, float *b,float *c,float *d,float *e,float *f,float *g,float *h,float *i,
    float *at, float *bt,float *ct,float *dt,float *et,float *ft,float *gt,float *ht,float *it) {
  /*
  float *a = cells->speeds[0];
  float *b = cells->speeds[1];
  float *c = cells->speeds[2];
  float *d = cells->speeds[3];
  float *e = cells->speeds[4];
  float *f = cells->speeds[5];
  float *g = cells->speeds[6];
  float *h = cells->speeds[7];
  float *i = cells->speeds[8];

  float *at = tmp_cells->speeds[0];
  float *bt = tmp_cells->speeds[1];
  float *ct = tmp_cells->speeds[2];
  float *dt = tmp_cells->speeds[3];
  float *et = tmp_cells->speeds[4];
  float *ft = tmp_cells->speeds[5];
  float *gt = tmp_cells->speeds[6];
  float *ht = tmp_cells->speeds[7];
  float *it = tmp_cells->speeds[8];
  */
  int N = params.nx * params.ny;
  // merge propagate, rebound, collision and av_velocity

int nx = params.nx;
int ny = params.ny;
float omega = params.omega;

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  //float tot_u = 0.f;         /* accumulated magnitudes of velocity for each cell */
  /* loop over _all_ cells */
  //#pragma omp target enter data map(to: a[0:N], b[0:N],c[0:N],d[0:N],e[0:N],f[0:N],g[0:N], \
    //h[0:N],i[0:N],at[0:N],bt[0:N],ct[0:N],dt[0:N],et[0:N],ft[0:N],gt[0:N],ht[0:N],it[0:N])
#pragma omp target teams distribute parallel for simd collapse(2)
for (int jj = 0; jj < ny; jj++)
{

  ///#pragma omp parallel for simd
  for (int ii = 0; ii < nx; ++ii)
  {

    /*
    t_speed currentVal = cells[jj*nx + ii];
    printf("BEFORE: speed1: %d, speed2: %d, speed6: %d\n", currentVal.speed[1],
                                    currentVal.speed[2], currentVal.speed[6]);
    */

    // PROPAGATE STUFF
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int y_n = (jj + 1) % ny;
    int x_e = (ii + 1) % nx;
    int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
    int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);
    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    at[ii + jj*nx] = a[ii + jj*nx]; /* central cell, no movement */
    bt[ii + jj*nx] = b[x_w + jj*nx]; /* east */
    ct[ii + jj*nx] = c[ii + y_s*nx]; /* north */
    dt[ii + jj*nx] = d[x_e + jj*nx]; /* west */
    et[ii + jj*nx] = e[ii + y_n*nx]; /* south */
    ft[ii + jj*nx] = f[x_w + y_s*nx]; /* north-east */
    gt[ii + jj*nx] = g[x_e + y_s*nx]; /* north-west */
    ht[ii + jj*nx] = h[x_e + y_n*nx]; /* south-west */
    it[ii + jj*nx] = i[x_w + y_n*nx]; /* south-east */

    /*
    tmp_cells->speeds[0][ii + jj*nx] = cells->speeds[0][ii + jj*nx];
    tmp_cells->speeds[1][ii + jj*nx] = cells->speeds[1][x_w + jj*nx];
    tmp_cells->speeds[2][ii + jj*nx] = cells->speeds[2][ii + y_s*nx];
    tmp_cells->speeds[3][ii + jj*nx] = cells->speeds[3][x_e + jj*nx];
    tmp_cells->speeds[4][ii + jj*nx] = cells->speeds[4][ii + y_n*nx];
    tmp_cells->speeds[5][ii + jj*nx] = cells->speeds[5][x_w + y_s*nx];
    tmp_cells->speeds[6][ii + jj*nx] = cells->speeds[6][x_e + y_s*nx];
    tmp_cells->speeds[7][ii + jj*nx] = cells->speeds[7][x_e + y_n*nx];
    tmp_cells->speeds[8][ii + jj*nx] = cells->speeds[8][x_w + y_n*nx];
    */
    // PROPAGATION DONE

    // REBOUND STUFF
    /* if the cell contains an obstacle */
    if (obstacles[jj*nx + ii])
    {
      /* called after propagate, so taking values from scratch space
      ** mirroring, and writing into main grid */
      //t_speed current_cell = tmp_cells[ii + jj*nx];
      float current_cell[NSPEEDS];
      current_cell[0] = at[ii + jj*nx];
      current_cell[1] = bt[ii + jj*nx];
      current_cell[2] = ct[ii + jj*nx];
      current_cell[3] = dt[ii + jj*nx];
      current_cell[4] = et[ii + jj*nx];
      current_cell[5] = ft[ii + jj*nx];
      current_cell[6] = gt[ii + jj*nx];
      current_cell[7] = ht[ii + jj*nx];
      current_cell[8] = it[ii + jj*nx];
      /*
      for(int kk = 0; kk < NSPEEDS; ++kk) {
        current_cell[kk] = tmp_cells->speeds[kk][ii + jj*nx];
      }
      */


      bt[ii + jj*nx] = current_cell[3];
      ct[ii + jj*nx] = current_cell[4];
      dt[ii + jj*nx] = current_cell[1];
      et[ii + jj*nx] = current_cell[2];
      ft[ii + jj*nx] = current_cell[7];
      gt[ii + jj*nx] = current_cell[8];
      ht[ii + jj*nx] = current_cell[5];
      it[ii + jj*nx] = current_cell[6];

      /*
      tmp_cells->speeds[1][ii + jj*nx] = current_cell[3];
      tmp_cells->speeds[2][ii + jj*nx] = current_cell[4];
      tmp_cells->speeds[3][ii + jj*nx] = current_cell[1];
      tmp_cells->speeds[4][ii + jj*nx] = current_cell[2];
      tmp_cells->speeds[5][ii + jj*nx] = current_cell[7];
      tmp_cells->speeds[6][ii + jj*nx] = current_cell[8];
      tmp_cells->speeds[7][ii + jj*nx] = current_cell[5];
      tmp_cells->speeds[8][ii + jj*nx] = current_cell[6];
      */
    }
    // REBOUND DONE

    // COLLISION STUFF
    /* don't consider occupied cells */
    else
    {
      /* compute local density total */
      float local_density = 0.f;
      local_density += at[ii + jj*nx];
      local_density += bt[ii + jj*nx];
      local_density += ct[ii + jj*nx];
      local_density += dt[ii + jj*nx];
      local_density += et[ii + jj*nx];
      local_density += ft[ii + jj*nx];
      local_density += gt[ii + jj*nx];
      local_density += ht[ii + jj*nx];
      local_density += it[ii + jj*nx];

      /*
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        local_density += tmp_cells->speeds[kk][ii + jj*nx];
      }
      */

      /* compute x velocity component */
      /*
      float u_x = (bt[ii + jj*nx]
                    + ft[ii + jj*nx]
                    + it[ii + jj*nx]
                    - (dt[ii + jj*nx]
                       + ft[ii + jj*nx]
                       + gt[ii + jj*nx]))
                   / local_density;

      float u_y = (ct[ii + jj*nx]
                    + ft[ii + jj*nx]
                    + gt[ii + jj*nx]
                    - (et[ii + jj*nx]
                       + ht[ii + jj*nx]
                       + it[ii + jj*nx]))
                   / local_density;
      */

      float u_x = (bt[ii + jj*nx]
                    + ft[ii + jj*nx]
                    + it[ii + jj*nx]
                    - (dt[ii + jj*nx]
                       + gt[ii + jj*nx]
                       + ht[ii + jj*nx]))
                   / local_density;

      float u_y = (ct[ii + jj*nx]
                    + ft[ii + jj*nx]
                    + gt[ii + jj*nx]
                    - (et[ii + jj*nx]
                       + ht[ii + jj*nx]
                       + it[ii + jj*nx]))
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
      at[ii + jj*nx] = at[ii + jj*nx]+ omega * (d_equ[0] - at[ii + jj*nx]);
      bt[ii + jj*nx] = bt[ii + jj*nx]+ omega * (d_equ[1] - bt[ii + jj*nx]);
      ct[ii + jj*nx] = ct[ii + jj*nx]+ omega * (d_equ[2] - ct[ii + jj*nx]);
      dt[ii + jj*nx] = dt[ii + jj*nx]+ omega * (d_equ[3] - dt[ii + jj*nx]);
      et[ii + jj*nx] = et[ii + jj*nx]+ omega * (d_equ[4] - et[ii + jj*nx]);
      ft[ii + jj*nx] = ft[ii + jj*nx]+ omega * (d_equ[5] - ft[ii + jj*nx]);
      gt[ii + jj*nx] = gt[ii + jj*nx]+ omega * (d_equ[6] - gt[ii + jj*nx]);
      ht[ii + jj*nx] = ht[ii + jj*nx]+ omega * (d_equ[7] - ht[ii + jj*nx]);
      it[ii + jj*nx] = it[ii + jj*nx]+ omega * (d_equ[8] - it[ii + jj*nx]);
      /*
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        tmp_cells->speeds[kk][ii + jj*nx] = tmp_cells->speeds[kk][ii + jj*nx]
                                                + params.omega
                                                * (d_equ[kk] - tmp_cells->speeds[kk][ii + jj*nx]);
      }
      */


    }
    // COLLISION DONE

    /*
    currentVal = tmp_cells[jj*nx + ii];
    printf("AFTER: speed1: %d, speed2: %d, speed6: %d\n", currentVal.speed[1],
                                    currentVal.speed[2], currentVal.speed[6]);
    */

  }
}
  //#pragma omp target exit data map(from: a[0:N], b[0:N],c[0:N],d[0:N],e[0:N],f[0:N],g[0:N], \
    //h[0:N],i[0:N],at[0:N],bt[0:N],ct[0:N],dt[0:N],et[0:N],ft[0:N],gt[0:N],ht[0:N],it[0:N])
  return 0;
}


void test_vels(const char* output_file, float *vels, int steps) {
  FILE* fp = fopen(output_file, "w");
  for(int i = 0; i < steps; ++i) {
    float vel = vels[i];
    fprintf(fp, "%.12lf\n", vel);
  }

  fclose(fp);
}

float av_velocity(const t_param params, int* obstacles, int flag, float* tot_u_buffer,
    float* a,float* b,float* c,float* d,float* e,float* f,float* g,float* h,float* i)
{
  /*
  float *a = cells->speeds[0];
  float *b = cells->speeds[1];
  float *c = cells->speeds[2];
  float *d = cells->speeds[3];
  float *e = cells->speeds[4];
  float *f = cells->speeds[5];
  float *g = cells->speeds[6];
  float *h = cells->speeds[7];
  float *i = cells->speeds[8];
  */

  int start, end, increment;
  start = 1;
  end = params.nx-1;
  increment = 1;

  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;
  int ny = params.ny;
  int nx = params.nx;

  tot_u_buffer[0] = 0.0;
  /* loop over all non-blocked cells */
#pragma omp target update to(tot_u_buffer[0:1])
{}
#pragma omp target teams distribute parallel for simd collapse(2) \
                                        reduction(+ : tot_u_buffer[0])
  for (int jj = 0; jj < ny; jj++)
  {
    //#pragma omp parallel for simd
    for (int ii = start; ii < end; ii += increment)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*nx])
      {
        /* local density total */
        float local_density = 0.f;
        local_density += a[ii + jj*nx];
        local_density += b[ii + jj*nx];
        local_density += c[ii + jj*nx];
        local_density += d[ii + jj*nx];
        local_density += e[ii + jj*nx];
        local_density += f[ii + jj*nx];
        local_density += g[ii + jj*nx];
        local_density += h[ii + jj*nx];
        local_density += i[ii + jj*nx];
        /*
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells->speeds[kk][ii + jj*params.nx];
        } */

        /* x-component of velocity */
        float u_x = (b[ii + jj*nx]
                      + f[ii + jj*nx]
                      + i[ii + jj*nx]
                      - (d[ii + jj*nx]
                         + g[ii + jj*nx]
                         + h[ii + jj*nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (c[ii + jj*nx]
                      + f[ii + jj*nx]
                      + g[ii + jj*nx]
                      - (e[ii + jj*nx]
                         + h[ii + jj*nx]
                         + i[ii + jj*nx]))
                     / local_density;
        /*

        float u_x = (cells->speeds[1][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[8][ii + jj*params.nx]
                      - (cells->speeds[3][ii + jj*params.nx]
                         + cells->speeds[6][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]))
                     / local_density;

        float u_y = (cells->speeds[2][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[6][ii + jj*params.nx]
                      - (cells->speeds[4][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]
                         + cells->speeds[8][ii + jj*params.nx]))
                     / local_density;
        */
        /* accumulate the norm of x- and y- velocity components */
        tot_u_buffer[0] += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        //++tot_cells;
      }
    }
  }
  #pragma omp target update from(tot_u_buffer[0:1])
  {}
  tot_u = tot_u_buffer[0];
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

/*
float calc_reynolds(const t_param params, t_speed_arrays* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  float value = 0;
  value += av_velocity(params, cells, obstacles, 0) * params.reynolds_dim / viscosity;
  value += av_velocity(params, cells, obstacles, 1) * params.reynolds_dim / viscosity;

  return value;
}*/

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
