#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  float *a;
  float *b;
  float *c;
} data;

void die(const char *message, const int line, const char *file);

void initialise(data *d_ptr, const int N);
void finalise(data *d_ptr, const int N);

int main(int argc, char const *argv[]) {
  int N = 1024; /* vector size */
  int num_iterations = 10;
  data d;
  initialise(&d, N);

  // Set values of a and b on the host
  for (int i = 0; i < N; i++) {
    d.a[i] = 1.f;
    d.b[i] = 2.f;
  }

  //NOTE HERE WE ARE USING LOCALLY STORED POINTERS
  float *a = d.a;
  float *b = d.b;
  float *c = d.c;

  // Copy the values to the device
  #pragma omp target update to(a[0:N], b[0:N])
  {}

  for (int itr = 0; itr < num_iterations; itr++) {
  // Execute vecadd on the target device
#pragma omp target teams distribute parallel for
    for (int i = 0; i < N; i++) {
      c[i] = a[i] + b[i];
    }
  }

  // Copy the result from the device
  #pragma omp target update from(c[0:N])
  {}

  // Verify the results
  int correct_results = 1;
  for (int i = 0; i < N; i++) {
    if (fabs(d.c[i] - 3.f) > 0.00001f) {
      printf("Incorrect answer at index %d\n", i);
      correct_results = 0;
    }
  }

  if (correct_results) {
    printf("Success!\n");
  }

  finalise(&d, N);
  return 0;
}

void initialise(data *d_ptr, const int N) {
  // Initialise the arrays on the host
  d_ptr->a = malloc(sizeof(float) * N);
  if (d_ptr->a == NULL)
    die("cannot allocate memory for a", __LINE__, __FILE__);
  d_ptr->b = malloc(sizeof(float) * N);
  if (d_ptr->b == NULL)
    die("cannot allocate memory for b", __LINE__, __FILE__);
  d_ptr->c = malloc(sizeof(float) * N);
  if (d_ptr->c == NULL)
    die("cannot allocate memory for c", __LINE__, __FILE__);

  // Have to place all pointers into local variables
  // for OpenMP to accept them in mapping clauses
  float *a = d_ptr->a;
  float *b = d_ptr->b;
  float *c = d_ptr->c;

  // Set up data region on device
  #pragma omp target enter data map(alloc: a[0:N], b[0:N], c[0:N])
  {}
}

void finalise(data *d_ptr, const int N) {
  // Have to place all pointers into local variables
  // for OpenMP to accept them in mapping clauses
  float *a = d_ptr->a;
  float *b = d_ptr->b;
  float *c = d_ptr->c;

  // End data region on device
  #pragma omp target exit data map(release: a[0:N], b[0:N], c[0:N])
  {}

  free(d_ptr->a);
  d_ptr->a = NULL;
  free(d_ptr->b);
  d_ptr->b = NULL;
  free(d_ptr->c);
  d_ptr->c = NULL;
}

void die(const char *message, const int line, const char *file) {
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}
