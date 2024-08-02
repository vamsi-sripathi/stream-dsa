#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include "immintrin.h"
#include "float.h"
#include "time.h"

#define NUM_JOBS_IN_BATCH  (8)
#define NTRIALS      (100)
#define ALIGNMENT_4K (4*1024)
#define ALIGNMENT_2M (2*1024*1024)
#define ALIGN        ALIGNMENT_4K
#define LLC_SIZE     (107520*1024)
#define BUFFER_2GB   (2*1024*1024*1024ULL)

#if !defined(USE_CPU) && !defined(USE_DSA)
#error USE_CPU of USE_DSA should be defined
#endif

#if defined(USE_CPU)
void copy(long long *p_n, double *p_src, double *p_dst)
{
  for (long long i=0; i<*p_n; i++) {
    p_dst[i] = p_src[i];
  }
}
#elif defined(USE_DSA)
#include "dml/dml.h"

dml_job_t* init_dml()
{
  dml_job_t *p_dml_job = NULL;
  dml_status_t status;
  uint32_t size;

  status  = dml_get_job_size(DML_PATH_HW, &size);
  if (DML_STATUS_OK != status) {
    exit(1);
  }

  p_dml_job = (dml_job_t *) malloc(size);
  status  = dml_init_job(DML_PATH_HW, p_dml_job);
  if (DML_STATUS_OK != status) {
    exit(1);
  }

  return p_dml_job;
}

void copy(long long *p_n, double *p_src, double *p_dst, dml_job_t *p_dml_job)
{
  dml_status_t status;

  p_dml_job->operation             = DML_OP_MEM_MOVE;
  p_dml_job->flags                 = DML_FLAG_COPY_ONLY;
  p_dml_job->source_first_ptr      = (void *)p_src;
  p_dml_job->destination_first_ptr = (void *)p_dst;
  p_dml_job->source_length         = (*p_n)*sizeof(double);
  p_dml_job->destination_length    = (*p_n)*sizeof(double);

  status = dml_execute_job(p_dml_job);

  if (status) {
    printf ("\tdml_execute_job failed, status = %u\n", status);
  }
}

void async_copy(long long *p_n, double *p_src, double *p_dst, dml_job_t *p_dml_job)
{
  dml_status_t status;

  p_dml_job->operation             = DML_OP_MEM_MOVE;
  p_dml_job->flags                 = DML_FLAG_COPY_ONLY;
  p_dml_job->source_first_ptr      = (void *)p_src;
  p_dml_job->destination_first_ptr = (void *)p_dst;
  p_dml_job->source_length         = (*p_n)*sizeof(double);
  p_dml_job->destination_length    = (*p_n)*sizeof(double);

  status = dml_submit_job(p_dml_job);

  if (status) {
    printf ("\tdml_submit_job failed, status = %u\n", status);
  }
}


dml_job_t* batch_init_dml()
{
  dml_job_t *p_dml_job = NULL;
  dml_status_t status;
  uint32_t size;

  status  = dml_get_job_size(DML_PATH_HW, &size);
  if (DML_STATUS_OK != status) {
    exit(1);
  }

  p_dml_job = (dml_job_t *) malloc(size);
  status  = dml_init_job(DML_PATH_HW, p_dml_job);
  if (DML_STATUS_OK != status) {
    exit(1);
  }

  // start setup for batch
  uint32_t op_count = NUM_JOBS_IN_BATCH;
  uint32_t batch_buffer_length = 0;

  status = dml_get_batch_size(p_dml_job, op_count, &batch_buffer_length);
  if (status != DML_STATUS_OK) {
    printf ("\tdml_get_batch_size failed\n");
    return NULL;
  }
  
  uint8_t *p_batch_buffer = (uint8_t *) malloc(batch_buffer_length);

  p_dml_job->operation = DML_OP_BATCH;
  p_dml_job->destination_first_ptr = p_batch_buffer;
  p_dml_job->destination_length = batch_buffer_length;


  return p_dml_job;
}


void batch_copy(long long *p_n, double *p_src, double *p_dst, dml_job_t *p_dml_job)
{
  dml_status_t status;


  long long chunk = (*(p_n))/NUM_JOBS_IN_BATCH;

  for (int i=0; i<NUM_JOBS_IN_BATCH; i++) {
    status = dml_batch_set_mem_move_by_index(p_dml_job, i, (void *)((double *)p_src+(i*chunk)), (void *)((double *)p_dst+(i*chunk)), chunk*sizeof(double), DML_FLAG_COPY_ONLY);
    if (status != DML_STATUS_OK) {
      printf ("\tdml_batch_set_mem_move_by_index failed at %d\n", i);
      return;
    }
  }

  status = dml_execute_job(p_dml_job);

  if (status) {
    printf ("\tdml_execute_job failed in batch copy, status = %u\n", status);
  }
}
#endif

int check_results(long long *p_n, double *p_src, double *p_dst)
{
  for (long long i=0; i<*p_n; i++) {
    if (p_src[i] != p_dst[i]) {
      printf ("\tERROR: Results differ: p_src[%ld] = %lf, p_dst[%ld] = %lf\n", i, p_src[i], i, p_dst[i]); 
      return 1;
    }
  }
  return 0;
}


int main (int argc, char **argv)
{
  double *p_src = NULL, *p_dst = NULL, *p_t_elapsed = NULL;
  double t_start;

  if (argc != 4) {
    printf("\nUSAGE: %s <start> <end> <step> all in bytes\n", argv[0]);
    exit(1);
  }
  long long bytes_start, bytes_end, bytes_step;
  bytes_start = atol(argv[1]);
  bytes_end   = atol(argv[2]);
  bytes_step  = atol(argv[3]);

  if ((bytes_start%sizeof(double)) || (bytes_step%sizeof(double))) {
    printf ("ERROR: input size should be multiple of 8 (sizeof(double))\n");
    printf ("Exiting..");
    exit (1);
  }
  
  if ((bytes_start > BUFFER_2GB) || (bytes_end > BUFFER_2GB)) {
    printf ("ERROR: Maximum size of input should be less than %lld bytes (2GB)\n", BUFFER_2GB);
    exit(1);
  }

  long long size = BUFFER_2GB/sizeof(double);
  p_src       = (double *)_mm_malloc(BUFFER_2GB, ALIGN);
  p_dst       = (double *)_mm_malloc(BUFFER_2GB, ALIGN);
  p_t_elapsed = (double *)_mm_malloc(sizeof(double)*NTRIALS, ALIGN);

  srand48(42);
  for (long long i=0; i<size; i++) {
    p_src[i] = drand48();
    p_dst[i] = 0.;
  }

  srand48(time(NULL));

#if defined (USE_DSA)
#if defined (USE_SINGLE_COPY)
  dml_job_t* p_dml_job = init_dml();
#elif defined (USE_IMPLICIT_BATCH_COPY)
  dml_job_t* p_dml_job = batch_init_dml();
#elif defined (USE_EXPLICIT_BATCH_COPY)
  dml_job_t* p_dml_job[NUM_JOBS_IN_BATCH];
  for (int i=0; i<NUM_JOBS_IN_BATCH; i++) {
    p_dml_job[i] = init_dml();
  }
#endif
#endif

  for (long long k=bytes_start; k<=bytes_end; k+=bytes_step) {
    long long copy_size_bytes = k;
    long long num_elems = copy_size_bytes/sizeof(double);
    double copy_size_kB = copy_size_bytes/1000.;
    double copy_size_mB = copy_size_kB/1000.;
    double copy_size_gB = copy_size_mB/1000.;

#if defined (USE_IMPLICIT_BATCH_COPY) || defined (USE_EXPLICIT_BATCH_COPY)
    if (num_elems%NUM_JOBS_IN_BATCH != 0) {
      printf ("\tSize is not divisible by num jobs in a batch, skipping..\n");
      continue;
    }
    long long chunk = num_elems/NUM_JOBS_IN_BATCH;
#endif

    printf ("\tNumber of bytes to be copied = %lld (bytes), %.2f (KB), %.2f(MB), %.2f(GB)\n", copy_size_bytes, copy_size_kB, copy_size_mB, copy_size_gB);

    for (int i=0; i<NTRIALS; i++) {
      t_start = dsecnd();

#if defined (USE_CPU)
      copy(&num_elems, p_src, p_dst);
#elif defined (USE_DSA)
#if defined (USE_SINGLE_COPY)
      copy(&num_elems, p_src, p_dst, p_dml_job);
#elif defined (USE_IMPLICIT_BATCH_COPY)
      batch_copy(&num_elems, p_src, p_dst, p_dml_job);
#elif defined (USE_EXPLICIT_BATCH_COPY)
      for (int ii=0; ii<NUM_JOBS_IN_BATCH; ii++) {
        async_copy(&chunk, p_src+(ii*chunk), p_dst+(ii*chunk), p_dml_job[ii]);
      }
      for (int ii=0; ii<NUM_JOBS_IN_BATCH; ii++) {
        dml_wait_job(p_dml_job[ii]);
      }
#endif
#endif

      p_t_elapsed[i] = dsecnd() - t_start;

      if (check_results(&num_elems, p_src, p_dst)) {
        printf ("\tValidation failed!\n");
        goto bailout;
      } else {
        if (!i) {
          printf ("\tValidation passed\n");
        }
      }
    }

    double t_best = FLT_MAX;
    double t_total = 0.;
    for (int i=0; i<NTRIALS; i++) {
      if (t_best > p_t_elapsed[i]) {
        t_best = p_t_elapsed[i];
      }
      t_total += p_t_elapsed[i];
    }
    double t_avg = t_total/NTRIALS;

    printf ("N = %ld, Size(MB) = %.2f, Best Time(ns) = %.2f, Best Bandwidth(GB/s) = %.2f\n", num_elems, copy_size_mB, t_best*1.e9, (2.*copy_size_gB)/(t_best));
  }

bailout:
  _mm_free(p_src);
  _mm_free(p_dst);
  _mm_free(p_t_elapsed);

  return 0;
}
