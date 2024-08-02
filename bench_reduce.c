#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "float.h"
#include "time.h"
#include "math.h"
#include "mkl.h"
#include "immintrin.h"

#define NTRIALS            (100)
#define ALIGN              (4*1024) 
#define QUEUE_DEPTH        (4)
#define BLK_SIZE_IN_BYTES  (1024*1024)

#if !defined(USE_CPU) && !defined(USE_DSA)
#error USE_CPU of USE_DSA should be defined
#endif

void demote_buffers(long long *p_n, double *p_src)
{
  long long n = *p_n;
  for (long long i=0; i<n; i+=8) {
    //_mm_cldemote(p_src);
    _mm_clflushopt(p_src); 
    p_src += 8;
  }
}
 
double cpu_reduce(long long *p_n, double *p_src)
{
  double tmp = 0.;

  for (long long i=0; i<*p_n; i++) {
    tmp += p_src[i];
  }

  return tmp;
}

void cpu_copy(long long *p_n, double *p_src, double *p_dst)
{
  for (long long i=0; i<*p_n; i++) {
    p_dst[i] = p_src[i];
  }
}

#if defined(USE_DSA)
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

void sync_copy(long long *p_n, double *p_src, double *p_dst, dml_job_t *p_dml_job)
{
  dml_status_t status;

  p_dml_job->operation             = DML_OP_MEM_MOVE;
  p_dml_job->flags                 = DML_FLAG_COPY_ONLY|DML_FLAG_PREFETCH_CACHE;
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
  p_dml_job->flags                 = DML_FLAG_COPY_ONLY|DML_FLAG_PREFETCH_CACHE;
  p_dml_job->source_first_ptr      = (void *)p_src;
  p_dml_job->destination_first_ptr = (void *)p_dst;
  p_dml_job->source_length         = (*p_n)*sizeof(double);
  p_dml_job->destination_length    = (*p_n)*sizeof(double);

  status = dml_submit_job(p_dml_job);

  if (status) {
    printf ("\tdml_submit_job failed, status = %u\n", status);
  }
}
#endif

int main (int argc, char **argv)
{
  double *p_src = NULL, *p_tmp = NULL, *p_t_elapsed = NULL;
  double res, t_start, t_dsa_start, t_dsa_elap, t_cpu_start, t_cpu_elap;
  double size_mB, size_gB;
  long long num_bytes, num_blks, blk_elems, num_elems, tail;

  if (argc != 2) {
    printf("\nUSAGE: %s <size> in bytes\n", argv[0]);
    exit(1);
  }
  num_bytes = atol(argv[1]);

  if ((num_bytes%sizeof(double))) {
    printf ("ERROR: input size should be multiple of 8 (sizeof(double))\n");
    printf ("Exiting..");
    exit (1);
  }
  
  num_elems   = num_bytes/sizeof(double);
  blk_elems   = BLK_SIZE_IN_BYTES/sizeof(double);
  num_blks    = num_elems/blk_elems;
  tail        = num_elems - (blk_elems * num_blks);

  p_src       = (double *)_mm_malloc(num_bytes, ALIGN);
  p_t_elapsed = (double *)_mm_malloc(sizeof(double)*NTRIALS, ALIGN);
  p_tmp       = (double *)_mm_malloc(BLK_SIZE_IN_BYTES*QUEUE_DEPTH, ALIGN);

#if defined (USE_DSA)
  for (long long i=0; i<blk_elems*QUEUE_DEPTH; i++) {
    p_tmp[i] = 0.;
  }

  dml_job_t* p_dml_jobs[QUEUE_DEPTH];
  for (int i=0; i<QUEUE_DEPTH; i++) {
    p_dml_jobs[i] = init_dml();
  }
#endif

  srand48(42);
  for (long long i=0; i<num_elems; i++) {
    p_src[i] = drand48();
  }


  double ref_res = cpu_reduce(&num_elems, p_src);

  size_mB = num_bytes/1.e6;
  size_gB = size_mB/1.e3;

  for (int i=0; i<NTRIALS; i++) {
    //demote_buffers(&num_elems, p_src);

    res = 0.;
    t_cpu_elap = 0.;
    t_dsa_elap = 0.;
    t_start = dsecnd();

#if defined (USE_CPU)
#define V1
#ifdef V1
    res = cpu_reduce(&num_elems, p_src);
#else 
    for (int j=0; j<num_blks; j++) {
      t_cpu_start = dsecnd();
      res += cpu_reduce(&blk_elems, p_src+(j*blk_elems));
      t_cpu_elap += dsecnd() - t_cpu_start;
    }

    if (tail) {
      res += cpu_reduce(&tail, p_src+(num_blks*blk_elems));
    }

    printf ("Iter[%d]: N = %ld, Size(MB) = %.2f, CPU = %.2f us (%.2f GB/s)\n",
            i, num_elems, size_mB, t_cpu_elap*1.e6, (size_gB/t_cpu_elap));
#endif

#elif defined (USE_DSA)
    for (int i=0; i<QUEUE_DEPTH; i++) {
      async_copy(&blk_elems, p_src+((QUEUE_DEPTH+i)*blk_elems), p_tmp+(i*blk_elems), p_dml_jobs[i]);
    }
    long long tmp_elems = QUEUE_DEPTH * blk_elems;
    res += cpu_reduce(&tmp_elems, p_src);

    for (int j=QUEUE_DEPTH; j<(num_blks-QUEUE_DEPTH); j++) {
      /* t_dsa_start = dsecnd(); */
      dml_wait_job(p_dml_jobs[j%QUEUE_DEPTH]);
      /* t_dsa_elap += dsecnd() - t_dsa_start; */

      /* t_cpu_start = dsecnd(); */
      res += cpu_reduce(&blk_elems, p_tmp+((j%QUEUE_DEPTH)*blk_elems));
      /* t_cpu_elap += dsecnd() - t_cpu_start; */

      async_copy(&blk_elems, p_src+((j+QUEUE_DEPTH)*blk_elems), p_tmp+((j%QUEUE_DEPTH)*blk_elems), p_dml_jobs[j%QUEUE_DEPTH]);
    }

    for (int i=0; i<QUEUE_DEPTH; i++) {
      res += cpu_reduce(&blk_elems, p_tmp+(i*blk_elems));
    }

    if (tail) {
      res += cpu_reduce(&tail, p_src+(num_blks*blk_elems));
    }

    for (int i=0; i<QUEUE_DEPTH; i++) {
      dml_wait_job(p_dml_jobs[i]);
    }

#ifdef DEBUG
    printf ("Iter[%d]: N = %ld, Size(MB) = %.2f, DSA = %.2f us (%.2f GB/s), CPU = %.2f us (%.2f GB/s)\n",
            i, num_elems, size_mB, t_dsa_elap*1.e6, (size_gB/t_dsa_elap), t_cpu_elap*1.e6, (size_gB/t_cpu_elap));
#endif
#endif

    if (fabs(res - ref_res) > 1.e-04) {
      printf ("validation failed. Expected = %lf, Observed = %lf\n", 3.5*num_elems, res);
      goto bailout;
    }

    p_t_elapsed[i] = dsecnd() - t_start;
    printf ("Iter[%d]: N = %ld, Size(MB) = %.2f, Time (us) = %.2f, Bandwidth(GB/s) = %.4f\n",
            i, num_elems, size_mB, p_t_elapsed[i]*1.e6, (size_gB)/(p_t_elapsed[i]));
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

  printf ("N = %ld, Size(MB) = %.2f, Best Time(us) = %.2f, Best Bandwidth(GB/s) = %.4f\n", num_elems, size_mB, t_best*1.e6, (size_gB)/(t_best));

bailout:
  _mm_free(p_src);
  _mm_free(p_tmp);
  _mm_free(p_t_elapsed);

  return 0;
}
