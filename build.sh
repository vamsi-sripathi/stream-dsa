#!/bin/bash

set -x

DML_ROOT=/nfs/pdx/home/vsripath/tools/dml-v0.1.9-beta-internal

SRC=bench_copy.c
icx -DUSE_CPU                           -qmkl -O3 -xCORE-AVX512 -qopt-streaming-stores=always -fno-alias ${SRC} -o cpu_copy.bin 
icx -DUSE_DSA -DUSE_EXPLICIT_BATCH_COPY -qmkl -O3 -xCORE-AVX512 -fno-alias ${SRC} -I${DML_ROOT}/include/ -o dsa_copy_batch_explicit.bin ${DML_ROOT}/lib64/libdml.a -ldl 
# icx -DUSE_DSA -DUSE_SINGLE_COPY         -qmkl -O3 -xCORE-AVX512 -fno-alias ${SRC} -I${DML_ROOT}/include/ -o dsa_copy.bin ${DML_ROOT}/lib64/libdml.a -ldl 
# icx -DUSE_DSA -DUSE_IMPLICIT_BATCH_COPY -qmkl -O3 -xCORE-AVX512 -fno-alias ${SRC} -I${DML_ROOT}/include/ -o dsa_copy_batch_implicit.bin ${DML_ROOT}/lib64/libdml.a -ldl 

SRC=bench_scale.c
icx -DUSE_CPU -qmkl -O3 -xCORE-AVX512 -qopt-streaming-stores=always -fno-alias -qopenmp ${SRC} -o cpu_scale_nt.bin
icx -DUSE_DSA -qmkl -O3 -xCORE-AVX512 -qopt-streaming-stores=always -fno-alias -qopenmp ${SRC} -I${DML_ROOT}/include/ -o dsa_scale_nt.bin ${DML_ROOT}/lib64/libdml.a -ldl 

SRC=bench_triad.c
icx -DUSE_CPU -qmkl -O3 -xCORE-AVX512 -qopt-streaming-stores=always -fno-alias -qopenmp ${SRC} -o cpu_triad_nt.bin 
icx -DUSE_DSA -qmkl -O3 -xCORE-AVX512 -qopt-streaming-stores=always -fno-alias -qopenmp ${SRC} -I${DML_ROOT}/include/ -o dsa_triad_nt.bin ${DML_ROOT}/lib64/libdml.a -ldl 

SRC=bench_dot.c
icx -DUSE_CPU -qmkl -O3 -xCORE-AVX512 -fno-alias -qopenmp ${SRC} -o cpu_dot.bin -lnuma
icx -DUSE_DSA -qmkl -O3 -xCORE-AVX512 -fno-alias -qopenmp ${SRC} -I${DML_ROOT}/include/ -o dsa_dot.bin ${DML_ROOT}/lib64/libdml.a -ldl -lnuma


# SRC=bench_reduce.c
# icx -DUSE_CPU -qmkl -O3 -xCORE-AVX512 -fno-alias ${SRC} -o cpu_reduce.bin 
# icx -DUSE_DSA -qmkl -O3 -xCORE-AVX512 -fno-alias ${SRC} -I${DML_ROOT}/include/ -o dsa_reduce.bin ${DML_ROOT}/lib64/libdml.a -ldl 
# SRC=bench_reduce_omp.c
# icx -DUSE_CPU -qmkl -O3 -xCORE-AVX512 -fno-alias -qopenmp ${SRC} -o cpu_reduce_omp.bin
# icx -DUSE_DSA -qmkl -O3 -xCORE-AVX512 -fno-alias -qopenmp ${SRC} -I${DML_ROOT}/include/ -o dsa_reduce_omp.bin ${DML_ROOT}/lib64/libdml.a -ldl
