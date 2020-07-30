/** ExaTensor::TAL-SH: Lower-level header:
    Parameters, derived types, and function prototypes
    used at the lower level of TAL-SH (device specific):
    CP-TAL, NV-TAL, XP-TAL, AM-TAL, etc.
REVISION: 2020/07/30

Copyright (C) 2014-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2020 Oak Ridge National Laboratory (UT-Battelle)

This file is part of ExaTensor.

ExaTensor is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ExaTensor is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with ExaTensor. If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------

PREPROCESSOR OPTIONS:
 # -D CUDA_ARCH=350: target Nvidia GPU compute capability (default is 130);
 # -D NO_GPU: disables Nvidia GPU usage (CPU structures only);
 # -D NO_BLAS: cuBLAS calls will be replaced by in-house routines (slower);
 # -D DEBUG_GPU: collection of debugging information will be activated;
NOTES:
 # GPU_ID is a unique CUDA GPU ID given to a specific NVidia GPU present on the Host node:
    0<=GPU_ID<MAX_GPUS_PER_NODE; GPU_ID=-1 sometimes will refer to the (multi-)CPU Host.
    All MPI processes running on the same node will have a flat numeration of all present GPUs,
    each process either having its own subrange of GPUs or sharing all of them with others.
 # MIC_ID, AMD_ID, etc. are defined completely analogously to GPU_ID.
   In general, ACC_ID (Accelerator ID) is its ID within its class (0,1,2,...).
 # A flat DEVICE_ID can refer either to the (multi-)CPU Host (=0), OR
    to a specific NVidia GPU: gpu_id=abs(DEVICE_ID)-1, OR
    to a specific Intel Xeon Phi: mic_id=abs(DEVICE_ID)-1-MAX_GPUS_PER_NODE, OR
    to a specific AMD GPU: amd_id=abs(DEVICE_ID)-1-MAX_GPUS_PER_NODE-MAX_MICS_PER_NODE, etc.
    Device numeration:
     Null: {-1};
     Host: {0};
     NVidia GPU: {1..MAX_GPUS_PER_NODE};
     Intel Xeon Phi: {MAX_GPUS_PER_NODE+1:MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE};
     AMD GPU: {MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+1:MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+MAX_AMDS_PER_NODE}, etc.
 # MAX_GPU_ARGS limits the maximal allowed number of argument-buffer entries on each GPU.
   It determines the amount of static constant memory allocated on each GPU. Since each
   tensor contraction can involve up to six tensor blocks (because of dimension permutations),
   the maximal number of simultaneously active tensor contractions on each GPU is limited to MAX_GPU_ARGS/6.
 # CUDA_TASK is considered completed successfully if the value of the .task_error field equals zero.
   Negative .task_error means that either the CUDA task is empty (.gpu_id<0) or it is in progress (gpu_id>=0).
   Positive .task_error means that an error occured during task scheduling/execution.
FOR DEVELOPERS ONLY:
 # CPU/GPU resource allocation API functions (memory, mutli-indices, streams, events, etc.) may
   return a status TRY_LATER or DEVICE_UNABLE, both are not errors. If this happens within a scheduling
   function (asynchronous tensor operation), all relevant objects, which have already been created,
   will be returned to their initial state (the state before the scheduling function call).
 # If for some reason a device resource is not released properly but the object destruction still
   has happened, a non-critical error NOT_CLEAN may be returned.
**/

#ifndef DEVICE_ALGEBRA_H_
#define DEVICE_ALGEBRA_H_

#include "tensor_algebra.h"

#include <ctime>

#ifndef NO_GPU

#include <hip/hip_runtime.h>

#include <hipblas.h>

#ifdef USE_CUTT
#include "cutt.hip.h"
#endif

#endif /*NO_GPU*/

//DEVICE COMPUTE CAPABILITY (for Host code, but use __CUDA_ARCH__ for device code):
#ifndef CUDA_ARCH
#define CUDA_ARCH 130 //minimal required compute capability
#endif

//GLOBAL PARAMETERS:
#define MAX_GPU_ARGS 128 //max allowed number of tensor arguments simultaneously residing on a GPU: Must be a multiple of 8
#define MAX_CUDA_TASKS 128 //max allowed number of simultaneously active CUDA tasks per CUDA device
#ifdef GPU_FINE_TIMING
#define NUM_EVENTS_PER_TASK 6 //number of CUDA events recorded per CUDA task
#else
#define NUM_EVENTS_PER_TASK 4 //number of CUDA events recorded per CUDA task
#endif
#define MAX_CUDA_EVENTS (MAX_CUDA_TASKS*NUM_EVENTS_PER_TASK) //max number of CUDA events per CUDA device

//KERNEL PARAMETERS FOR NVIDIA GPU:
#if CUDA_ARCH >= 300
#define UNIFIED_ADDRESSING 1       //non-zero value will assume the availability of unified virtual addressing on GPU(s)
#else
#undef UNIFIED_ADDRESSING
#endif
#define GPU_CACHE_LINE_LEN 128     //cache line length in bytes
#define GPU_SHMEM_WIDTH 8          //default width of the GPU shared memory banks (4 or 8 bytes)
#define WMMA_ALIGN 8               //matrix dimension size alignment requirement for fast math on tensor cores
#define MAX_CUDA_BLOCKS 1024       //max number of CUDA thread blocks per kernel
#if CUDA_ARCH >= 300
#define TENS_TRANSP_BUF_SIZE 2048  //buffer size (elements) for <gpu_tensor_block_copy_dlf__>
#else
#define TENS_TRANSP_BUF_SIZE 1536  //buffer size (elements) for <gpu_tensor_block_copy_dlf__>
#endif
#define TENS_TRANSP_TAB_SIZE 69    //look up table size (integers) for <gpu_tensor_block_copy_dlf__>
#define MAT_MULT_TILE_DIMY 16      //Y tile dimension size for <gpu_matrix_multiply_tn__>
#if CUDA_ARCH >= 200
#define MAT_MULT_TILE_DIMX 32      //X tile dimension size for <gpu_matrix_multiply_tn__>: Must be multiple of MAT_MULT_TILE_DIMY
#else
#define MAT_MULT_TILE_DIMX 16      //X tile dimension size for <gpu_matrix_multiply_tn__>: Must be multiple of MAT_MULT_TILE_DIMY
#endif
#define THRDS_ARRAY_PRODUCT 256    //threads per block for <gpu_array_product__>
#define THRDS_ARRAY_NORM2 256      //threads per block for <gpu_array_norm2__>
#define THRDS_ARRAY_INIT 256       //threads per block for <gpu_array_init__>
#define THRDS_ARRAY_SCALE 256      //threads per block for <gpu_array_scale__> and <gpu_array_dot_product__>
#define THRDS_ARRAY_ADD 256        //threads per block for <gpu_array_add__>
#if CUDA_ARCH >= 200
#define THRDS_TENSOR_COPY 256      //threads per block for <gpu_tensor_block_copy_dlf__>
#else
#define THRDS_TENSOR_COPY 192      //threads per block for <gpu_tensor_block_copy_dlf__>
#endif
#define THRDS_TENSOR_COPY_SCAT 256 //threads per block for <gpu_tensor_block_copy_scatter_dlf__>

//CUDA TASK STATUS (keep consistent with tensor_algebra.F90):
#define CUDA_TASK_ERROR -1
#define CUDA_TASK_EMPTY 0
#define CUDA_TASK_SCHEDULED 1
#define CUDA_TASK_STARTED 2
#define CUDA_TASK_INPUT_THERE 3
#define CUDA_TASK_OUTPUT_THERE 4
#define CUDA_TASK_COMPLETED 5

//ALIASES (keep consistent with tensor_algebra.F90):
#define NVTAL_SUCCESS 0
#define NVTAL_FAILURE -666
#define NVTAL_DEFERRED 918273645

//DERIVED TYPES (keep consistent with tensor_algebra.F90):
// Tensor block (for the use on NVidia GPU):
typedef struct{
 int data_kind;             //tensor data kind: float (R4), double (R8), complex float (C4), complex double (C8)
 talsh_tens_shape_t shape;  //tensor shape: pinned Host memory (pointer components use the multi-index slab, miBank)
 talsh_dev_rsc_t * src_rsc; //source of the data (memory resource where the data resides before the task)
 talsh_dev_rsc_t * dst_rsc; //destination of the data (memory resource where the data will reside after the task)
 talsh_dev_rsc_t * tmp_rsc; //temporary memory resource (for tensor transposes)
} tensBlck_t;

// Tensor argument (for the use on Nvidia GPU):
typedef struct{
 tensBlck_t * tens_p; //pointer to a tensor block
 int * prmn_p;        //tensor block dimension permutation: pinnned HOST memory (miBank)
 int const_mem_entry; //NVidia GPU constant memory entry handle (>=0, -1:None)
} cudaTensArg_t;

// CUDA task (returned by non-blocking CUDA functions):
typedef struct{
 int task_error;         //error code (<0: Task is either empty or in progress; 0: Success; >0: Error code)
 int gpu_id;             //NVidia GPU ID the task was scheduled on (>=0, -1 means the task is empty)
 int stream_hl;          //CUDA stream handle the task went into
 int event_start_hl;     //handle of the CUDA event recorded at the beginning of the task
 int event_comput_hl;    //handle of the CUDA event recorded before the CUDA kernels start (all input data is on Device)
 int event_output_hl;    //handle of the CUDA event recorded when the CUDA kernels finish (before output to the Host)
 int event_finish_hl;    //handle of the CUDA event recorded at the end of the task (full completion)
#ifdef GPU_FINE_TIMING
 int event_mmbeg_hl;     //handle of the CUDA event recorded before the matrix multiplication starts
 int event_mmend_hl;     //handle of the CUDA event recorded after the matrix multiplication finishes
#endif
 unsigned int coherence; //coherence control for this task (see COPY_X, COPY_XX, and COPY_XXX constants)
 unsigned int num_args;  //number of tensor arguments participating in the tensor operation
 cudaTensArg_t tens_args[MAX_TENSOR_OPERANDS]; //tensor arguments participating in the tensor operation
#ifdef USE_CUTENSOR
 cutensorTensorDescriptor_t tens_cudesc[MAX_TENSOR_OPERANDS]; //tensor descriptors for cuTensor
#endif
 void * pref_ptr;        //tensor operation prefactor location
} cudaTask_t;
//Note: Adding new CUDA events will require adjustment of NUM_EVENTS_PER_TASK.

//FUNCTION PROTOTYPES:
extern "C"{
#ifndef NO_GPU
// NVidia GPU operations (NV-TAL):
//  NV-TAL debugging:
 int gpu_get_error_count();
 int gpu_get_debug_dump(int *dump);
//  NV-TAL initialization/shutdown (for internal use only):
 int init_gpus(int gpu_beg, int gpu_end);
 int free_gpus(int gpu_beg, int gpu_end);
//  NV-TAL query/action API:
 int gpu_get_device_count(int * dev_count);
 int gpu_is_mine(int gpu_num);
 int gpu_busy_least();
 int gpu_in_focus(int gpu_num = -1);
 int gpu_activate(int gpu_num);
 size_t gpu_device_memory_size(int gpu_num);
 double gpu_get_flops(int gpu_num = -1);
//  NV-TAL internal control:
 int gpu_set_shmem_width(int width);
 int gpu_enable_fast_math(int gpu_num = -1);
 int gpu_disable_fast_math(int gpu_num = -1);
 int gpu_query_fast_math(int gpu_num);
 void gpu_set_transpose_algorithm(int alg); //{EFF_TRN_OFF,EFF_TRN_ON,EFF_TRN_ON_CUTT}
 void gpu_set_matmult_algorithm(int alg);
 int gpu_print_stats(int gpu_num = -1);
#endif /*NO_GPU*/
//  C tensor block API:
//   NV-TAL tensor block:
 int tensBlck_create(tensBlck_t **ctens);
 int tensBlck_clean(tensBlck_t *ctens);
 int tensBlck_destroy(tensBlck_t *ctens);
 int tensBlck_construct(tensBlck_t *ctens, int pinned,
                        int trank, const int *dims = NULL, const int *divs = NULL, const int *grps = NULL);
 int tensBlck_attach_body(tensBlck_t *ctens, int data_kind, int dev_id = -1, void *body_ptr = NULL, int buf_entry = -1);
 int tensBlck_destruct(tensBlck_t *ctens, int release_body = YEP, int which_body = EVERYTHING);
 int tensBlck_src_dev_id(const tensBlck_t * ctens, int * dev_kind = NULL);
 int tensBlck_present(const tensBlck_t * ctens, int dev_id = DEV_NULL, int dev_kind = DEV_NULL);
 size_t tensBlck_volume(const tensBlck_t * ctens);
 void tensBlck_print(const tensBlck_t * ctens);
 int tensBlck_init_host(tensBlck_t * ctens, double init_val);
 double tensBlck_norm2_host(const tensBlck_t * ctens);
#ifndef NO_GPU
//  NV-TAL CUDA task API:
 int cuda_task_create(cudaTask_t **cuda_task);
 int cuda_task_clean(cudaTask_t *cuda_task);
 int cuda_task_construct(cudaTask_t *cuda_task, int gpu_id = -1);
 int cuda_task_destruct(cudaTask_t *cuda_task);
 int cuda_task_destroy(cudaTask_t *cuda_task);
 int cuda_task_gpu_id(const cudaTask_t *cuda_task);
 int cuda_task_status(cudaTask_t *cuda_task);
 int cuda_task_completed(cudaTask_t *cuda_task);
 int cuda_task_wait(cudaTask_t *cuda_task);
 int cuda_tasks_wait(unsigned int num_tasks, cudaTask_t **cuda_tasks, int *task_stats);
 int cuda_task_error_code(const cudaTask_t *cuda_task);
 int cuda_task_dev_rsc_copy(const cudaTask_t *cuda_task, unsigned int arg_num, char which, talsh_dev_rsc_t *dev_rsc);
 int cuda_task_dev_rsc_move(cudaTask_t *cuda_task, unsigned int arg_num, char which, talsh_dev_rsc_t *dev_rsc);
 int cuda_task_arg_has_resource(cudaTask_t *cuda_task, unsigned int arg_num, char which, int *ierr);
 int cuda_task_arg_destroy(cudaTask_t *cuda_task, int arg_num = -1);
 float cuda_task_time(const cudaTask_t *cuda_task, float *in_copy = NULL, float *out_copy = NULL, float *comp = NULL,
                      float *mmul = NULL);
 float cuda_task_time_(const cudaTask_t *cuda_task, float *in_copy, float *out_copy, float *comp, float *mmul);
 void cuda_task_print(const cudaTask_t *cuda_task);
//  NV-TAL tensor operations:
 int gpu_tensor_block_place(tensBlck_t *ctens, int gpu_id, unsigned int coh_ctrl, cudaTask_t *cuda_task, void *dev_mem = NULL);
 int gpu_tensor_block_init(tensBlck_t *dtens, double val_real, double val_imag, unsigned int coh_ctrl, cudaTask_t *cuda_task, int gpu_id = -1);
 int gpu_tensor_block_slice(tensBlck_t *ltens, tensBlck_t *dtens, const int * offsets,
                            unsigned int coh_ctrl, cudaTask_t *cuda_task, int gpu_id = -1, int accumulative = NOPE);
 int gpu_tensor_block_insert(tensBlck_t *ltens, tensBlck_t *dtens, const int * offsets,
                             unsigned int coh_ctrl, cudaTask_t *cuda_task, int gpu_id = -1, int accumulative = NOPE);
 int gpu_tensor_block_copy(const int *cptrn, tensBlck_t *ltens, tensBlck_t *dtens, unsigned int coh_ctrl, cudaTask_t *cuda_task,
                           int gpu_id = -1, int conj_bits = 0);
 int gpu_tensor_block_add(const int *cptrn, tensBlck_t *ltens, tensBlck_t *dtens, unsigned int coh_ctrl, cudaTask_t *cuda_task,
                          int gpu_id = -1, double scale_real = 1.0, double scale_imag = 0.0, int conj_bits = 0);
 int gpu_tensor_block_contract_dlf(const int *cptrn, tensBlck_t *ltens, tensBlck_t *rtens, tensBlck_t *dtens,
                                   unsigned int coh_ctrl, cudaTask_t *cuda_task, int gpu_id = -1,
                                   double scale_real = 1.0, double scale_imag = 0.0, int conj_bits = 0, int accumulative = YEP);
 int gpu_tensor_block_decompose_svd(const char absorb, tensBlck_t *dtens, tensBlck_t *ltens, tensBlck_t *rtens, tensBlck_t *stens,
                                    int gpu_id = -1);
#endif /*NO_GPU*/
}

template <typename T> int gpu_matrix_multiply_tn(size_t ll, size_t lr, size_t lc, const T * lmat, const T * rmat, T * dmat);

#endif /*DEVICE_ALGEBRA_H_*/
