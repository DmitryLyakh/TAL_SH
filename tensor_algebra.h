/** ExaTensor::TAL-SH: Lower-level header:
    Parameters, derived types, and function prototypes
    used at the lower level of TAL-SH (device specific):
    CP-TAL, NV-TAL, XP-TAL, AM-TAL, etc.
REVISION: 2015/09/28
Copyright (C) 2015 Dmitry I. Lyakh (email: quant4me@gmail.com)
Copyright (C) 2015 Oak Ridge National Laboratory (UT-Battelle)

This source file is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
-------------------------------------------------------------------------------
PREPROCESSOR OPTIONS:
 # -D CUDA_ARCH=350: target device compute capability (default is 130);
 # -D NO_GPU: disables GPU usage (CPU structures only);
 # -D NO_BLAS: cuBLAS calls will be replaced by in-house routines;
 # -D DEBUG_GPU: collection of debugging information will be activated;
NOTES:
 # GPU_ID is a unique CUDA GPU ID given to a specific NVidia GPU present on the Host node:
    0<=GPU_ID<MAX_GPUS_PER_NODE; GPU_ID=-1 will refer to the (multi-)CPU Host.
    All MPI processes running on the same node will have a flat numeration of all present GPUs,
    each process either having its own subrange of GPUs or sharing all of them with others.
 # MIC_ID, AMD_ID, etc. are defined completely analogously to GPU_ID.
   In general, ACC_ID (Accelerator ID) is its ID within its class (0,1,2,...).
 # DEVICE_ID can refer either to the (multi-)CPU Host (=0), OR
    to a specific NVidia GPU: gpu_id=abs(DEVICE_ID)-1, OR
    to a specific Intel Xeon Phi: mic_id=abs(DEVICE_ID)-1-MAX_GPUS_PER_NODE, OR
    to a specific AMD GPU: amd_id=abs(DEVICE_ID)-1-MAX_GPUS_PER_NODE-MAX_MICS_PER_NODE, etc.
    Device numeration:
     Null: {-1};
     Host: {0};
     NVidia GPU: {1..MAX_GPUS_PER_NODE};
     Intel Xeon Phi: {MAX_GPUS_PER_NODE+1:MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE};
     AMD GPU: {MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+1:MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+MAX_AMDS_PER_NODE}, etc.
    DEVICE_ID is used in tensBlck_t: If tensor elements are already on the Device it is positive, otherwise negative.
 # MAX_SCR_ENTRY_COUNT regulates the maximal amount of additional device argument-buffer entries
   allocated per tensor operation (it is 3 because of tensor contractions).
 # MAX_GPU_ARGS limits the maximal allowed number of argument-buffer entries on a GPU.
   It determines the amount of static constant memory allocated on each GPU.
 # tensBlck_t is a hardware-specific specification of a tensor-block argument used with NVidia GPU.
   In tensBlck_t, the tensor shape and tensor body must always point to the pinned Host memory
   allocated either in the HAB or explicitly via <host_mem_alloc_pin>, unless the Host copy of the
   tensor block had been released (freed).
 # CUDA_TASK is considered completed successfully if the value of the .task_error field equals zero.
   Negative .task_error means that either the CUDA task is empty or it is in progress. In the former case, .gpu_id=-1.
   Positive .task_error means that an error occured during the task scheduling/execution process.
FOR DEVELOPERS ONLY:
 # CPU/GPU resource allocation API functions (memory, streams, events, etc.) may return a status
   TRY_LATER or DEVICE_UNABLE, both are not errors. If this happens within a scheduling function
   (asynchronous tensor operation), all relevant objects, which have already been allocated, must
   be cleared and returned in their initial state (the state before the scheduling function call).
**/
//BEGINNING OF TENSOR_ALGEBRA_H
#ifndef TENSOR_ALGEBRA_H
#define TENSOR_ALGEBRA_H

#include <time.h>

//DEVICE COMPUTE CAPABILITY:
#ifndef CUDA_ARCH
#define CUDA_ARCH 130
#endif

//GLOBAL PARAMETERS:
#define MAX_TENSOR_RANK 32         //max allowed tensor rank: Must be multiple of 4
#define MAX_GPU_ARGS 128           //max allowed number of tensor arguments simultaneously residing on a GPU: Must be a multiple of 8
#define MAX_SCR_ENTRY_COUNT 3      //max allowed number of additional GPU argument entries allocated per tensor operation
#define MAX_CUDA_TASKS 128         //max allowed number of simultaneously active CUDA tasks per CUDA device
#define NUM_EVENTS_PER_TASK 4      //number of CUDA events recorded per CUDA task
#define MAX_CUDA_EVENTS MAX_CUDA_TASKS*NUM_EVENTS_PER_TASK //max number of CUDA events per CUDA device

//DEVICE KINDS:
#define MAX_GPUS_PER_NODE 8        //max allowed number of NVidia GPUs on a node
#define MAX_MICS_PER_NODE 8        //max allowed number of Intel MICs on a node
#define MAX_AMDS_PER_NODE 8        //max allowed number of AMD GPUs on a node
#define DEV_NULL -1                //abstract null device
#define DEV_HOST 0                 //multicore CPU Host (includes any self-hosted system)
#define DEV_NVIDIA_GPU 1           //NVidia GPU (as an accelerator)
#define DEV_INTEL_MIC 2            //Intel Xeon Phi (as an accelerator)
#define DEV_AMD_GPU 3              //AMD GPU (as an accelerator)
#define DEV_MAX 1+MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+MAX_AMDS_PER_NODE

//KERNEL PARAMETERS for NVidia GPU:
#define GPU_CACHE_LINE_LEN 128     //cache line length in bytes
#define GPU_SHMEM_WIDTH 8          //default width of the GPU shared memory banks (4 or 8 bytes)
#define MAX_CUDA_BLOCKS 1024       //max number of CUDA thread blocks per kernel
#if CUDA_ARCH >= 300
#define TENS_TRANSP_BUF_SIZE 2560  //buffer size (elements) for <gpu_tensor_block_copy_dlf_XX__>
#else
#define TENS_TRANSP_BUF_SIZE 1536  //buffer size (elements) for <gpu_tensor_block_copy_dlf_XX__>
#endif
#define TENS_TRANSP_TAB_SIZE 69    //look up table size (integers) for <gpu_tensor_block_copy_dlf_XX__>
#define MAT_MULT_TILE_DIMY 16      //Y tile dimension size for <gpu_matrix_multiply_tn_XX__>
#if CUDA_ARCH >= 200
#define MAT_MULT_TILE_DIMX 32      //X tile dimension size for <gpu_matrix_multiply_tn_XX__>: Must be multiple of MAT_MULT_TILE_DIMY
#else
#define MAT_MULT_TILE_DIMX 16      //X tile dimension size for <gpu_matrix_multiply_tn_XX__>: Must be multiple of MAT_MULT_TILE_DIMY
#endif
#define THRDS_ARRAY_PRODUCT 256    //threads per block for <gpu_array_product_XX__>
#define THRDS_ARRAY_NORM2 256      //threads per block for <gpu_array_2norm2_XX__>
#define THRDS_ARRAY_INIT 256       //threads per block for <gpu_array_init_XX__>
#define THRDS_ARRAY_SCALE 256      //threads per block for <gpu_array_scale_XX__> and <gpu_array_dot_product_XX__>
#define THRDS_ARRAY_ADD 256        //threads per block for <gpu_array_add_XX__>
#if CUDA_ARCH >= 200
#define THRDS_TENSOR_COPY 256      //threads per block for <gpu_tensor_block_copy_dlf_XX__>
#else
#define THRDS_TENSOR_COPY 192      //threads per block for <gpu_tensor_block_copy_dlf_XX__>
#endif
#define THRDS_TENSOR_COPY_SCAT 256 //threads per block for <gpu_tensor_block_copy_scatter_dlf_XX__>

//DATA KINDS (keep consistent with tensor_algebra.F90):
#define NO_TYPE 0 //null type
#define R4 4      //float data kind (keep consistent with c_process.f90::tens_blck_pack/unpack)
#define R8 8      //double data kind (keep consistent with c_process.f90::tens_blck_pack/unpack)
#define C8 16     //double complex data kind (keep consistent with c_process.f90::tens_blck_pack/unpack)

//CUDA TASK STATUS (keep consistent with tensor_algebra.F90):
#define CUDA_TASK_ERROR -1
#define CUDA_TASK_EMPTY 0
#define CUDA_TASK_SCHEDULED 1
#define CUDA_TASK_STARTED 2
#define CUDA_TASK_INPUT_THERE 3
#define CUDA_TASK_OUTPUT_THERE 4
#define CUDA_TASK_COMPLETED 5

//ALIASES (keep consistent with tensor_algebra.F90):
#define TALSH_SUCCESS 0
#define TALSH_FAILURE -666
#define EVENTS_OFF 0
#define EVENTS_ON 1
#define BLAS_ON 0
#define BLAS_OFF 1
#define EFF_TRN_OFF 0
#define EFF_TRN_ON 1
#define TRY_LATER -918273645
#define DEVICE_UNABLE -546372819
#define NOPE 0
#define YEP 1
#define DEV_OFF 0
#define DEV_ON 1
#define DEV_ON_BLAS 2
#define GPU_OFF 0
#define GPU_MINE 1
#define GPU_MINE_CUBLAS 2
#define NO_COPY_BACK 0
#define COPY_BACK 1
/*
#define COPY_F 0
#define COPY_K 1
#define COPY_FF 0
#define COPY_FK 1
#define COPY_KF 2
#define COPY_KK 3
#define COPY_FFF 0
#define COPY_FFK 1
#define COPY_FKF 2
#define COPY_FKK 3
#define COPY_KFF 4
#define COPY_KFK 5
#define COPY_KKF 6
#define COPY_KKK 7
*/
#define COPY_D 0
#define COPY_M 1
#define COPY_T 2
#define COPY_K 3
#define COPY_DD 0
#define COPY_DM 1
#define COPY_DT 2
#define COPY_DK 3
#define COPY_MD 4
#define COPY_MM 5
#define COPY_MT 6
#define COPY_MK 7
#define COPY_TD 8
#define COPY_TM 9
#define COPY_TT 10
#define COPY_TK 11
#define COPY_KD 12
#define COPY_KM 13
#define COPY_KT 14
#define COPY_KK 15
#define COPY_DDD 0
#define COPY_DDM 1
#define COPY_DDT 2
#define COPY_DDK 3
#define COPY_DMD 4
#define COPY_DMM 5
#define COPY_DMT 6
#define COPY_DMK 7
#define COPY_DTD 8
#define COPY_DTM 9
#define COPY_DTT 10
#define COPY_DTK 11
#define COPY_DKD 12
#define COPY_DKM 13
#define COPY_DKT 14
#define COPY_DKK 15
#define COPY_MDD 16
#define COPY_MDM 17
#define COPY_MDT 18
#define COPY_MDK 19
#define COPY_MMD 20
#define COPY_MMM 21
#define COPY_MMT 22
#define COPY_MMK 23
#define COPY_MTD 24
#define COPY_MTM 25
#define COPY_MTT 26
#define COPY_MTK 27
#define COPY_MKD 28
#define COPY_MKM 29
#define COPY_MKT 30
#define COPY_MKK 31
#define COPY_TDD 32
#define COPY_TDM 33
#define COPY_TDT 34
#define COPY_TDK 35
#define COPY_TMD 36
#define COPY_TMM 37
#define COPY_TMT 38
#define COPY_TMK 39
#define COPY_TTD 40
#define COPY_TTM 41
#define COPY_TTT 42
#define COPY_TTK 43
#define COPY_TKD 44
#define COPY_TKM 45
#define COPY_TKT 46
#define COPY_TKK 47
#define COPY_KDD 48
#define COPY_KDM 49
#define COPY_KDT 50
#define COPY_KDK 51
#define COPY_KMD 52
#define COPY_KMM 53
#define COPY_KMT 54
#define COPY_KMK 55
#define COPY_KTD 56
#define COPY_KTM 57
#define COPY_KTT 58
#define COPY_KTK 59
#define COPY_KKD 60
#define COPY_KKM 61
#define COPY_KKT 62
#define COPY_KKK 63

//MACRO FUNCTIONS:
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

//DERIVED TYPES (keep consistent with tensor_algebra.F90):
// Tensor shape:
typedef struct{
 int num_dim;   //tensor rank (number of dimensions)
 int * dims;    //tensor dimension extents
 int * divs;    //tensor dimension dividers
 int * grps;    //tensor dimension groups
} talsh_tens_shape_t;
// Device resource (occupied by a tensor block):
typedef struct{
 void * gmem_p;       //pointer to Host/Device global memory where the tensor body resides
 int buf_entry;       //argument buffer entry number (in Host/Device global memory)
 int const_mem_entry; //NVidia GPU constant memory entry number
} talsh_dev_rsc_t;
//Note: Not all fields above are defined on each device (some are device specific).
// Tensor block (for the use on NVidia GPU):
typedef struct{
 int device_id;        //device on which the tensor block already resides (+) or will reside (-) (device_id=0 means Host)
 int data_kind;        //tensor element size in bytes: float (4), double (8), or double complex (16)
 int rank;             //tensor block rank (>=0)
 int *dims_h;          //tensor block dimension extents (0th index is the most minor one): HOST memory
 int *divs_h;          //tensor block dimension dividers: HOST memory
 int *grps_h;          //tensor block dimension groups: HOST memory
 int *prmn_h;          //tensor block dimension permutation (not to be set by user!): HOST memory
 void *elems_h;        //tensor block elements (dlf): HOST memory (only one element for scalars)
 void *elems_d;        //tensor block elements (dlf): DEVICE global memory (only one element for scalars)
 int buf_entry_host;   //Host argument buffer entry pointed to by *elems_h: Host pinned memory
 int buf_entry_gpu;    //GPU argument buffer entry pointed to by *elems_d: GPU global memory
 int const_args_entry; //entry number in const_args[]: GPU constant memory (dims[] and prmn[] arrays are stored there)
} tensBlck_t;
// Interoperable tensor block:
typedef struct{
 int ndev;                  //number of devices the tensor block resides on
 int last_write;            //flat device id where the last write happened, -1 means coherence on all devices where the tensor block resides
 int * dev_list;            //list of the flat device id's which the tensor block resides on
 talsh_dev_rsc_t * dev_rsc; //list of the device resources occupied by the tensor block on each device
 void * tensF;              //pointer to Fortran <tensor_block_t>
 void * tensC;              //pointer to C <tensBlck_t>
} talsh_tens_t;
// Interoperable TAL-SH task handle:
typedef struct{
 int dev_kind;  //device kind
 void * task_p; //pointer to the corresponding task object
} talsh_task_t;

// Interface for a user-defined tensor block initialization routine:
typedef void (*talsh_tens_init_i)(void * tens_ptr, int data_type, int tens_rank, int tens_dims[], int * ierr);

#ifndef NO_GPU
// CUDA task (returned by non-blocking CUDA functions):
// Adding new CUDA events will require adjustment of NUM_EVENTS_PER_TASK.
typedef struct{
 int task_error;                     //error code (<0: Task is either empty or in progress; 0: Success; >0: Error code)
 int gpu_id;                         //NVidia GPU ID on which the task was scheduled (-1 means CPU Host)
 cudaStream_t task_stream;           //CUDA stream the task went into
 cudaEvent_t task_start;             //CUDA event recorded at the beginning of the task
 cudaEvent_t task_comput;            //CUDA event recorded before the computing kernels start (all input data is on Device)
 cudaEvent_t task_output;            //CUDA event recorded when the computing kernels finish (before output to the Host)
 cudaEvent_t task_finish;            //CUDA event recorded at the end of the task (full completion)
 int scr_entry_count;                //number of additional GPU argument-buffer entries allocated by the task
 int scr_entry[MAX_SCR_ENTRY_COUNT]; //additional GPU argument-buffer entries allocated by the task
} cudaTask_t;
#endif

// Device statistics:
typedef struct{
 unsigned long long int tasks_submitted; //number of TAL-SH tasks submitted to the device
 unsigned long long int tasks_completed; //number of TAL-SH tasks completed by the device
 unsigned long long int tasks_deferred;  //number of TAL-SH tasks deferred for later (TRY_LATER)
 unsigned long long int tasks_failed;    //number of TAL-SH tasks failed (except TRY_LATER)
 double flops;                           //total number of Flops processed (successfully completed)
 double traffic_in;                      //total number of bytes transferred in
 double traffic_out;                     //total number of bytes transferred out
 double time_active;                     //time in seconds device is active
 clock_t time_start;                     //time when the library was initialized (internal use only)
} talsh_stats_t;

//FUNCTION PROTOTYPES:
#ifdef __cplusplus
extern "C"{
#endif
// Buffer memory management (all devices):
 int arg_buf_allocate(size_t *arg_buf_size, int *arg_max, int gpu_beg, int gpu_end); //generic
 int arg_buf_deallocate(int gpu_beg, int gpu_end); //generic
 int arg_buf_clean_host(); //Host only
 int arg_buf_clean_gpu(int gpu_num); //NVidia GPU only
 int get_blck_buf_sizes_host(size_t *blck_sizes); //Host only
 int get_blck_buf_sizes_gpu(int gpu_num, size_t *blck_sizes); //NVidia GPU only
 int get_buf_entry_host(size_t bsize, char **entry_ptr, int *entry_num); //Host only
 int free_buf_entry_host(int entry_num); //Host only
 int get_buf_entry_gpu(int gpu_num, size_t bsize, char **entry_ptr, int *entry_num); //NVidia GPU only
 int free_buf_entry_gpu(int gpu_num, int entry_num); //NVidia GPU only
 int const_args_entry_get(int gpu_num, int *entry_num); //NVidia GPU only
 int const_args_entry_free(int gpu_num, int entry_num); //NVidia GPU only
 int mem_free_left(int dev_id, size_t * free_mem); //generic
 int mem_print_stats(int dev_id); //generic
#ifndef NO_GPU
 int host_mem_alloc_pin(void **host_ptr, size_t tsize); //NVidia GPU only
 int host_mem_free_pin(void *host_ptr); //NVidia GPU only
 int host_mem_register(void *host_ptr, size_t tsize); //NVidia GPU only
 int host_mem_unregister(void *host_ptr); //NVidia GPU only
 int gpu_mem_alloc(void **dev_ptr, size_t tsize); //NVidia GPU only
 int gpu_mem_free(void *dev_ptr); //NVidia GPU only
// NVidia GPU operations (NV-TAL):
//  Device id conversion:
 int encode_device_id(int dev_kind, int dev_num);
 int decode_device_id(int dev_id, int *dev_kind);
//  NV-TAL debugging:
 int gpu_get_error_count();
 int gpu_get_debug_dump(int *dump);
//  NV-TAL initialization/shutdown (for internal use only):
 int init_gpus(int gpu_beg, int gpu_end);
 int free_gpus(int gpu_beg, int gpu_end);
//  NV-TAL internal control:
 int gpu_set_shmem_width(int width);
 void gpu_set_event_policy(int alg);
 void gpu_set_transpose_algorithm(int alg);
 void gpu_set_matmult_algorithm(int alg);
 int gpu_print_stats(int gpu_num);
//  NV-TAL tensor block API:
 int tensBlck_create(tensBlck_t **ctens);
 int tensBlck_destroy(tensBlck_t *ctens);
 int tensBlck_construct(tensBlck_t *ctens, int dev_kind, int dev_num, int data_kind, int trank,
                        const int *dims, const int *divs, const int *grps, const int *prmn,
                        void *addr_host, void *addr_gpu, int entry_host, int entry_gpu, int entry_const);
 int tensBlck_alloc(tensBlck_t *ctens, int dev_num, int data_kind, int trank, const int *dims);
 int tensBlck_free(tensBlck_t *ctens);
 int tensBlck_acc_id(const tensBlck_t *ctens, int *dev_kind, int *entry_gpu, int *entry_const, int *data_kind, int *there);
 int tensBlck_set_presence(tensBlck_t *ctens);
 int tensBlck_set_absence(tensBlck_t *ctens);
 int tensBlck_present(const tensBlck_t *ctens);
 int tensBlck_hab_free(tensBlck_t *ctens);
 size_t tensBlck_volume(const tensBlck_t *ctens);
//  NV-TAL CUDA task API:
 int cuda_task_create(cudaTask_t **cuda_task);
 int cuda_task_destroy(cudaTask_t *cuda_task);
 int cuda_task_clean(cudaTask_t *cuda_task);
 int cuda_task_gpu_id(const cudaTask_t *cuda_task);
 int cuda_task_status(cudaTask_t *cuda_task);
 int cuda_task_complete(cudaTask_t *cuda_task);
 int cuda_task_wait(cudaTask_t *cuda_task);
 int cuda_tasks_wait(int num_tasks, cudaTask_t **cuda_tasks, int* task_stats);
 float cuda_task_time(const cudaTask_t *cuda_task, float *in_copy, float *out_copy, float *comp);
//  NV-TAL query/action API:
 int gpu_is_mine(int gpu_num);
 int gpu_busy_least();
 int gpu_activate(int gpu_num);
//  NV-TAL tensor operations:
 int gpu_put_arg(tensBlck_t *ctens);
 int gpu_get_arg(tensBlck_t *ctens);
 int gpu_put_arg_(tensBlck_t *ctens, cudaTask_t *cuda_task);
 int gpu_get_arg_(tensBlck_t *ctens, cudaTask_t *cuda_task);
 int gpu_array_2norm2_r4(size_t size, const float *arr, float *norm2);
 int gpu_array_2norm2_r8(size_t size, const double *arr, double *norm2);
 int gpu_matrix_multiply_tn_r4(size_t ll, size_t lr, size_t lc, const float *lmat, const float *rmat, float *dmat);
 int gpu_matrix_multiply_tn_r8(size_t ll, size_t lr, size_t lc, const double *lmat, const double *rmat, double *dmat);
 int gpu_tensor_block_init_(tensBlck_t *ctens, double val, int copy_back, cudaTask_t *cuda_task);
 int gpu_tensor_block_scale_(tensBlck_t *ctens, double val, int copy_back, cudaTask_t *cuda_task);
 int gpu_tensor_block_add_dlf_(tensBlck_t *ctens0, tensBlck_t *ctens1, double val, int copy_back, cudaTask_t *cuda_task);
 int gpu_tensor_block_copy_dlf(const int *dim_trn, tensBlck_t *tens_in, tensBlck_t *tens_out);
 int gpu_tensor_block_copy_dlf_(const int *dim_trn, tensBlck_t *tens_in, tensBlck_t *tens_out,
                                int copy_back, cudaTask_t *cuda_task);
 int gpu_tensor_block_contract_dlf_(const int *cptrn, const tensBlck_t *ltens, const tensBlck_t *rtens,
                                    tensBlck_t *dtens, int copy_back, cudaTask_t *cuda_task);
#endif
#ifdef __cplusplus
}
#endif

//END OF TENSOR_ALGEBRA_H
#endif
