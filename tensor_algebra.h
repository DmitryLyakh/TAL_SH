/** Parameters, derived types, and function prototypes used
    in tensor_algebra_gpu_nvidia.cu, c_proc_bufs.cu (NV-TAL).
REVISION: 2015/02/07
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
 # -DNO_GPU: disables GPU usage (CPU structures only).
 # -DNO_BLAS: cuBLAS calls will be replaced by in-house routines.
 # -DDIL_DEBUG_GPU: collection of debugging information will be activated.
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
     Host: {0};
     NVidia GPU: {1..MAX_GPUS_PER_NODE};
     Intel Xeon Phi: {MAX_GPUS_PER_NODE+1:MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE};
     AMD GPU: {MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+1:MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+MAX_AMDS_PER_NODE}, etc.
    DEVICE_ID is used in tensBlck_t: If tensor elements are already on the Device it is positive, otherwise negative.
 # MAX_SCR_ENTRY_COUNT regulates the maximal amount of additional device argument-buffer entries
    allocated per tensor operation (it is no more than 3 for tensor contractions).
 # MAX_GPU_ARGS regulates the maximal allowed amount of device argument-buffer entries.
 # CUDA_TASK is considered completed successfully if the value of the .task_error field equals zero.
    Negative .task_error means that either the CUDA task is empty or it is in progress.
    In the former case, .gpu_id=-1 and .task_stream is undefined.
    Positive .task_error means that an error occured during the task scheduling/execution process.
**/
//BEGINNING OF TENSOR_ALGEBRA_H
#ifndef TENSOR_ALGEBRA_H
#define TENSOR_ALGEBRA_H

//GLOBAL PARAMETERS:
#define MAX_TENSOR_RANK 32         //max allowed tensor rank: Must be multiple of 4
#define MAX_GPU_ARGS 128           //max allowed number of tensor arguments simultaneously residing on a GPU: Must be multiple of 8
#define MAX_SCR_ENTRY_COUNT 3      //max allowed number of additional GPU argument entries allocated per tensor operation

//DEVICE KINDS:
#define MAX_GPUS_PER_NODE 8        //max allowed number of NVidia GPUs on a node
#define MAX_MICS_PER_NODE 8        //max allowed number of Intel MICs on a node
#define MAX_AMDS_PER_NODE 8        //max allowed number of AMD GPUs on a node
#define DEV_HOST 0                 //(multi-)CPU Host
#define DEV_NVIDIA_GPU 1           //NVidia GPU
#define DEV_INTEL_MIC 2            //Intel Xeon Phi
#define DEV_AMD_GPU 3              //AMD GPU
#define DEV_MAX 1+MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+MAX_AMDS_PER_NODE

//KERNEL PARAMETERS:
#define GPU_CACHE_LINE_LEN 128     //cache line length in bytes
#define MAX_CUDA_BLOCKS 1024       //max number of CUDA thread blocks per kernel
#if __CUDA_ARCH__ >= 300
#define TENS_TRANSP_BUF_SIZE 2560  //buffer size (elements) for <gpu_tensor_block_copy_dlf_XX__>
#else
#define TENS_TRANSP_BUF_SIZE 1536  //buffer size (elements) for <gpu_tensor_block_copy_dlf_XX__>
#endif
#define TENS_TRANSP_TAB_SIZE 69    //look up table size (integers) for <gpu_tensor_block_copy_dlf_XX__>
#define MAT_MULT_TILE_DIM 16       //tile dimension size for <gpu_matrix_multiply_tn_XX__>
#define THRDS_ARRAY_PRODUCT 256    //threads per block for <gpu_array_product_XX__>
#define THRDS_ARRAY_NORM2 256      //threads per block for <gpu_array_2norm2_XX__>
#define THRDS_ARRAY_INIT 256       //threads per block for <gpu_array_init_XX__>
#define THRDS_ARRAY_SCALE 256      //threads per block for <gpu_array_scale_XX__> and <gpu_array_dot_product_XX__>
#define THRDS_ARRAY_ADD 256        //threads per block for <gpu_array_add_XX__>
#if __CUDA_ARCH__ >= 200
#define THRDS_TENSOR_COPY 256      //threads per block for <gpu_tensor_block_copy_dlf_XX__>
#else
#define THRDS_TENSOR_COPY 192      //threads per block for <gpu_tensor_block_copy_dlf_XX__>
#endif
#define THRDS_TENSOR_COPY_SCAT 256 //threads per block for <gpu_tensor_block_copy_scatter_dlf_XX__>

//DATA KINDS:
#define R4 4     //float data kind (keep consistent with c_process.f90::tens_blck_pack/unpack)
#define R8 8     //double data kind (keep consistent with c_process.f90::tens_blck_pack/unpack)
#define C8 16    //double complex data kind (keep consistent with c_process.f90::tens_blck_pack/unpack)

//CUDA TASK STATUS:
#define CUDA_TASK_ERROR -1
#define CUDA_TASK_EMPTY 0
#define CUDA_TASK_SCHEDULED 1
#define CUDA_TASK_STARTED 2
#define CUDA_TASK_INPUT_THERE 3
#define CUDA_TASK_OUTPUT_THERE 4
#define CUDA_TASK_COMPLETED 5

//ALIASES:
#define NOPE 0
#define YEP 1
#define GPU_MINE 1
#define GPU_MINE_CUBLAS 2
#define NO_COPY_BACK 0
#define COPY_BACK 1
#define EVENTS_OFF 0
#define EVENTS_ON 1
#define BLAS_ON 0
#define BLAS_OFF 1
#define EFF_TRN_OFF 0
#define EFF_TRN_ON 1

//MACRO FUNCTIONS:
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

//DERIVED TYPES:
// Tensor block:
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

#ifndef NO_GPU
// CUDA task (returned by non-blocking CUDA calling functions):
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

//FUNCTION PROTOTYPES:
#ifdef __cplusplus
extern "C"{
#endif
 int arg_buf_allocate(size_t *arg_buf_size, int *arg_max, int gpu_beg, int gpu_end);
 int arg_buf_deallocate(int gpu_beg, int gpu_end);
 int arg_buf_clean_host();
 int arg_buf_clean_gpu(int gpu_num);
 int get_blck_buf_sizes_host(size_t *blck_sizes);
 int get_blck_buf_sizes_gpu(int gpu_num, size_t *blck_sizes);
 int get_buf_entry_host(size_t bsize, char **entry_ptr, int *entry_num);
 int free_buf_entry_host(int entry_num);
 int get_buf_entry_gpu(int gpu_num, size_t bsize, char **entry_ptr, int *entry_num);
 int free_buf_entry_gpu(int gpu_num, int entry_num);
 int const_args_entry_get(int gpu_num, int *entry_num);
 int const_args_entry_free(int gpu_num, int entry_num);
 char* ptr_offset(char *byte_ptr, size_t byte_offset);
#ifndef NO_GPU
 int host_mem_alloc_pin(void **host_ptr, size_t tsize);
 int host_mem_free_pin(void *host_ptr);
 int host_mem_register(void *host_ptr, size_t tsize);
 int host_mem_unregister(void *host_ptr);
 int gpu_mem_alloc(void **dev_ptr, size_t tsize);
 int gpu_mem_free(void *dev_ptr);
 int gpu_get_error_count();
 int gpu_get_debug_dump(int *dump);
 void gpu_set_event_policy(int alg);
 void gpu_set_transpose_algorithm(int alg);
 void gpu_set_matmult_algorithm(int alg);
 int encode_device_id(int dev_kind, int dev_num);
 int decode_device_id(int dev_id, int *dev_kind);
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
 int cuda_task_create(cudaTask_t **cuda_task);
 int cuda_task_clean(cudaTask_t *cuda_task);
 int cuda_task_destroy(cudaTask_t *cuda_task);
 int cuda_task_gpu_id(const cudaTask_t *cuda_task);
 int cuda_task_status(cudaTask_t *cuda_task);
 int cuda_task_complete(cudaTask_t *cuda_task);
 int cuda_task_wait(cudaTask_t *cuda_task);
 int cuda_tasks_wait(int num_tasks, cudaTask_t **cuda_tasks, int* task_stats);
 float cuda_task_time(const cudaTask_t *cuda_task, float *in_copy, float *out_copy, float *comp);
 int init_gpus(int gpu_beg, int gpu_end);
 int free_gpus(int gpu_beg, int gpu_end);
 int gpu_is_mine(int gpu_num);
 int gpu_busy_least();
 int gpu_activate(int gpu_num);
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
