/** Tensor Algebra Library for NVidia GPU: NV-TAL (CUDA based).
AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com, liakhdi@ornl.gov
REVISION: 2019/02/19

Copyright (C) 2014-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2019 Oak Ridge National Laboratory (UT-Battelle)

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
OPTIONS:
 # -D CUDA_ARCH=350: target GPU compute capability (default is 130);
 # -D NO_GPU: disables GPU usage;
 # -D NO_BLAS: disables cuBLAS calls, they will be replaced by in-house routines (slower);
 # -D USE_CUTT: enables an optimized tensor transpose via the cuTT library;
 # -D DEBUG_GPU: collection of debugging information will be activated;
NOTES:
 # Minimal required compute capability is 1.1 (1.3 for double precision).
 # cuBLAS.v2 is required when BLAS is enabled.
 # Non-blocking tensor algebra functions carry an additional output argument <cuda_task> (task handle).
 # Non-blocking tensor algebra functions carry an additional input argument <coherence_ctrl>
   which controls the tensor data consistency synchronization accross different devices
   after the tensor operation has completed successfully.
FOR DEVELOPERS ONLY:
 # Currently used device resources:
    - Global memory pointer (any device);
    - Argument buffer entry handle (any device);
    - Multi-index entry * (Host pinned memory, entry length = MAX_TENSOR_RANK);
    - GPU constant-memory entry handle (Nvidia GPU);
    - CUDA stream handle (Nvidia GPU);
    - CUDA event handle (Nvidia GPU).
 # A life cycle of a C object (for example, tensBlck_t):
    a) Allocate memory for the object itself, if needed: Suffix _alloc or _create (includes cleaning);
    b) Clean (initialize to null) an allocated (empty) object: Suffix _clean (normally included in _create);
    c) Construct (define or redefine) an existing object (resources will be acquired/released): Suffix _construct;
    d) Destruct a defined object (resources will be released, the object will be reset to clean): Suffix _destruct;
    e) Free the memory occupied by an object: Suffix _free or _destroy (may include _destruct, if needed).
   Thus, as a rule, the device resource acquisition/release occurs solely in _construct and _destruct functions.
 # A state of a C object:
    a) Undefined: After the memory allocation (either dynamic or static);
    b) Defined-empty (clean): After cleaning or destruction (dynamic object creation produces a clean object);
    c) Defined to a value (value-defined): After construction;
    d) Dead: After memory deallocation (if it was allocated dynamically).
 # Resource acquisition/release:
    - Tensor block constructor/destructor acquires/releases global memory resources, including
      both pointers and buffer entries, as well as multi-index bank entries (pinned Host memory).
    - CUDA task constructor/destructor acquires/releases CUDA resources (stream, events).
    - Tensor operation scheduling functions acquire GPU global memory resources,
      GPU constant memory resources, Host pinned multi-index entries.
    - CUDA task completion/error query functions release GPU global memory resources,
      GPU constant memory resources, Host pinned multi-index entries.
    - Coherence control is only applied to successfully finished CUDA tasks.
 # Functions which construct tensor blocks or perform asynchronous operations on them
   allocate resources (global/constant memory, etc). In case the corresponding resource
   allocator returns TRY_LATER or DEVICE_UNABLE (or an error), the corresponding function
   must clean the partially created tensor block or the CUDA task before returning:
   The corresponding object will be kept in its initial state if no SUCCESS.
 # Some CUDA kernels operating on two or more arguments assume no aliases
   for GPU pointers (__restrict__). Check each specific operation to see whether
   it is ok for the two tensor arguments to refer to the same tensor body.
TO BE FIXED:
 # In tensor operation scheduling functions, if a scheduling error occurs after
   the data transfer or CUDA kernel has been scheduled, the CUDA task finalization
   must not begin until the partially scheduled CUDA task has completed on GPU.
   Insert cudaStreamSynchronize in the finalization procedure.
 # Invoke cudaDeviceCanAccessPeer() in tensor operations to check whether
   two devices of the same kind can access each other's memory.
 # Account for implicit data transfers to/from peer GPUs in their statistics.
 # User-provided Alpha factors for gpu_tensor_block_contract() and
   gpu_tensor_block_add() reside on Host, thus requiring a slab in GPU
   memory (either global or constant) as a temporary for BLAS references.
**/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tensor_algebra.h" /*includes mem_manager.h*/

#ifndef NO_GPU
//PARAMETERS:
#define GPU_DEBUG_DUMP_SIZE 128 //size of the GPU debug dump (int array)
#endif /*NO_GPU*/
//----------------------------------------------------------------------
//FUNCTION PROTOTYPES:
// IMPORTED:
#ifdef __cplusplus
extern "C" {
#endif
 void get_contr_permutations(int lrank, int rrank, const int *cptrn, int conj_bits,
                             int *dprm, int *lprm, int *rprm, int *ncd, int *nlu, int *nru, int *ierr);
#ifdef __cplusplus
}
#endif
// LOCAL (PRIVATE):
static int prmn_convert(int n, const int *o2n, int *n2o);
static int non_trivial_prmn(int n, const int *prm);
#ifndef NO_GPU
static int cuda_stream_get(int gpu_num, int * cuda_stream_handle);
static int cuda_stream_release(int gpu_num, int cuda_stream_handle);
static cudaStream_t * cuda_stream_ptr(int gpu_num, int cuda_stream_handle);
static int cuda_event_get(int gpu_num, int * cuda_event_handle);
static int cuda_event_release(int gpu_num, int cuda_event_handle);
static cudaEvent_t * cuda_event_ptr(int gpu_num, int cuda_event_handle);
static void limit_cuda_blocks2d(int max_blocks, int *bx, int *by);
static int tens_op_best_gpu(const tensBlck_t *tens0 = NULL, const tensBlck_t *tens1 = NULL, const tensBlck_t *tens2 = NULL);
static int cuda_task_set_arg(cudaTask_t *cuda_task, unsigned int arg_num, tensBlck_t *tens_p);
static int cuda_task_set_prefactor(cudaTask_t *cuda_task, talshComplex4 prefactor);
static int cuda_task_set_prefactor(cudaTask_t *cuda_task, talshComplex8 prefactor);
static int cuda_task_record(cudaTask_t *cuda_task, unsigned int coh_ctrl, unsigned int err_code = 0);
static int cuda_task_finalize(cudaTask_t *cuda_task);
// CUDA KERNELS:
template <typename T>
__global__ void gpu_array_norm2__(size_t tsize, const T * __restrict__ arr, volatile double * bnorm2);
template <typename T>
__global__ void gpu_array_init__(size_t tsize, T * arr, T val);
template <typename T>
__global__ void gpu_scalar_multiply__(const T * left_arg, const T * right_arg, T * dest_arg, T alpha,
                                      int left_conj = 0, int right_conj = 0);
template <typename T>
__global__ void gpu_array_scale__(size_t tsize, T * arr, T alpha);
template <typename T>
__global__ void gpu_array_add__(size_t tsize, T * __restrict__ arr0, const T * __restrict__ arr1, T alpha, int left_conj = 0);
template <typename T>
__global__ void gpu_array_add__(size_t tsize, T * __restrict__ arr0, const T * __restrict__ arr1, const T * __restrict__ scalar,
                                T alpha, int left_conj = 0);
template <typename T>
__global__ void gpu_array_dot_product__(size_t tsize, const T * arr1, const T * arr2, volatile T * dprod,
                                        T alpha, int left_conj = 0, int right_conj = 0);
template <typename T>
__global__ void gpu_array_product__(size_t tsize1, const T * arr1, size_t tsize2, const T * arr2, T * arr0,
                                    T alpha, int left_conj = 0, int right_conj = 0);
template <typename T>
__global__ void gpu_tensor_block_copy_dlf__(int dmo, int drc, int dim_num, int const_args_pos,
                                            const T * __restrict__ tens_in, T * __restrict__ tens_out);
template <typename T>
__global__ void gpu_tensor_block_copy_scatter_dlf__(int dmo, int drc, int dim_num, int const_args_pos,
                                                    const T * __restrict__ tens_in, T * __restrict__ tens_out);
template <typename T>
__global__ void gpu_matrix_multiply_tn__(size_t ll, size_t lr, size_t lc, const T * arg1, const T * arg2, T * arg0, T alpha);
template <typename T>
__global__ void gpu_matrix_multiply_nt__(size_t ll, size_t lr, size_t lc, const T * arg1, const T * arg2, T * arg0, T alpha);
template <typename T>
__global__ void gpu_matrix_multiply_nn__(size_t ll, size_t lr, size_t lc, const T * arg1, const T * arg2, T * arg0, T alpha);
template <typename T>
__global__ void gpu_matrix_multiply_tt__(size_t ll, size_t lr, size_t lc, const T * arg1, const T * arg2, T * arg0, T alpha);

#endif /*NO_GPU*/
//----------------------------------------------------------------------------------------------------
//PARAMETERS:
static int VERBOSE=1; //verbosity for error messages
static int DEBUG=0; //debugging mode
#ifndef NO_GPU
//GLOBAL DATA:
// GPU control on the current MPI process:
static int gpu_up[MAX_GPUS_PER_NODE]={GPU_OFF}; //GPU_OFF(0): GPU is disabled; GPU_MINE(1): GPU is enabled; GPU_MINE_CUBLAS(2): GPU is BLAS enabled
static cudaDeviceProp gpu_prop[MAX_GPUS_PER_NODE]; //properties of all GPUs present on the node
static talsh_stats_t gpu_stats[MAX_GPUS_PER_NODE]; //runtime statistics for all GPUs present on the node
#ifndef NO_BLAS
// Infrastructure for CUBLAS:
static cublasHandle_t cublas_handle[MAX_GPUS_PER_NODE]; //each GPU present on a node obtains its own cuBLAS context handle
#endif /*NO_BLAS*/
// Slabs for the GPU asynchronous resources:
//  CUDA stream handles:
static cudaStream_t CUDAStreamBank[MAX_GPUS_PER_NODE][MAX_CUDA_TASKS]; //pre-allocated CUDA stream handles (for each CUDA device)
static int CUDAStreamFreeHandle[MAX_GPUS_PER_NODE][MAX_CUDA_TASKS]; //free CUDA stream handles
static int CUDAStreamFFE[MAX_GPUS_PER_NODE]; //number of free handles left in CUDAStreamFreeHandle
//  CUDA event handles:
static cudaEvent_t CUDAEventBank[MAX_GPUS_PER_NODE][MAX_CUDA_EVENTS]; //pre-allocated CUDA event handles (for each CUDA device)
static int CUDAEventFreeHandle[MAX_GPUS_PER_NODE][MAX_CUDA_EVENTS]; //free CUDA event handles
static int CUDAEventFFE[MAX_GPUS_PER_NODE]; //number of free handles left in CUDAEventFreeHandle
// Mapped slab of tensor operation prefactors for GPU usage:
static slab_t prefactors;         //mapped slab of prefactors
static void * gpu_prefs_base_ptr; //mapped device pointer of the slab base
// Slab of GPU constant memory arguments for each GPU (managed by "mem_manager.cpp"):
__device__ __constant__ int const_args_dims[MAX_GPU_ARGS][MAX_TENSOR_RANK]; //storage for device constant memory arguments: dimension extents
__device__ __constant__ int const_args_prmn[MAX_GPU_ARGS][MAX_TENSOR_RANK]; //storage for device constant memory arguments: permutation
// GPU error control and debugging for each GPU:
__device__ static int gpu_error_count=0; //total number of CUDA errors registered on device till the current moment
__device__ static int gpu_debug_dump[GPU_DEBUG_DUMP_SIZE]; //debug dump
// Global CUDA event recording policy:
static int PRINT_TIMING=1; //non-zero value enables time printing statements
// Infrastructure for function <gpu_tensor_block_copy_dlf> (blocking and non-blocking):
#ifdef USE_CUTT
static int TRANS_SHMEM=EFF_TRN_ON_CUTT; //switch between shared-memory tensor transpose and scatter tensor transpose
#else
static int TRANS_SHMEM=EFF_TRN_ON; //switch between shared-memory tensor transpose and scatter tensor transpose
#endif /*USE_CUTT*/
// Infrastructure for <gpu_tensor_block_contract_dlf> (non-blocking):
#ifndef NO_BLAS
static int DISABLE_BLAS=0; //non-zero value will disable cuBLAS usage (if it had been cuBLAS compiled/linked)
#else
static int DISABLE_BLAS=1; //non-zero value will disable cuBLAS usage (if it had been cuBLAS compiled/linked)
#endif /*NO_BLAS*/
static cudaTask_t * LastTask[MAX_GPUS_PER_NODE]; //last CUDA task successfully scheduled on each GPU
__device__ __constant__ static float sgemm_alpha=1.0f;                //default alpha constant for SGEMM
__device__ __constant__ static float sgemm_beta=1.0f;                 //default beta constant SGEMM
__device__ __constant__ static double dgemm_alpha=1.0;                //default alpha constant for DGEMM
__device__ __constant__ static double dgemm_beta=1.0;                 //default beta constant DGEMM
__device__ __constant__ static cuComplex cgemm_alpha={1.0f,0.0f};     //default alpha constant CGEMM
__device__ __constant__ static cuComplex cgemm_beta={1.0f,0.0f};      //default beta constant CGEMM
__device__ __constant__ static cuDoubleComplex zgemm_alpha={1.0,0.0}; //default alpha constant ZGEMM
__device__ __constant__ static cuDoubleComplex zgemm_beta={1.0,0.0};  //default beta constant ZGEMM
// Infrastructure for kernels <gpu_array_norm2__>:
__device__ static int norm2_wr_lock=0; //write lock shared by all <gpu_array_norm2__> running on GPU
// Infrastructure for kernels <gpu_array_dot_product__>:
__device__ static int dot_product_wr_lock=0; //write lock shared by all <gpu_array_dot_product__> running on GPU
#endif /*NO_GPU*/
//---------------------------------------------------------------------------------------------------
#ifndef NO_GPU
//CUDA KERNELS:
// SUM OF THE SQUARES OF ABSOLUTE VALUES OF ALL ARRAY ELEMENTS:
// REAL:
template <typename T>
__global__ void gpu_array_norm2__(size_t tsize, const T * __restrict__ arr, volatile double * bnorm2)
/** Computes the squared 2-norm of array arr(0:tsize-1)
INPUT:
 # tsize - size of the array;
 # arr(0:tsize-1) - array;
 # bnorm2 - must be zero on entrance (resides on device as well);
OUTPUT:
 # bnorm2 - squared 2-norm of the array (resides on device as well);
**/
{
 size_t i,n;
 double _thread_norm2;
 extern __shared__ double thread_norms2[]; //size = blockDim.x*sizeof(double) Bytes per thread block

 n=gridDim.x*blockDim.x; _thread_norm2=0.0;
 for(i=blockIdx.x*blockDim.x+threadIdx.x;i<tsize;i+=n) _thread_norm2+=arr[i]*arr[i];
 thread_norms2[threadIdx.x]=_thread_norm2;
 __syncthreads();
 if(threadIdx.x == 0){ //global reduction among thread blocks
  _thread_norm2=thread_norms2[0];
  for(i=1;i<blockDim.x;i++) _thread_norm2+=thread_norms2[i];
  i=1; while(i == 1){i=atomicMax(&norm2_wr_lock,1);} //waiting for the lock to unlock, then lock
  __threadfence();
  *bnorm2+=_thread_norm2;
  __threadfence();
  i=atomicExch(&norm2_wr_lock,0); //unlock
 }
 __syncthreads();
 return;
}
// COMPLEX4:
template <>
__global__ void gpu_array_norm2__<talshComplex4>(size_t tsize, const talshComplex4 * __restrict__ arr, volatile double * bnorm2)
/** Computes the squared 2-norm of array arr(0:tsize-1)
INPUT:
 # tsize - size of the array;
 # arr(0:tsize-1) - array;
 # bnorm2 - must be zero on entrance (resides on device as well);
OUTPUT:
 # bnorm2 - squared 2-norm of the array (resides on device as well);
**/
{
 size_t i,n;
 double _thread_norm2;
 extern __shared__ double thread_norms2[]; //size = blockDim.x*sizeof(double) Bytes per thread block

 n=gridDim.x*blockDim.x; _thread_norm2=0.0;
 for(i=blockIdx.x*blockDim.x+threadIdx.x;i<tsize;i+=n) _thread_norm2+=talshComplex4Asq(arr[i]);
 thread_norms2[threadIdx.x]=_thread_norm2;
 __syncthreads();
 if(threadIdx.x == 0){ //global reduction among thread blocks (one thread per block)
  _thread_norm2=thread_norms2[0];
  for(i=1;i<blockDim.x;i++) _thread_norm2+=thread_norms2[i];
  i=1; while(i == 1){i=atomicMax(&norm2_wr_lock,1);} //waiting for the lock to unlock, then lock
  __threadfence();
  *bnorm2+=_thread_norm2;
  __threadfence();
  i=atomicExch(&norm2_wr_lock,0); //unlock
 }
 __syncthreads();
 return;
}
// COMPLEX8:
template <>
__global__ void gpu_array_norm2__<talshComplex8>(size_t tsize, const talshComplex8 * __restrict__ arr, volatile double * bnorm2)
/** Computes the squared 2-norm of array arr(0:tsize-1)
INPUT:
 # tsize - size of the array;
 # arr(0:tsize-1) - array;
 # bnorm2 - must be zero on entrance (resides on device as well);
OUTPUT:
 # bnorm2 - squared 2-norm of the array (resides on device as well);
**/
{
 size_t i,n;
 double _thread_norm2;
 extern __shared__ double thread_norms2[]; //size = blockDim.x*sizeof(double) Bytes per thread block

 n=gridDim.x*blockDim.x; _thread_norm2=0.0;
 for(i=blockIdx.x*blockDim.x+threadIdx.x;i<tsize;i+=n) _thread_norm2+=talshComplex8Asq(arr[i]);
 thread_norms2[threadIdx.x]=_thread_norm2;
 __syncthreads();
 if(threadIdx.x == 0){ //global reduction among thread blocks (one thread per block)
  _thread_norm2=thread_norms2[0];
  for(i=1;i<blockDim.x;i++) _thread_norm2+=thread_norms2[i];
  i=1; while(i == 1){i=atomicMax(&norm2_wr_lock,1);} //waiting for the lock to unlock, then lock
  __threadfence();
  *bnorm2+=_thread_norm2;
  __threadfence();
  i=atomicExch(&norm2_wr_lock,0); //unlock
 }
 __syncthreads();
 return;
}
//------------------------------------------------------------
// ARRAY INITIALIZATION:
template <typename T>
__global__ void gpu_array_init__(size_t tsize, T * arr, T val)
/** arr(0:tsize-1)=val **/
{
 size_t _ti = blockIdx.x*blockDim.x + threadIdx.x;
 size_t _gd = gridDim.x*blockDim.x;
 for(size_t l = _ti; l < tsize; l += _gd) arr[l] = val;
 return;
}
//---------------------------------------------------------------------------------------------------
// SCALAR MULTIPLICATION:
// REAL:
template <typename T>
__global__ void gpu_scalar_multiply__(const T * left_arg, const T * right_arg, T * dest_arg, T alpha,
                                      int left_conj, int right_conj)
/** Scalar += Scalar * Scalar * Alpha **/
{
 if(blockIdx.x == 0 && threadIdx.x == 0){
  *dest_arg+=(*left_arg)*(*right_arg)*alpha;
 }
 return;
}
// COMPLEX4:
template <>
__global__ void gpu_scalar_multiply__<talshComplex4>(const talshComplex4 * left_arg, const talshComplex4 * right_arg,
                                                     talshComplex4 * dest_arg, talshComplex4 alpha,
                                                     int left_conj, int right_conj)
/** Scalar += Scalar * Scalar * Alpha **/
{
 if(blockIdx.x == 0 && threadIdx.x == 0){
  if(left_conj != 0){
   if(right_conj != 0){
    *dest_arg=talshComplex4Add(*dest_arg,talshComplex4Mul(talshComplex4Mul(talshComplex4Conjg(*left_arg),talshComplex4Conjg(*right_arg)),alpha));
   }else{
    *dest_arg=talshComplex4Add(*dest_arg,talshComplex4Mul(talshComplex4Mul(talshComplex4Conjg(*left_arg),*right_arg),alpha));
   }
  }else{
   if(right_conj != 0){
    *dest_arg=talshComplex4Add(*dest_arg,talshComplex4Mul(talshComplex4Mul(*left_arg,talshComplex4Conjg(*right_arg)),alpha));
   }else{
    *dest_arg=talshComplex4Add(*dest_arg,talshComplex4Mul(talshComplex4Mul(*left_arg,*right_arg),alpha));
   }
  }
 }
 return;
}
// COMPLEX8:
template <>
__global__ void gpu_scalar_multiply__<talshComplex8>(const talshComplex8 * left_arg, const talshComplex8 * right_arg,
                                                     talshComplex8 * dest_arg, talshComplex8 alpha,
                                                     int left_conj, int right_conj)
/** Scalar += Scalar * Scalar * Alpha **/
{
 if(blockIdx.x == 0 && threadIdx.x == 0){
  if(left_conj != 0){
   if(right_conj != 0){
    *dest_arg=talshComplex8Add(*dest_arg,talshComplex8Mul(talshComplex8Mul(talshComplex8Conjg(*left_arg),talshComplex8Conjg(*right_arg)),alpha));
   }else{
    *dest_arg=talshComplex8Add(*dest_arg,talshComplex8Mul(talshComplex8Mul(talshComplex8Conjg(*left_arg),*right_arg),alpha));
   }
  }else{
   if(right_conj != 0){
    *dest_arg=talshComplex8Add(*dest_arg,talshComplex8Mul(talshComplex8Mul(*left_arg,talshComplex8Conjg(*right_arg)),alpha));
   }else{
    *dest_arg=talshComplex8Add(*dest_arg,talshComplex8Mul(talshComplex8Mul(*left_arg,*right_arg),alpha));
   }
  }
 }
 return;
}
//---------------------------------------------------------------
// ARRAY RESCALING:
// REAL:
template <typename T>
__global__ void gpu_array_scale__(size_t tsize, T * arr, T alpha)
/** arr(0:tsize-1)*=alpha **/
{
 size_t _ti = blockIdx.x*blockDim.x + threadIdx.x;
 size_t _gd = gridDim.x*blockDim.x;
 for(size_t l = _ti; l < tsize; l += _gd) arr[l]*=alpha;
 return;
}
// COMPLEX4:
template <>
__global__ void gpu_array_scale__<talshComplex4>(size_t tsize, talshComplex4 * arr, talshComplex4 alpha)
/** arr(0:tsize-1)*=alpha **/
{
 size_t _ti = blockIdx.x*blockDim.x + threadIdx.x;
 size_t _gd = gridDim.x*blockDim.x;
 for(size_t l = _ti; l < tsize; l += _gd) arr[l]=talshComplex4Mul(arr[l],alpha);
 return;
}
// COMPLEX8:
template <>
__global__ void gpu_array_scale__<talshComplex8>(size_t tsize, talshComplex8 * arr, talshComplex8 alpha)
/** arr(0:tsize-1)*=alpha **/
{
 size_t _ti = blockIdx.x*blockDim.x + threadIdx.x;
 size_t _gd = gridDim.x*blockDim.x;
 for(size_t l = _ti; l < tsize; l += _gd) arr[l]=talshComplex8Mul(arr[l],alpha);
 return;
}
//-----------------------------------------------------------------------------------------------------------------------
// ARRAY ADDITION:
// REAL:
template <typename T>
__global__ void gpu_array_add__(size_t tsize, T * __restrict__ arr0, const T * __restrict__ arr1, T alpha, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*alpha **/
{
 size_t _ti = blockIdx.x*blockDim.x + threadIdx.x;
 size_t _gd = gridDim.x*blockDim.x;
 for(size_t l = _ti; l < tsize; l += _gd) arr0[l]+=(arr1[l]*alpha);
 return;
}
// COMPLEX4:
template <>
__global__ void gpu_array_add__<talshComplex4>(size_t tsize, talshComplex4 * __restrict__ arr0, const talshComplex4 * __restrict__ arr1,
                                               talshComplex4 alpha, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*alpha **/
{
 size_t _ti = blockIdx.x*blockDim.x + threadIdx.x;
 size_t _gd = gridDim.x*blockDim.x;
 if(left_conj != 0){
  for(size_t l = _ti; l < tsize; l += _gd) arr0[l]=talshComplex4Add(arr0[l],talshComplex4Mul(talshComplex4Conjg(arr1[l]),alpha));
 }else{
  for(size_t l = _ti; l < tsize; l += _gd) arr0[l]=talshComplex4Add(arr0[l],talshComplex4Mul(arr1[l],alpha));
 }
 return;
}
// COMPLEX8:
template <>
__global__ void gpu_array_add__<talshComplex8>(size_t tsize, talshComplex8 * __restrict__ arr0, const talshComplex8 * __restrict__ arr1,
                                               talshComplex8 alpha, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*alpha **/
{
 size_t _ti = blockIdx.x*blockDim.x + threadIdx.x;
 size_t _gd = gridDim.x*blockDim.x;
 if(left_conj != 0){
  for(size_t l = _ti; l < tsize; l += _gd) arr0[l]=talshComplex8Add(arr0[l],talshComplex8Mul(talshComplex8Conjg(arr1[l]),alpha));
 }else{
  for(size_t l = _ti; l < tsize; l += _gd) arr0[l]=talshComplex8Add(arr0[l],talshComplex8Mul(arr1[l],alpha));
 }
 return;
}
//------------------------------------------------------------------------------------------------------------------------------
// ARRAY ADDITION AND SCALING:
// REAL:
template <typename T>
__global__ void gpu_array_add__(size_t tsize, T * __restrict__ arr0, const T * __restrict__ arr1, const T * __restrict__ scalar,
                                T alpha, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*scalar*alpha **/
{
 size_t _ti = blockIdx.x*blockDim.x + threadIdx.x;
 size_t _gd = gridDim.x*blockDim.x;
 T pref = (*scalar) * alpha;
 for(size_t l = _ti; l < tsize; l += _gd) arr0[l]+=(arr1[l]*pref);
 return;
}
// COMPLEX4:
template <>
__global__ void gpu_array_add__<talshComplex4>(size_t tsize, talshComplex4 * __restrict__ arr0, const talshComplex4 * __restrict__ arr1,
                                               const talshComplex4 * __restrict__ scalar, talshComplex4 alpha, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*scalar*alpha **/
{
 size_t _ti = blockIdx.x*blockDim.x + threadIdx.x;
 size_t _gd = gridDim.x*blockDim.x;
 talshComplex4 pref = talshComplex4Mul(*scalar,alpha);
 if(left_conj != 0){
  for(size_t l = _ti; l < tsize; l += _gd) arr0[l]=talshComplex4Add(arr0[l],talshComplex4Mul(talshComplex4Conjg(arr1[l]),pref));
 }else{
  for(size_t l = _ti; l < tsize; l += _gd) arr0[l]=talshComplex4Add(arr0[l],talshComplex4Mul(arr1[l],pref));
 }
 return;
}
// COMPLEX8:
template <>
__global__ void gpu_array_add__<talshComplex8>(size_t tsize, talshComplex8 * __restrict__ arr0, const talshComplex8 * __restrict__ arr1,
                                               const talshComplex8 * __restrict__ scalar, talshComplex8 alpha, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*scalar*alpha **/
{
 size_t _ti = blockIdx.x*blockDim.x + threadIdx.x;
 size_t _gd = gridDim.x*blockDim.x;
 talshComplex8 pref = talshComplex8Mul(*scalar,alpha);
 if(left_conj != 0){
  for(size_t l = _ti; l < tsize; l += _gd) arr0[l]=talshComplex8Add(arr0[l],talshComplex8Mul(talshComplex8Conjg(arr1[l]),pref));
 }else{
  for(size_t l = _ti; l < tsize; l += _gd) arr0[l]=talshComplex8Add(arr0[l],talshComplex8Mul(arr1[l],pref));
 }
 return;
}
//-------------------------------------------------------------------------------------------------------
// ARRAY DOT-PRODUCT:
// REAL:
template <typename T>
__global__ void gpu_array_dot_product__(size_t tsize, const T * arr1, const T * arr2, volatile T * dprod,
                                        T alpha, int left_conj, int right_conj)
/** Scalar (GPU) += arr1(0:tsize-1) * arr2(0:tsize-1) * alpha **/
{
 extern __shared__ char sh_buf[]; //size = blockDim.x * sizeof(T) Bytes per thread block
 T *dprs;
 T dpr;
 size_t l;
 unsigned int j,s;
 int i;

 dprs=(T*)(&sh_buf[0]); //dynamic shared memory buffer
 dpr=static_cast<T>(0.0);
 for(l = blockIdx.x*blockDim.x+threadIdx.x; l < tsize; l += gridDim.x*blockDim.x) dpr+=arr1[l]*arr2[l];
 dprs[threadIdx.x]=dpr*alpha;
 __syncthreads();
 s = blockDim.x;
 while(s > 1){
  j = (s+1U)>>1;
  if(threadIdx.x + j < s) dprs[threadIdx.x] += dprs[threadIdx.x+j];
  __syncthreads();
  s=j;
 }
 if(threadIdx.x == 0){
  i=1; while(i != 0){i=atomicMax(&dot_product_wr_lock,1); if(i == 0) *dprod+=dprs[0];}
  __threadfence();
  i=atomicExch(&dot_product_wr_lock,0); //unlock
 }
 __syncthreads();
 return;
}
// COMPLEX4:
template <>
__global__ void gpu_array_dot_product__<talshComplex4>(size_t tsize, const talshComplex4 * arr1, const talshComplex4 * arr2,
                                                       volatile talshComplex4 * dprod, talshComplex4 alpha,
                                                       int left_conj, int right_conj)
/** Scalar (GPU) += arr1(0:tsize-1) * arr2(0:tsize-1) * alpha **/
{
 extern __shared__ char sh_buf[]; //size = blockDim.x * sizeof(T) Bytes per thread block
 talshComplex4 *dprs;
 talshComplex4 dpr;
 size_t l;
 unsigned int j,s;
 int i;

 dprs=(talshComplex4*)(&sh_buf[0]); //dynamic shared memory buffer
 dpr=talshComplex4Set(0.0f,0.0f);
 if(left_conj != 0){
  if(right_conj != 0){
   for(l = blockIdx.x*blockDim.x+threadIdx.x; l < tsize; l += gridDim.x*blockDim.x){
    dpr=talshComplex4Add(dpr,talshComplex4Mul(talshComplex4Conjg(arr1[l]),talshComplex4Conjg(arr2[l])));
   }
  }else{
   for(l = blockIdx.x*blockDim.x+threadIdx.x; l < tsize; l += gridDim.x*blockDim.x){
    dpr=talshComplex4Add(dpr,talshComplex4Mul(talshComplex4Conjg(arr1[l]),arr2[l]));
   }
  }
 }else{
  if(right_conj != 0){
   for(l = blockIdx.x*blockDim.x+threadIdx.x; l < tsize; l += gridDim.x*blockDim.x){
    dpr=talshComplex4Add(dpr,talshComplex4Mul(arr1[l],talshComplex4Conjg(arr2[l])));
   }
  }else{
   for(l = blockIdx.x*blockDim.x+threadIdx.x; l < tsize; l += gridDim.x*blockDim.x){
    dpr=talshComplex4Add(dpr,talshComplex4Mul(arr1[l],arr2[l]));
   }
  }
 }
 dprs[threadIdx.x]=talshComplex4Mul(dpr,alpha);
 __syncthreads();
 s = blockDim.x;
 while(s > 1){
  j = (s+1U)>>1;
  if(threadIdx.x + j < s) dprs[threadIdx.x] = talshComplex4Add(dprs[threadIdx.x],dprs[threadIdx.x+j]);
  __syncthreads();
  s=j;
 }
 if(threadIdx.x == 0){
  i=1; while(i != 0){
   i=atomicMax(&dot_product_wr_lock,1);
   if(i == 0){
    dprod->x += talshComplex4Real(dprs[0]);
    dprod->y += talshComplex4Imag(dprs[0]);
   }
  }
  __threadfence();
  i=atomicExch(&dot_product_wr_lock,0); //unlock
 }
 __syncthreads();
 return;
}
// COMPLEX8:
template <>
__global__ void gpu_array_dot_product__<talshComplex8>(size_t tsize, const talshComplex8 * arr1, const talshComplex8 * arr2,
                                                       volatile talshComplex8 * dprod, talshComplex8 alpha,
                                                       int left_conj, int right_conj)
/** Scalar (GPU) += arr1(0:tsize-1) * arr2(0:tsize-1) * alpha **/
{
 extern __shared__ char sh_buf[]; //size = blockDim.x * sizeof(T) Bytes per thread block
 talshComplex8 *dprs;
 talshComplex8 dpr;
 size_t l;
 unsigned int j,s;
 int i;

 dprs=(talshComplex8*)(&sh_buf[0]); //dynamic shared memory buffer
 dpr=talshComplex8Set(0.0,0.0);
 if(left_conj != 0){
  if(right_conj != 0){
   for(l = blockIdx.x*blockDim.x+threadIdx.x; l < tsize; l += gridDim.x*blockDim.x){
    dpr=talshComplex8Add(dpr,talshComplex8Mul(talshComplex8Conjg(arr1[l]),talshComplex8Conjg(arr2[l])));
   }
  }else{
   for(l = blockIdx.x*blockDim.x+threadIdx.x; l < tsize; l += gridDim.x*blockDim.x){
    dpr=talshComplex8Add(dpr,talshComplex8Mul(talshComplex8Conjg(arr1[l]),arr2[l]));
   }
  }
 }else{
  if(right_conj != 0){
   for(l = blockIdx.x*blockDim.x+threadIdx.x; l < tsize; l += gridDim.x*blockDim.x){
    dpr=talshComplex8Add(dpr,talshComplex8Mul(arr1[l],talshComplex8Conjg(arr2[l])));
   }
  }else{
   for(l = blockIdx.x*blockDim.x+threadIdx.x; l < tsize; l += gridDim.x*blockDim.x){
    dpr=talshComplex8Add(dpr,talshComplex8Mul(arr1[l],arr2[l]));
   }
  }
 }
 dprs[threadIdx.x]=talshComplex8Mul(dpr,alpha);
 __syncthreads();
 s = blockDim.x;
 while(s > 1){
  j = (s+1U)>>1;
  if(threadIdx.x + j < s) dprs[threadIdx.x] = talshComplex8Add(dprs[threadIdx.x],dprs[threadIdx.x+j]);
  __syncthreads();
  s=j;
 }
 if(threadIdx.x == 0){
  i=1; while(i != 0){
   i=atomicMax(&dot_product_wr_lock,1);
   if(i == 0){
    dprod->x += talshComplex8Real(dprs[0]);
    dprod->y += talshComplex8Imag(dprs[0]);
   }
  }
  __threadfence();
  i=atomicExch(&dot_product_wr_lock,0); //unlock
 }
 __syncthreads();
 return;
}
//---------------------------------------------------------------------------------------------------------
// ARRAY DIRECT PRODUCT:
// REAL:
template <typename T>
__global__ void gpu_array_product__(size_t tsize1, const T * arr1, size_t tsize2, const T * arr2, T * arr0,
                                    T alpha, int left_conj, int right_conj)
/** arr0[0:tsize2-1][0:tsize1-1]+=arr1[0:tsize1-1]*arr2[0:tsize2-1]*alpha **/
{
 __shared__ T lbuf[THRDS_ARRAY_PRODUCT+1],rbuf[THRDS_ARRAY_PRODUCT];
 size_t _ib,_in,_jb,_jn,_tx,_jc;

 _tx=(size_t)threadIdx.x;
 for(_jb = blockIdx.y*THRDS_ARRAY_PRODUCT; _jb < tsize2; _jb += gridDim.y*THRDS_ARRAY_PRODUCT){
  if(_jb+THRDS_ARRAY_PRODUCT > tsize2){_jn=tsize2-_jb;}else{_jn=THRDS_ARRAY_PRODUCT;}
  if(_tx < _jn) rbuf[_tx]=arr2[_jb+_tx]*alpha;
  for(_ib = blockIdx.x*THRDS_ARRAY_PRODUCT; _ib < tsize1; _ib += gridDim.x*THRDS_ARRAY_PRODUCT){
   if(_ib+THRDS_ARRAY_PRODUCT > tsize1){_in=tsize1-_ib;}else{_in=THRDS_ARRAY_PRODUCT;}
   if(_tx < _in) lbuf[_tx]=arr1[_ib+_tx];
   __syncthreads();
   for(_jc = 0; _jc < _jn; _jc++){if(_tx < _in) arr0[(_jb+_jc)*tsize1+_ib+_tx]+=lbuf[_tx]*rbuf[_jc];}
   __syncthreads();
  }
 }
 return;
}
// COMPLEX4:
template <>
__global__ void gpu_array_product__<talshComplex4>(size_t tsize1, const talshComplex4 * arr1,
                                                   size_t tsize2, const talshComplex4 * arr2,
                                                   talshComplex4 * arr0, talshComplex4 alpha,
                                                   int left_conj, int right_conj)
/** arr0[0:tsize2-1][0:tsize1-1]+=arr1[0:tsize1-1]*arr2[0:tsize2-1]*alpha **/
{
 __shared__ talshComplex4 lbuf[THRDS_ARRAY_PRODUCT+1],rbuf[THRDS_ARRAY_PRODUCT];
 size_t _ib,_in,_jb,_jn,_tx,_jc,_ja;

 _tx=(size_t)threadIdx.x;
 for(_jb = blockIdx.y*THRDS_ARRAY_PRODUCT; _jb < tsize2; _jb += gridDim.y*THRDS_ARRAY_PRODUCT){
  if(_jb+THRDS_ARRAY_PRODUCT > tsize2){_jn=tsize2-_jb;}else{_jn=THRDS_ARRAY_PRODUCT;}
  if(right_conj != 0){
   if(_tx < _jn) rbuf[_tx]=talshComplex4Mul(talshComplex4Conjg(arr2[_jb+_tx]),alpha);
  }else{
   if(_tx < _jn) rbuf[_tx]=talshComplex4Mul(arr2[_jb+_tx],alpha);
  }
  for(_ib = blockIdx.x*THRDS_ARRAY_PRODUCT; _ib < tsize1; _ib += gridDim.x*THRDS_ARRAY_PRODUCT){
   if(_ib+THRDS_ARRAY_PRODUCT > tsize1){_in=tsize1-_ib;}else{_in=THRDS_ARRAY_PRODUCT;}
   if(left_conj != 0){
    if(_tx < _in) lbuf[_tx]=talshComplex4Conjg(arr1[_ib+_tx]);
   }else{
    if(_tx < _in) lbuf[_tx]=arr1[_ib+_tx];
   }
   __syncthreads();
   for(_jc = 0; _jc < _jn; _jc++){
    if(_tx < _in){
     _ja = (_jb+_jc)*tsize1 + (_ib+_tx);
     arr0[_ja]=talshComplex4Add(arr0[_ja],talshComplex4Mul(lbuf[_tx],rbuf[_jc]));
    }
   }
   __syncthreads();
  }
 }
 return;
}
// COMPLEX8:
template <>
__global__ void gpu_array_product__<talshComplex8>(size_t tsize1, const talshComplex8 * arr1,
                                                   size_t tsize2, const talshComplex8 * arr2,
                                                   talshComplex8 * arr0, talshComplex8 alpha,
                                                   int left_conj, int right_conj)
/** arr0[0:tsize2-1][0:tsize1-1]+=arr1[0:tsize1-1]*arr2[0:tsize2-1]*alpha **/
{
 __shared__ talshComplex8 lbuf[THRDS_ARRAY_PRODUCT+1],rbuf[THRDS_ARRAY_PRODUCT];
 size_t _ib,_in,_jb,_jn,_tx,_jc,_ja;

 _tx=(size_t)threadIdx.x;
 for(_jb = blockIdx.y*THRDS_ARRAY_PRODUCT; _jb < tsize2; _jb += gridDim.y*THRDS_ARRAY_PRODUCT){
  if(_jb+THRDS_ARRAY_PRODUCT > tsize2){_jn=tsize2-_jb;}else{_jn=THRDS_ARRAY_PRODUCT;}
  if(right_conj != 0){
   if(_tx < _jn) rbuf[_tx]=talshComplex8Mul(talshComplex8Conjg(arr2[_jb+_tx]),alpha);
  }else{
   if(_tx < _jn) rbuf[_tx]=talshComplex8Mul(arr2[_jb+_tx],alpha);
  }
  for(_ib = blockIdx.x*THRDS_ARRAY_PRODUCT; _ib < tsize1; _ib += gridDim.x*THRDS_ARRAY_PRODUCT){
   if(_ib+THRDS_ARRAY_PRODUCT > tsize1){_in=tsize1-_ib;}else{_in=THRDS_ARRAY_PRODUCT;}
   if(left_conj != 0){
    if(_tx < _in) lbuf[_tx]=talshComplex8Conjg(arr1[_ib+_tx]);
   }else{
    if(_tx < _in) lbuf[_tx]=arr1[_ib+_tx];
   }
   __syncthreads();
   for(_jc = 0; _jc < _jn; _jc++){
    if(_tx < _in){
     _ja = (_jb+_jc)*tsize1 + (_ib+_tx);
     arr0[_ja]=talshComplex8Add(arr0[_ja],talshComplex8Mul(lbuf[_tx],rbuf[_jc]));
    }
   }
   __syncthreads();
  }
 }
 return;
}
//----------------------------------------------------------------------------------------------------
// TENSOR TRANSPOSE (legacy shared-memory version):
template <typename T>
__global__ void gpu_tensor_block_copy_dlf__(int dmo, int drc, int dim_num, int const_args_pos,
                                            const T * __restrict__ tens_in, T * __restrict__ tens_out)
/**
Shared-memory version of tensor transpose: tens_out=TRN(tens_in):
INPUT:
 # dmo - dimension extents order (0: normal, as it is in <const_args>; not 0: permuted dimension order will be imposed);
 # drc - index permutation direction (0: normal, as it is in <const_args>; not 0: inversed permutation will be used);
 # dim_num - tensor block rank;
 # const_args_pos - entry in the __constant__ memory bank where tensor block dimension extents (const_args_dims)
                    and index permutation (const_args_prmn) are stored;
 # tens_in[0:] - input tensor;
OUTPUT:
 # tens_out[0:] - output (transposed) tensor;
NOTES:
 # Minimal CUDA execution configuration is <<<1,warpSize>>>
 # Number of threads per block must be multiple of the warpSize!
**/
{
 __shared__ T buf0[TENS_TRANSP_BUF_SIZE];
 __shared__ float val;
 __shared__ size_t base_in[MAX_TENSOR_RANK],base_out[MAX_TENSOR_RANK];
 __shared__ size_t ftb[TENS_TRANSP_TAB_SIZE],gtb[TENS_TRANSP_TAB_SIZE];
 __shared__ int htb[TENS_TRANSP_TAB_SIZE],stb[TENS_TRANSP_TAB_SIZE];
 __shared__ int dim_in[MAX_TENSOR_RANK],dim_out[MAX_TENSOR_RANK],o2n[MAX_TENSOR_RANK],n2o[MAX_TENSOR_RANK];
 __shared__ int pri[MAX_TENSOR_RANK],tmp0[MAX_TENSOR_RANK];
 __shared__ int err_code,minor,minor_in,minor_out,s1_ind,s1_ond,s1_step,s1_dim,s2_ind,s2_ond,s2_step,s2_dim,ns1,ns2;
 __shared__ size_t vol,vol_ext;
 size_t _vol,_addr_in,_addr_out,_addr,_work_piece;
 int i,j,k,l,m,n,_vol_minor,_vol_in,_vol_out,_s1,_s2;
/*
SHARED MEMORY USE (bytes) =
 + TENS_TRANSP_BUF_SIZE*sizeof(T)
 + MAX_TENSOR_RANK*(8+8+4+4+4+4+4+4)
 + TENS_TRANSP_TAB_SIZE*(8+8+4+4)
 + 4*15 + 8*2
MIN REGISTER USE (bytes) per thread =
 + 4*4 + 4*11 + 8*5 = 100
*/

//Determine the minor index set (only the master thread in each thread block):
 if(threadIdx.x == 0){
  err_code=0;
  if(dim_num >= 0 && dim_num <= MAX_TENSOR_RANK && blockDim.x >= warpSize && blockDim.x%warpSize == 0){
   s1_ind=dim_num+1; s2_ind=dim_num-1;
   _vol=1; for(i=0;i<dim_num;i++){
    _vol*=const_args_dims[const_args_pos][i]; if(const_args_prmn[const_args_pos][i] != i+1) s1_ind=0;
   }; vol=_vol; //total volume (number of tensor elements)
   if(s1_ind == 0){ //non-trivial permutation
// Set input/output permutations and dimension extents:
    if(drc == 0){ //normal index permutation
     for(i=0;i<dim_num;i++) o2n[i]=const_args_prmn[const_args_pos][i]-1; for(i=0;i<dim_num;i++) n2o[o2n[i]]=i;
    }else{ //inversed index permutation
     for(i=0;i<dim_num;i++) n2o[i]=const_args_prmn[const_args_pos][i]-1; for(i=0;i<dim_num;i++) o2n[n2o[i]]=i;
    }
    if(dmo == 0){ //normal dimension order
     for(i=0;i<dim_num;i++) dim_in[i]=const_args_dims[const_args_pos][i];
     for(i=0;i<dim_num;i++) dim_out[o2n[i]]=dim_in[i];
    }else{ //inversed dimension order
     for(i=0;i<dim_num;i++) dim_out[i]=const_args_dims[const_args_pos][i];
     for(i=0;i<dim_num;i++) dim_in[n2o[i]]=dim_out[i];
    }
    s1_step=dim_in[s1_ind]; s2_step=dim_in[s2_ind];
    if(_vol > TENS_TRANSP_BUF_SIZE){ //tensor block does not fit into the shared memory buffer
// Determine the input/output minor index sets and the combined minor index set:
     l=(int)(sqrt((float)TENS_TRANSP_BUF_SIZE));
     minor_in=0; _vol_in=1; for(i=0;i<dim_num;i++){j=_vol_in*dim_in[i]; if(j>l) break; minor_in++; _vol_in=j;}
     minor_out=0; _vol_out=1; for(i=0;i<dim_num;i++){j=_vol_out*dim_out[i]; if(j>l) break; minor_out++; _vol_out=j;}
     minor=minor_in; _vol_minor=_vol_in; for(i=0;i<minor_out;i++){if(n2o[i]>=minor_in){minor++; _vol_minor*=dim_out[i];}}
     m=1; _s1=0; _s2=0;
     while(_vol_minor < TENS_TRANSP_BUF_SIZE && m != 0){
      m=0;
      if(_s1 == 0){for(i=minor_in;i<dim_num;i++){if(o2n[i]<minor_out){minor_in++; _vol_in*=dim_in[i];}else{break;}}}
      if(_s2 == 0){for(i=minor_out;i<dim_num;i++){if(n2o[i]<minor_in){minor_out++; _vol_out*=dim_out[i];}else{break;}}}
      j=dim_in[minor_in]; l=dim_out[minor_out];
      if(minor_in == n2o[minor_out] && _s1+_s2 == 0){ //same candidate index to both the input and output index sets
       if(j > 1 && TENS_TRANSP_BUF_SIZE < _vol_minor*2) break;
       if(_vol_minor*j > TENS_TRANSP_BUF_SIZE){s1_ind=minor_in; s1_step=TENS_TRANSP_BUF_SIZE/_vol_minor; _s1++; _s2++;}
       minor_in++; _vol_in*=j; minor_out++; _vol_out*=j; minor++; _vol_minor*=j; m++;
      }else{ //the input and output index sets consider two different candidates
       if(_vol_minor*j*l <= TENS_TRANSP_BUF_SIZE && _s1+_s2 == 0){ //accept both, no splitting
        minor_in++; _vol_in*=j; minor_out++; _vol_out*=l; minor+=2; _vol_minor*=(j*l); m++;
       }else{ //try to accept either one of the two OR both with splitting
        if(j == 1 || l == 1){
         if(j == 1 && _s1 == 0){minor_in++; minor++; m++;}
         if(l == 1 && _s2 == 0){minor_out++; minor++; m++;}
        }else{
         if(_vol_minor*j <= TENS_TRANSP_BUF_SIZE && _vol_minor*l > TENS_TRANSP_BUF_SIZE &&
            _vol_out >= warpSize && _s1 == 0){ //accept the input index, no splitting
          minor_in++; _vol_in*=j; minor++; _vol_minor*=j; m++;
         }else if(_vol_minor*j > TENS_TRANSP_BUF_SIZE && _vol_minor*l <= TENS_TRANSP_BUF_SIZE &&
                  _vol_in >= warpSize && _s2 == 0){ //accept the output index, no splitting
          minor_out++; _vol_out*=l; minor++; _vol_minor*=l; m++;
         }else{ //splitting is unavoidable (both OR one OR none)
          if(TENS_TRANSP_BUF_SIZE >= _vol_minor*2){
           if(j >= 4 && l >= 4){ //dimension extents are large enough to be split
            if(_vol_minor*4 > TENS_TRANSP_BUF_SIZE){ //impossible to split both indices
             if(_vol_in <= _vol_out && _s1 == 0){ //split the input candidate index
              s1_ind=minor_in; s1_step=TENS_TRANSP_BUF_SIZE/_vol_minor;
              minor_in++; _vol_in*=j; minor++; _vol_minor*=j; _s1++; m++;
             }else{ //split the output candidate index
              if(_s2 == 0){
               s1_ind=n2o[minor_out]; s1_step=TENS_TRANSP_BUF_SIZE/_vol_minor;
               minor_out++; _vol_out*=l; minor++; _vol_minor*=l; _s2++; m++;
              }
             }
            }else{ //possible to split both indices
             i=(int)sqrt(((float)TENS_TRANSP_BUF_SIZE)/(float)_vol_minor); if(i < 2) i=2; //uniform splitting
             s1_step=i; s2_step=i; val=(float)_vol_out/(float)_vol_in;
             if(val < 1.0f){ //scale the initial uniform splitting to reflect the disbalance between _vol_in and _vol_out
              if(val*(float)i < 1.0f) val=1.0f/(float)i; if(val*(float)l < (float)i) val=(float)i/(float)l;
             }else{
              if(val*(float)i > (float)j) val=(float)j/(float)i; if(val > float(i)) val=(float)i;
             }
             s1_step=(int)(((float)i)*val); s2_step=(int)(((float)i)/val);
             if(s1_step >= 2 && _s1 == 0){ //&& s1_step <= dim_in[minor_in]
              s1_ind=minor_in; minor_in++; _vol_in*=j; minor++; _vol_minor*=j; _s1++; m++;
             }else{
              s1_step=dim_in[s1_ind];
             }
             if(s2_step >= 2 && _s2 == 0){ //&& s2_step <= dim_out[minor_out]
              s2_ind=n2o[minor_out]; minor_out++; _vol_out*=l; minor++; _vol_minor*=l; _s2++; m++;
             }else{
              s2_step=dim_in[s2_ind];
             }
            }
           }else if(j >= 4 && l < 4 && _s1 == 0){ //split the input candidate index
            s1_ind=minor_in; s1_step=TENS_TRANSP_BUF_SIZE/_vol_minor;
            minor_in++; _vol_in*=j; minor++; _vol_minor*=j; _s1++; m++;
           }else if(j < 4 && l >= 4 && _s2 == 0){ //split the output candidate index
            s1_ind=n2o[minor_out]; s1_step=TENS_TRANSP_BUF_SIZE/_vol_minor;
            minor_out++; _vol_out*=l; minor++; _vol_minor*=l; _s2++; m++;
           }else{ //both candidate indices have too small extent to be split: try to add one of them fully
            if(_vol_minor*j <= TENS_TRANSP_BUF_SIZE && _s1 == 0){
             minor_in++; _vol_in*=j; minor++; _vol_minor*=j; m++;
            }else if(_vol_minor*l <= TENS_TRANSP_BUF_SIZE && _s2 == 0){
             minor_out++; _vol_out*=l; minor++; _vol_minor*=l; m++;
            }
           }
          }else{ //unable to add more indices in the minor set
           break;
          }
         }
        }
       }
      }
     }
     if(s1_ind == dim_num-1 && s2_ind == dim_num-1){s2_ind=0; s2_step=dim_in[0];} //s1_ind was set while s2_ind was not
    }else{ //tensor block fits into the shared memory buffer from the beginning
     minor=dim_num; minor_in=dim_num; minor_out=dim_num; _vol_minor=_vol; _vol_in=_vol; _vol_out=_vol;
    }
// Share the tensor transpose configuration with other threads in each block:
    vol_ext=_vol/_vol_minor; s1_dim=dim_in[s1_ind]; s2_dim=dim_in[s2_ind];
// Set indexing bases (OUT:{out,in_c,ext_in}_new; IN:{in,out_c,ext_in}_old):
//  OUTPUT indexing (dim_out[], base_out[]: prioritized new numeration):
    for(i=0;i<dim_num;i++){tmp0[i]=dim_out[i];} //save output dimension extents (new numeration)
    j=0; for(i=0;i<minor_out;i++){pri[j++]=i;} //output minor index set (new numeration))
    for(i=0;i<dim_num;i++){if(o2n[i]>=minor_out) pri[j++]=o2n[i];} //{compl.input minor + external} index set (new numeration)
    j=1; for(i=0;i<dim_num;i++){dim_out[i]=j; j*=tmp0[i];} //output bases (new numeration)
    for(i=0;i<dim_num;i++){base_out[i]=dim_out[pri[i]];} //output bases (prioritized new numeration)
    for(i=0;i<dim_num;i++){dim_out[i]=tmp0[pri[i]];} //output extents (prioritized new numeration)
    for(i=0;i<dim_num;i++){if(n2o[pri[i]]==s1_ind){s1_ond=i;}else if(n2o[pri[i]]==s2_ind){s2_ond=i;}} //split indices (prioritized new numeration)
//  INPUT indexing (dim_in[], base_in[]: prioritized old numeration):
    for(i=0;i<dim_num;i++){tmp0[i]=dim_in[i];} //save input dimension extents (old numeration)
    j=0; for(i=0;i<minor_in;i++){pri[j++]=i;} //input minor index set (old numeration)
    for(i=0;i<minor_out;i++){if(n2o[i]>=minor_in) pri[j++]=n2o[i];} //compl.output minor idex set (old numeration)
    for(i=j;i<dim_num;i++){pri[i]=n2o[pri[i]];} //external index set (just convert new numbers to old ones for consistency)
    j=1; for(i=0;i<dim_num;i++){dim_in[i]=j; j*=tmp0[i];} //input bases (old numeration)
    for(i=0;i<dim_num;i++){base_in[i]=dim_in[pri[i]];} //input bases (prioritized old numeration)
    for(i=0;i<dim_num;i++){dim_in[i]=tmp0[pri[i]];} //input extents (prioritized old numeration)
    for(i=0;i<dim_num;i++){if(pri[i]==s1_ind){_s1=i;}else if(pri[i]==s2_ind){_s2=i;}} //split indices (prioritized old numeration)
    s1_ind=_s1; s2_ind=_s2;
    ns1=1+(s1_dim-1)/s1_step; //number of segments from the 1st split minor index
    ns2=1+(s2_dim-1)/s2_step; //number of segments from the 2nd split minor index
//  Index position correspondence for the minor index set (pri-new --> pri-old):
    j=0; for(i=0;i<minor_out;i++){if(n2o[i]<minor_in){pri[i]=n2o[i];}else{pri[i]=(minor_in+j); j++;}}
    j=0; for(i=0;i<minor_in;i++){if(o2n[i]<minor_out){pri[o2n[i]]=i;}else{pri[minor_out+j]=i; j++;}}
// Check tensor transpose configuration parameters:
    if(minor <= 0 || minor_in <= 0 || minor_out <= 0 || _vol <= 0 || _vol_minor <= 0) err_code+=5000; //trap
    if(s1_ind >= dim_num || s2_ind >= dim_num || s1_ond >= dim_num || s2_ond >= dim_num ||
       s1_ind == s2_ind || s1_ond == s2_ond || s1_step <= 0 || s2_step <= 0) err_code+=1000; //trap
    if((s1_step != dim_in[s1_ind] && s1_ind != minor_in-1 && s1_ond != minor_out-1) ||
       (s2_step != dim_in[s2_ind] && s2_ind != minor_in-1 && s2_ond != minor_out-1)) err_code+=500; //trap
    if((_vol_minor*s1_step*s2_step)/(s1_dim*s2_dim) > TENS_TRANSP_BUF_SIZE) err_code+=100; //trap
   } //endif: non-trivial permutation
  }else{
   err_code=1+2*blockDim.x%warpSize;
  }
 } //endif: Master thread.
#ifdef DEBUG_GPU
//DEBUG RECORD begin:
 if(blockIdx.x == 0 && threadIdx.x == 0){
  j=0; gpu_debug_dump[j++]=dim_num;
  for(i=0;i<dim_num;i++) gpu_debug_dump[j++]=const_args_dims[const_args_pos][i];
  for(i=0;i<dim_num;i++) gpu_debug_dump[j++]=const_args_prmn[const_args_pos][i];
  for(i=0;i<dim_num;i++) gpu_debug_dump[j++]=base_in[i];
  for(i=0;i<dim_num;i++) gpu_debug_dump[j++]=base_out[i];
  gpu_debug_dump[j++]=vol; gpu_debug_dump[j++]=vol_ext; gpu_debug_dump[j++]=vol/vol_ext;
  gpu_debug_dump[j++]=minor; gpu_debug_dump[j++]=minor_in; gpu_debug_dump[j++]=minor_out;
  gpu_debug_dump[j++]=s1_ind; gpu_debug_dump[j++]=s1_ond; gpu_debug_dump[j++]=s1_step; gpu_debug_dump[j++]=s1_dim;
  gpu_debug_dump[j++]=s2_ind; gpu_debug_dump[j++]=s2_ond; gpu_debug_dump[j++]=s2_step; gpu_debug_dump[j++]=s2_dim;
  for(i=0;i<dim_num;i++) gpu_debug_dump[j++]=pri[i];
  gpu_debug_dump[j++]=err_code; gpu_debug_dump[j++]=-1;
 }
//DEBUG RECORD end.
#endif /*DEBUG_GPU*/
 __syncthreads();

//Proceed:
 if(err_code == 0){
  if(s1_ind > dim_num){ //tag of a trivial permutation
// Direct copy:
   _vol=vol; j=gridDim.x*blockDim.x; i=blockIdx.x*blockDim.x+threadIdx.x; _addr_in=_vol-_vol%j;
   for(_addr=0;_addr<_addr_in;_addr+=j){
    _addr_out=_addr+i; tens_out[_addr_out]=tens_in[_addr_out];
   }
   _addr_out=_addr_in+i; if(_addr_out<_vol) tens_out[_addr_out]=tens_in[_addr_out];
  }else{ //non-trivial permutation
   l=threadIdx.x/warpSize; //l: warp number
// Distribute work accross CUDA blocks (external multi-index + splitting):
   for(_work_piece=blockIdx.x;_work_piece<vol_ext*ns1*ns2;_work_piece+=gridDim.x){ //(ns1*ns2*vol_ext) is the total number of independent tasks
    _addr=_work_piece; _addr/=vol_ext; _vol=_work_piece-_addr*vol_ext; _s2=(int)(_addr/ns1); _s1=(int)(_addr-_s2*ns1); //{_addr_ext,_s1,_s2} --> tensor subblock (CUDA block)
//  Modify dimension extents due to possible dimension splitting:
    if(threadIdx.x == 0){
     if(_s1+1 == ns1){ //last segment of the 1st split index
      j=s1_dim-_s1*s1_step; dim_in[s1_ind]=j; dim_out[s1_ond]=j;
     }else{ //internal segment of the 1st split index
      dim_in[s1_ind]=s1_step; dim_out[s1_ond]=s1_step;
     }
     if(_s2+1 == ns2){ //last segment of the 2nd split index
      j=s2_dim-_s2*s2_step; dim_in[s2_ind]=j; dim_out[s2_ond]=j;
     }else{ //internal segment of the 2nd split index
      dim_in[s2_ind]=s2_step; dim_out[s2_ond]=s2_step;
     }
     j=1; for(i=0;i<minor;i++){tmp0[i]=j; j*=dim_in[i];} //minor buffer bases (pri-old)
     for(i=0;i<minor;i++) n2o[i]=tmp0[pri[i]]; //look up table to accelerate further accesses to tmp0[]
    }
    __syncthreads();
//  Mount input/output volumes and bases:
    _vol_in=dim_in[0]; for(i=1;i<minor_in;i++){_vol_in*=dim_in[i];}
    _vol_out=dim_out[0]; for(i=1;i<minor_out;i++){_vol_out*=dim_out[i];}
    _vol_minor=_vol_out; for(i=minor_out;i<minor;i++){_vol_minor*=dim_out[i];}
    _addr_in=(_s1*s1_step)*base_in[s1_ind]+(_s2*s2_step)*base_in[s2_ind]; _addr_out=_vol;
    for(i=minor;i<dim_num;i++){_addr=_vol/dim_in[i]; _addr_in+=(_vol-_addr*dim_in[i])*base_in[i]; _vol=_addr;}
    _vol=_addr_out; _addr_out=(_s1*s1_step)*base_out[s1_ond]+(_s2*s2_step)*base_out[s2_ond];
    for(i=minor;i<dim_num;i++){_addr=_vol/dim_out[i]; _addr_out+=(_vol-_addr*dim_out[i])*base_out[i]; _vol=_addr;}
    if(_vol_out > TENS_TRANSP_TAB_SIZE || _vol_minor > _vol_in*TENS_TRANSP_TAB_SIZE ||
       _vol_minor > _vol_out*TENS_TRANSP_TAB_SIZE){
//  Algorithm 0 (slower):
//   Read the minor volume into the buffer from the input tensor block:
     _vol_minor/=_vol_in; //vol_in_c
     _s1=1+(_vol_in-1)/warpSize; //number of warps (lines) which fully cover the input volume
     _s2=blockDim.x/warpSize; //number of whole warps in a thread block (each warp treats one line)
     for(j=l;j<_s1*_vol_minor;j+=_s2){ //j: Line number
      m=j/_s1; _addr=_addr_in; n=m; //n: Input column number (in_c)
      for(i=minor_in;i<minor;i++){k=m/dim_in[i]; _addr+=(m-k*dim_in[i])*base_in[i]; m=k;}
//    m=(j%_s1)*warpSize+threadIdx.x%warpSize; //elemental offset in the input volume
      m=threadIdx.x+(j-n*_s1-l)*warpSize; //elemental offset in the input volume (alternative)
      if(m < _vol_in){buf0[n*_vol_in+m]=tens_in[_addr+m];}
     }
     __syncthreads();
//   Write the minor volume from the buffer into the output tensor block:
     _vol_minor=(_vol_minor*_vol_in)/_vol_out; //vol_out_c
     _s1=1+(_vol_out-1)/warpSize; //number of warps (lines) which fully cover the output volume
     for(j=l;j<_s1*_vol_minor;j+=_s2){ //j: Line number
      n=j/_s1; _addr=_addr_out; _vol=n; _vol_in=0; //_vol: Output column number (out_c)
//    for(i=minor_out;i<minor;i++){m=n%dim_out[i]; n/=dim_out[i]; _addr+=m*base_out[i]; _vol_in+=m*tmp0[pri[i]];}
      for(i=minor_out;i<minor;i++){k=n/dim_out[i]; m=n-k*dim_out[i]; n=k; _addr+=m*base_out[i]; _vol_in+=m*n2o[i];}
//    m=(j%_s1)*warpSize+threadIdx.x%warpSize; //elemental offset in the output volume
      m=threadIdx.x+(j-(int)_vol*_s1-l)*warpSize; //elemental offset in the output volume (alternative)
      if(m < _vol_out){
       _addr+=m;
//     for(i=0;i<minor_out;i++){_vol_in+=(m%dim_out[i])*tmp0[pri[i]]; m/=dim_out[i];}
       for(i=0;i<minor_out;i++){k=m/dim_out[i]; _vol_in+=(m-k*dim_out[i])*n2o[i]; m=k;}
       tens_out[_addr]=buf0[_vol_in];
      }
     }
     __syncthreads();
    }else{
//  Algorithm 1 (presumably faster):
//   Create per-block look-up tables:
     m=_vol_minor/_vol_in; //vol_in_c
     for(j=threadIdx.x;j<m;j+=blockDim.x){ //column number (input)
      _addr=0; _s1=j;
//    for(i=minor_in;i<minor;i++){_addr+=(_s1%dim_in[i])*base_in[i]; _s1/=dim_in[i];}
      for(i=minor_in;i<minor;i++){_s2=_s1/dim_in[i]; _addr+=(_s1-_s2*dim_in[i])*base_in[i]; _s1=_s2;}
      ftb[j]=_addr;
     }
     m=_vol_minor/_vol_out; //vol_out_c
     for(j=threadIdx.x;j<m;j+=blockDim.x){ //column number (output)
      _addr=0; _s1=j;
//    for(i=minor_out;i<minor;i++){_addr+=(_s1%dim_out[i])*base_out[i]; _s1/=dim_out[i];}
      for(i=minor_out;i<minor;i++){_s2=_s1/dim_out[i]; _addr+=(_s1-_s2*dim_out[i])*base_out[i]; _s1=_s2;}
      gtb[j]=_addr;
     }
     for(j=threadIdx.x;j<m;j+=blockDim.x){ //column number (output)
      n=0; _s1=j;
//    for(i=minor_out;i<minor;i++){n+=(_s1%dim_out[i])*n2o[i]; _s1/=dim_out[i];}
      for(i=minor_out;i<minor;i++){_s2=_s1/dim_out[i]; n+=(_s1-_s2*dim_out[i])*n2o[i]; _s1=_s2;}
      htb[j]=n;
     }
     for(j=threadIdx.x;j<_vol_out;j+=blockDim.x){
      n=0; _s1=j;
//    for(i=0;i<minor_out;i++){n+=(_s1%dim_out[i])*n2o[i]; _s1/=dim_out[i];}
      for(i=0;i<minor_out;i++){_s2=_s1/dim_out[i]; n+=(_s1-_s2*dim_out[i])*n2o[i]; _s1=_s2;}
      stb[j]=n;
     }
     __syncthreads();
//   Read the minor volume into the buffer from the input tensor block:
     _vol_minor/=_vol_in; //vol_in_c
     _s1=1+(_vol_in-1)/warpSize; //number of warps (lines) which fully cover the input volume
     _s2=blockDim.x/warpSize; //number of whole warps in a thread block (each warp treats one line)
     for(j=l;j<_s1*_vol_minor;j+=_s2){ //j: Line number
      m=j/_s1; n=threadIdx.x+(j-m*_s1-l)*warpSize; //m: Input column number (in_c); n: Offset in the column
      if(n < _vol_in){_addr=_addr_in+ftb[m]+n; buf0[m*_vol_in+n]=tens_in[_addr];}
     }
     __syncthreads();
//   Write the minor volume from the buffer into the output tensor block:
     _vol_minor=(_vol_minor*_vol_in)/_vol_out; //vol_out_c
     _s1=1+(_vol_out-1)/warpSize; //number of warps (lines) which fully cover the output volume
     for(j=l;j<_s1*_vol_minor;j+=_s2){ //j: Line number
      m=j/_s1; n=threadIdx.x+(j-m*_s1-l)*warpSize; //m: Output column number (out_c); n: Offset in the column
      if(n < _vol_out){_addr=_addr_out+gtb[m]+n; _vol_in=htb[m]+stb[n]; tens_out[_addr]=buf0[_vol_in];}
     }
     __syncthreads();
    }
   } //enddo _work_piece: independent work distribution among thread blocks
  }
 }
//Record errors if occured (for each block):
 if(threadIdx.x == 0){if(err_code != 0) i=atomicAdd(&gpu_error_count,1);}
 return;
}
//------------------------------------------------------------------------------------------------------------
// TENSOR TRANSPOSE (brute-force scatter version):
template <typename T>
__global__ void gpu_tensor_block_copy_scatter_dlf__(int dmo, int drc, int dim_num, int const_args_pos,
                                                    const T * __restrict__ tens_in, T * __restrict__ tens_out)
/**
Scattering version of tensor transpose: tens_out=TRN(tens_in):
INPUT:
 # dmo - dimension extents order (0: normal, as it is in <const_args>; not 0: permuted dimension order will be imposed);
 # drc - index permutation direction (0: normal, as it is in <const_args>; not 0: inversed permutation will be used);
 # dim_num - tensor block rank;
 # const_args_pos - entry in the __constant__ memory bank where tensor block dimension extents (const_args_dims)
                    and index permutation (const_args_prmn) are stored;
 # tens_in[0:] - input tensor;
OUTPUT:
 # tens_out[0:] - output (transposed) tensor;
**/
{
 __shared__ int n2o[MAX_TENSOR_RANK];
 __shared__ size_t vol,base_in[MAX_TENSOR_RANK],base_out[MAX_TENSOR_RANK];
 int i,j,k;
 size_t _vol,_addr_in,_addr_out,_si;

 if(dim_num == 0){
  if(blockIdx.x == 0 && threadIdx.x == 0) tens_out[0]=tens_in[0];
 }else if(dim_num == 1){
  _vol=const_args_dims[const_args_pos][0];
  j=blockIdx.x*blockDim.x+threadIdx.x;
  for(_addr_in=j;_addr_in<_vol;_addr_in+=gridDim.x*blockDim.x){tens_out[_addr_in]=tens_in[_addr_in];}
 }else if(dim_num > 1){
  if(threadIdx.x == 0){
   k=0; for(i=0;i<dim_num;i++){j=const_args_prmn[const_args_pos][i]-1; n2o[j]=i; if(j!=i) k=1;}
   if(k == 0){ //trivial permutation
    n2o[0]=dim_num; //trivial permutation flag
    _vol=1; for(i=0;i<dim_num;i++){_vol*=const_args_dims[const_args_pos][i];}; vol=_vol;
   }else{ //non-trivial permutation
    if(dmo == 0){ //normal dimension order
     _vol=1; for(i=0;i<dim_num;i++){base_in[i]=_vol; _vol*=const_args_dims[const_args_pos][i];}; vol=_vol;
     if(drc == 0){ //normal index permutation
      _vol=1; for(i=0;i<dim_num;i++){k=n2o[i]; base_out[k]=_vol; _vol*=const_args_dims[const_args_pos][k];}
     }else{ //inversed index permutation
      _vol=1; for(i=0;i<dim_num;i++){
       k=const_args_prmn[const_args_pos][i]-1; base_out[k]=_vol; _vol*=const_args_dims[const_args_pos][k];
      }
     }
    }else{ //inversed dimension order
     if(drc == 0){ //normal index permutation
      _vol=1; for(i=0;i<dim_num;i++){
       k=const_args_prmn[const_args_pos][i]-1; base_in[i]=_vol; _vol*=const_args_dims[const_args_pos][k];
      }; vol=_vol;
      _vol=1; for(i=0;i<dim_num;i++){k=n2o[i]; base_out[k]=_vol; _vol*=const_args_dims[const_args_pos][i];}
     }else{ //inversed index permutation
      _vol=1; for(i=0;i<dim_num;i++){
       k=n2o[i]; base_in[i]=_vol; _vol*=const_args_dims[const_args_pos][k];
      }; vol=_vol;
      _vol=1; for(i=0;i<dim_num;i++){
       k=const_args_prmn[const_args_pos][i]-1; base_out[k]=_vol; _vol*=const_args_dims[const_args_pos][i];
      }
     }
    }
   }
  }
#ifdef DEBUG_GPU
//DEBUG RECORD begin:
  if(blockIdx.x == 0 && threadIdx.x == 0){
   j=0; gpu_debug_dump[j++]=dim_num;
   for(i=0;i<dim_num;i++) gpu_debug_dump[j++]=const_args_dims[const_args_pos][i];
   for(i=0;i<dim_num;i++) gpu_debug_dump[j++]=const_args_prmn[const_args_pos][i];
   for(i=0;i<dim_num;i++) gpu_debug_dump[j++]=base_in[i];
   for(i=0;i<dim_num;i++) gpu_debug_dump[j++]=base_out[i];
   gpu_debug_dump[j++]=vol; gpu_debug_dump[j++]=-1;
  }
//DEBUG RECORD end.
#endif /*DEBUG_GPU*/
  __syncthreads();
  _vol=vol;
  if(n2o[0] >= dim_num){ //trivial permutation
   k=gridDim.x*blockDim.x; j=blockIdx.x*blockDim.x+threadIdx.x;
   for(_addr_in=j;_addr_in<_vol;_addr_in+=k){tens_out[_addr_in]=tens_in[_addr_in];}
  }else{ //non-trivial permutation
   j=blockIdx.x*blockDim.x+threadIdx.x;
   for(_addr_in=j;_addr_in<_vol;_addr_in+=gridDim.x*blockDim.x){
    _addr_out=0; _si=_addr_in; for(i=dim_num-1;i>=0;i--){_addr_out+=(_si/base_in[i])*base_out[i]; _si%=base_in[i];}
    tens_out[_addr_out]=tens_in[_addr_in];
   }
  }
 }else{ //dim_num < 0
  if(threadIdx.x == 0) i=atomicAdd(&gpu_error_count,1); //record an error (for each thread block)
 }
 return;
}
//--------------------------------------------------------------------------------------------------------------------------
// MATRIX MULTIPLICATION (slow):
template <typename T>
__global__ void gpu_matrix_multiply_tn__(size_t ll, size_t lr, size_t lc, const T * arg1, const T * arg2, T * arg0, T alpha)
/** arg0(0:ll-1,0:lr-1)+=arg1(0:lc-1,0:ll-1)*arg2(0:lc-1,0:lr-1)*alpha
NOTES:
 # Thread block dimensions (.x and .y) must be equal to MAT_MULT_TILE_DIM(X,Y), respectively.
**/
{
 __shared__ T buf1[MAT_MULT_TILE_DIMX+1][MAT_MULT_TILE_DIMX+1],buf2[MAT_MULT_TILE_DIMY+1][MAT_MULT_TILE_DIMX+1];
 size_t k,_col,_row,_col_base,_row_base;
 int i,j,l,m;
 T _val;

 if(lc > 0 && ll > 0 && lr > 0 && blockDim.x == MAT_MULT_TILE_DIMX && blockDim.y == MAT_MULT_TILE_DIMY){
  _val=static_cast<T>(0.0); j=threadIdx.y; i=threadIdx.x;
  _col_base=blockIdx.y*MAT_MULT_TILE_DIMY;
  while(_col_base < lr){
   _row_base=blockIdx.x*MAT_MULT_TILE_DIMX;
   while(_row_base < ll){
    for(k=0;k<lc;k+=MAT_MULT_TILE_DIMX){
     _col=_col_base+j; _row=_row_base+j;
// Load two blocks into shared memory:
     if(k+MAT_MULT_TILE_DIMX > lc){m=lc-k;}else{m=MAT_MULT_TILE_DIMX;}
     if(i < m){ //(k+i)<lc
      for(l=0;l<MAT_MULT_TILE_DIMX;l+=MAT_MULT_TILE_DIMY){
       if(_row < ll){buf1[l+j][i]=arg1[_row*lc+(k+i)]*alpha;} // Load a block of the 1st argument into the shared memory
       _row+=MAT_MULT_TILE_DIMY;
      }
      if(_col < lr){buf2[j][i]=arg2[_col*lc+(k+i)];} // Load a block of the 2nd argument into the shared memory
     }
     __syncthreads();
// Multiply the two blocks:
     _row=_row_base+i;
     if(_col < lr){
      if(_row < ll){
       _col=_col*ll+_row;
       for(l=0;l<m;l++){_val+=buf1[i][l]*buf2[j][l];}
       arg0[_col]+=_val; _val=static_cast<T>(0.0);
      }
     }
     __syncthreads();
    }
    _row_base+=gridDim.x*MAT_MULT_TILE_DIMX;
   }
   _col_base+=gridDim.y*MAT_MULT_TILE_DIMY;
  }
 }else{
  if(threadIdx.x == 0 && threadIdx.y == 0) i=atomicAdd(&gpu_error_count,1); //record an error (for each thread block)
 }
 return;
}
#endif //NO_GPU

//GENERIC:
int tens_valid_data_kind(int datk, int * datk_size)
/** Returns YEP if the data kind <datk> is valid in TAL-SH, NOPE otherwise.
    Optionally, the data kind size can be returned in <datk_size>. **/
{
 int datk_sz=-1;
 int ans=NOPE;
 switch(datk){
  case R4: ans=YEP; datk_sz=sizeof(float); break;    //real float
  case R8: ans=YEP; datk_sz=sizeof(double); break;   //real double
  case C4: ans=YEP; datk_sz=sizeof(float)*2; break;  //complex float
  case C8: ans=YEP; datk_sz=sizeof(double)*2; break; //complex double
  case NO_TYPE: ans=YEP; datk_sz=0; break; //NO_TYPE is a valid data kind
 }
 if(datk_size != NULL) *datk_size=datk_sz;
 return ans;
}

int tens_valid_data_kind_(int datk, int * datk_size) //Fortran binding
{
 return tens_valid_data_kind(datk,datk_size);
}

size_t tens_elem_offset_f(unsigned int num_dim, const unsigned int * dims, const unsigned int * mlndx)
/** Returns the offset of a tensor element specified by its multi-index with Fortran storage layout.
    Each index in the multi-index has lower bound of zero. **/
{
 unsigned int i;
 size_t offset;

 offset=0;
 for(i=num_dim-1;i>0;--i){offset+=mlndx[i]; offset*=dims[i-1];};
 offset+=mlndx[0];
 return offset;
}

void tens_elem_mlndx_f(size_t offset, unsigned int num_dim, const unsigned int * dims, unsigned int * mlndx)
/** Returns the multi-index of a tensor element specified by its offset with Fortran storage layout.
    Each index in the multi-index has lower bound of zero. **/
{
 unsigned int i;
 size_t d;

 for(i=0;i<num_dim;++i){d=offset/dims[i]; mlndx[i]=offset-d*dims[i]; offset=d;};
 return;
}

unsigned int argument_coherence_get_value(unsigned int coh_ctrl, unsigned int tot_args, unsigned int arg_num)
/** Given a composite coherence control value, returns an individual component.
    No argument consistency check (0 <= arg_num < tot_args). **/
{
 const unsigned int TWO_BITS_SET = 3;
 unsigned int coh = ((coh_ctrl>>((tot_args-(arg_num+1))*2))&(TWO_BITS_SET));
 return coh;
}

int argument_coherence_set_value(unsigned int * coh_ctrl, unsigned int tot_args, unsigned int arg_num, unsigned int coh_val)
/** Sets the coherence value for a specific argument in a composite coherence control value. **/
{
 if(arg_num < tot_args){
  const unsigned int TWO_BITS_SET = 3;
  if((coh_val&(~TWO_BITS_SET)) == 0){
   const unsigned int clear_mask = ((TWO_BITS_SET)<<((tot_args-(arg_num+1))*2));
   const unsigned int set_mask = ((coh_val)<<((tot_args-(arg_num+1))*2));
   const unsigned int coh = (((*coh_ctrl)&(~clear_mask))|set_mask);
   *coh_ctrl=coh;
  }else{
   return 2;
  }
 }else{
  return 1;
 }
 return 0;
}

//DEVICE ID CONVERSION:
int valid_device_kind(int dev_kind)
/** Returns YEP if <dev_kind> is a valid device kind, inlcluding DEV_NULL. NOPE otherwise. **/
{
 if(dev_kind == DEV_NULL ||
    dev_kind == DEV_HOST ||
    dev_kind == DEV_NVIDIA_GPU ||
    dev_kind == DEV_INTEL_MIC ||
    dev_kind == DEV_AMD_GPU) return YEP;
 return NOPE;
}

int encode_device_id(int dev_kind, int dev_num)
/** Given a device ID <dev_num> within its kind <dev_kind>, returns the flat device ID.
    DEV_MAX value on return means that the arguments were invalid. **/
{
 int dev_id=DEV_MAX; //Return of this value (= outside devices range) will mean that the arguments were invalid
 switch(dev_kind){
  case DEV_HOST: if(dev_num == 0) dev_id=0; break;
  case DEV_NVIDIA_GPU: if(dev_num >= 0 && dev_num < MAX_GPUS_PER_NODE) dev_id=1+dev_num; break;
  case DEV_INTEL_MIC: if(dev_num >= 0 && dev_num < MAX_MICS_PER_NODE) dev_id=1+MAX_GPUS_PER_NODE+dev_num; break;
  case DEV_AMD_GPU: if(dev_num >= 0 && dev_num < MAX_AMDS_PER_NODE) dev_id=1+MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+dev_num; break;
  default: dev_id=DEV_MAX; //unknown device kind
 }
 return dev_id;
}

int decode_device_id(int dev_id, int * dev_kind)
/** Given a flat device ID <dev_id>, returns the device kind <dev_kind> (optional)
    and the kind-specific device ID (>=0) as the return value.
    A negative return status (DEV_NULL) indicates an invalid <dev_id>. **/
{
 int dvn,dvid;

 dvn=DEV_NULL; //negative return value will correspond to an invalid <dev_id>
 if(dev_kind != NULL) *dev_kind=DEV_NULL;
 dvid=abs(dev_id); //flat device id is defined up to a sign
 if(dvid == 0){ //Host
  if(dev_kind != NULL) *dev_kind=DEV_HOST;
  dvn=0;
 }else if(dvid >= 1 && dvid <= MAX_GPUS_PER_NODE){ //Nvidia GPU
  if(dev_kind != NULL) *dev_kind=DEV_NVIDIA_GPU;
  dvn=dvid-1;
 }else if(dvid >= 1+MAX_GPUS_PER_NODE && dvid <= MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE){ //Intel MIC
  if(dev_kind != NULL) *dev_kind=DEV_INTEL_MIC;
  dvn=dvid-1-MAX_GPUS_PER_NODE;
 }else if(dvid >= 1+MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE && dvid <= MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+MAX_AMDS_PER_NODE){ //AMD GPU
  if(dev_kind != NULL) *dev_kind=DEV_AMD_GPU;
  dvn=dvid-1-MAX_GPUS_PER_NODE-MAX_MICS_PER_NODE;
 }
 return dvn; //ID of the device within its kind
}

#ifndef NO_GPU
//GPU DEBUG FUNCTIONS:
__host__ int gpu_get_error_count()
/** Returns the total number of CUDA errors occured on current GPU.
    A negative return status means an error occurred. **/
{
 int i;
 cudaError_t err=cudaMemcpyFromSymbol((void*)&i,gpu_error_count,sizeof(gpu_error_count),0,cudaMemcpyDeviceToHost);
 if(err == cudaSuccess){return i;}else{return -1;}
}

__host__ int gpu_get_debug_dump(int *dump)
/** Returns the debug dump (int array) from current GPU.
    A positive return status is the length of the debug dump.
    A negative return status means an error occurred. **/
{
 cudaError_t err=cudaMemcpyFromSymbol((void*)dump,gpu_debug_dump,sizeof(int)*GPU_DEBUG_DUMP_SIZE,0,cudaMemcpyDeviceToHost);
 if(err == cudaSuccess){return GPU_DEBUG_DUMP_SIZE;}else{return -1;}
}
#endif /*NO_GPU*/

//AUXILIARY FUNCTIONS:
static int prmn_convert(int n, const int *o2n, int *n2o)
/** Converts an O2N permutation into N2O (length = n). Both permutations
    are sign-free and the numeration starts from 1. **/
{
 int i,j;
 if(n >= 0){
  for(i=0;i<n;i++){j=o2n[i]-1; if(j >= 0 && j < n){n2o[j]=i+1;}else{return 1;}}
 }else{
  return 2;
 }
 return 0;
}

static int non_trivial_prmn(int n, const int *prm)
/** Returns NOPE if the permutation prm[0:n-1] is trivial, YEP otherwise.
    The permutation is sign-free and the numeration starts from 1. No error check. **/
{
 int i,f=NOPE;
 for(i=0;i<n;i++){if(prm[i] != i+1){f=YEP; break;}}
 return f;
}

//DEVICE RESOURCE MANAGEMENT:
int tensDevRsc_create(talsh_dev_rsc_t **drsc)
/** Creates a new device resource descriptor and inits it to null. **/
{
 int errc = 0;
 *drsc=(talsh_dev_rsc_t*)malloc(sizeof(talsh_dev_rsc_t)); if(*drsc == NULL) return TRY_LATER;
 errc=tensDevRsc_clean(*drsc); errc=0;
 return errc;
}

int tensDevRsc_clean(talsh_dev_rsc_t * drsc)
/** Cleans (initializes to null) a device resource descriptor. **/
{
 if(drsc != NULL){
  drsc->dev_id=DEV_NULL; //flat device id
  drsc->gmem_p=NULL;     //device global memory pointer (any device)
  drsc->buf_entry=-1;    //device argument buffer entry (any device)
  drsc->mem_attached=0;  //memory attachement flag (distinguishes between allocated and attached memory)
 }else{
  return -1;
 }
 return 0;
}

int tensDevRsc_is_empty(talsh_dev_rsc_t * drsc)
/** Returns YEP if the device resource descriptor is empty, NOPE otherwise.
    Negative return status means an error. **/
{
 int errc = 0;
 if(drsc == NULL) return -1;
 if(drsc->dev_id >= 0 && drsc->dev_id < DEV_MAX){if(drsc->gmem_p != NULL) return NOPE;}
 errc=tensDevRsc_clean(drsc); errc=YEP;
 return errc;
}

int tensDevRsc_same(const talsh_dev_rsc_t * drsc0, const talsh_dev_rsc_t * drsc1)
/** Returns YEP if two resource descriptors point to the same resources, NOPE otherwise.
    A negative return status indicates an error. **/
{
 if(drsc0 == NULL) return -1;
 if(drsc1 == NULL) return -2;
 if(drsc0->dev_id == drsc1->dev_id &&
    drsc0->gmem_p == drsc1->gmem_p) return YEP;
 return NOPE;
}

int tensDevRsc_clone(const talsh_dev_rsc_t * drsc_in, talsh_dev_rsc_t * drsc_out)
/** Copy constructor for a device resource. **/
{
 if(drsc_in == NULL) return -1;
 if(drsc_out == NULL) return -2;
 drsc_out->dev_id=drsc_in->dev_id;
 drsc_out->gmem_p=drsc_in->gmem_p;
 drsc_out->buf_entry=drsc_in->buf_entry;
 drsc_out->mem_attached=drsc_in->mem_attached;
 return 0;
}

int tensDevRsc_attach_mem(talsh_dev_rsc_t * drsc, int dev_id, void * mem_p, int buf_entry)
/** Attaches a chunk of existing global memory to a device resource descriptor.
    If <buf_entry> >= 0, that means that the global memory is in the argument buffer.
    If the resource descriptor had already been assigned a device, the <dev_id>
    argument must match that one. **/
{
 if(drsc == NULL) return -1;
 if(dev_id < 0 || dev_id >= DEV_MAX) return -2;
 if(mem_p == NULL) return -3;
 if(drsc->dev_id >= 0 && drsc->dev_id != dev_id) return 1; //a non-empty descriptor must be associated with the same device
 if(drsc->gmem_p != NULL || drsc->buf_entry >= 0) return 2; //resource already has global memory attached
 drsc->dev_id=dev_id; drsc->gmem_p=mem_p; drsc->buf_entry=buf_entry; drsc->mem_attached=1;
 return 0;
}

int tensDevRsc_detach_mem(talsh_dev_rsc_t * drsc)
/** Detaches a chunk of external memory from a device resource descriptor.
    Regardless of the origin, that memory is not released. **/
{
 int errc = 0;
 if(drsc == NULL) return -1;
 if(drsc->dev_id < 0 || drsc->dev_id >= DEV_MAX) return -2; //empty resource descriptor
 if(drsc->gmem_p == NULL || drsc->mem_attached == 0) return 1; //no global memory attached
 drsc->gmem_p=NULL; drsc->buf_entry=-1; drsc->mem_attached=0;
 errc=tensDevRsc_is_empty(drsc); errc=0;
 return errc;
}

int tensDevRsc_allocate_mem(talsh_dev_rsc_t * drsc, int dev_id, size_t mem_size, int in_arg_buf)
/** Allocates global memory on device <dev_id> and attaches it to a device resource descriptor.
    If <in_arg_buf> = YEP, the memory will be allocated via that device's argument buffer.
    A return status TRY_LATER or DEVICE_UNABLE indicates the resource shortage and is not an error. **/
{
 int i,devk,devn,errc;
 char *byte_ptr;

 if(drsc == NULL) return -1;
 if(dev_id < 0 || dev_id >= DEV_MAX) return -2;
 if(mem_size <= 0) return -3;
 devn=decode_device_id(dev_id,&devk); if(devn < 0) return -4; //invalid flat device id
 if(drsc->dev_id >= 0 && drsc->dev_id != dev_id) return 1; //resource was assigned to a different device
 if(drsc->gmem_p != NULL || drsc->buf_entry >= 0) return 2; //resource already has global memory attached
 switch(devk){
  case DEV_HOST:
   if(in_arg_buf == NOPE){
    errc=host_mem_alloc_pin(&(drsc->gmem_p),mem_size); if(errc != 0){drsc->gmem_p = NULL; return 3;}
   }else{
    errc=get_buf_entry_host(mem_size,&byte_ptr,&i);
    if(errc != 0){if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return 4;}}
    drsc->gmem_p=(void*)byte_ptr; drsc->buf_entry=i;
   }
   drsc->mem_attached=0;
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   if(in_arg_buf == NOPE){
    errc=gpu_mem_alloc(&(drsc->gmem_p),mem_size,devn); if(errc != 0){drsc->gmem_p = NULL; return 5;}
   }else{
    errc=get_buf_entry_gpu(devn,mem_size,&byte_ptr,&i);
    if(errc != 0){if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return 6;}}
    drsc->gmem_p=(void*)byte_ptr; drsc->buf_entry=i;
   }
   drsc->mem_attached=0;
   break;
#else
   return -5;
#endif
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   //`Future
   break;
#else
   return -6;
#endif
  case DEV_AMD_GPU:
#ifndef NO_AMD
   //`Future
   break;
#else
   return -7;
#endif
  default:
   return -8; //unknown device kind
 }
 drsc->dev_id=dev_id;
 return 0;
}

int tensDevRsc_free_mem(talsh_dev_rsc_t * drsc)
/** Releases global memory referred to by a device resource descriptor.
    An unsuccessful release of the global memory is marked with
    an error status NOT_CLEAN, but the corresponding components of
    the resource descriptor are cleared anyway. **/
{
 int n,devn,devk,errc;

 n=0;
 if(drsc == NULL) return -1;
 if(drsc->dev_id < 0 || drsc->dev_id >= DEV_MAX) return -2;
 if(drsc->gmem_p == NULL) return -3;
 devn=decode_device_id(drsc->dev_id,&devk); if(devn < 0) return -4; //invalid flat device id
 if(drsc->mem_attached != 0) return 1; //memory was not allocated but attached
 switch(devk){
  case DEV_HOST:
   if(drsc->buf_entry >= 0){
    errc=free_buf_entry_host(drsc->buf_entry);
    if(errc != 0){
     if(VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_free_mem): free_buf_entry_host error %d\n",errc);
     n=NOT_CLEAN;
    }
    drsc->buf_entry=-1;
   }else{
    errc=host_mem_free_pin(drsc->gmem_p);
    if(errc != 0){
     if(VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_free_mem): host_mem_free_pin error %d\n",errc);
     n=NOT_CLEAN;
    }
   }
   drsc->gmem_p=NULL;
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   if(drsc->buf_entry >= 0){
    errc=free_buf_entry_gpu(devn,drsc->buf_entry);
    if(errc != 0){
     if(VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_free_mem): free_buf_entry_gpu error %d\n",errc);
     n=NOT_CLEAN;
    }
    drsc->buf_entry=-1;
   }else{
    errc=gpu_mem_free(drsc->gmem_p,devn);
    if(errc != 0){
     if(VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_free_mem): gpu_mem_free error %d\n",errc);
     n=NOT_CLEAN;
    }
   }
   drsc->gmem_p=NULL;
   break;
#else
   return -5;
#endif
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   //`Future
   break;
#else
   return -6;
#endif
  case DEV_AMD_GPU:
#ifndef NO_AMD
   //`Future
   break;
#else
   return -7;
#endif
  default:
   return -8; //invalid device kind
 }
 errc=tensDevRsc_is_empty(drsc);
 return n;
}

int tensDevRsc_get_gmem_ptr(talsh_dev_rsc_t * drsc, void ** gmem_p)
/** Returns the pointer to global memory (.gmem_p component) of the device resource. **/
{
 if(drsc == NULL) return -1;
 if(tensDevRsc_is_empty(drsc) == YEP) return 1;
 *gmem_p=drsc->gmem_p;
 return 0;
}

int tensDevRsc_device_id(talsh_dev_rsc_t * drsc)
/** Returns the device id of the resource. **/
{return drsc->dev_id;}

int tensDevRsc_release_all(talsh_dev_rsc_t * drsc)
/** Releases all device resources in <drsc>. An unsuccessful release
    of one or more resources is marked with a return status NOT_CLEAN,
    but the corresponding components of the device resource descriptor
    are cleaned anyway. An empty resource causes no action. **/
{
 int n,errc;

 n=0;
 if(drsc == NULL) return -1;
 if(drsc->dev_id >= 0 && drsc->dev_id < DEV_MAX){ //resource handle is not empty
//Release global memory:
  if(drsc->gmem_p != NULL){
   if(drsc->mem_attached){
    errc=tensDevRsc_detach_mem(drsc);
    if(errc){
     if(VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_release_all): tensDevRsc_detach_mem error %d\n",errc);
     n=NOT_CLEAN;
    }
   }else{
    errc=tensDevRsc_free_mem(drsc);
    if(errc){
     if(VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_release_all): tensDevRsc_free_mem error %d\n",errc);
     n=NOT_CLEAN;
    }
   }
  }
 }
 errc=tensDevRsc_clean(drsc);
 if(n != 0 && VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_release_all): Error %d\n",n);
 return n;
}

int tensDevRsc_destroy(talsh_dev_rsc_t * drsc)
/** Completely destroys a device resource descriptor. A return status NOT_CLEAN
    means that certain resources have not been released cleanly,
    but it is not a critical error in general (however, a leak can occur). **/
{
 int n,errc;
 n=0;
 if(drsc == NULL) return -1;
 errc=tensDevRsc_release_all(drsc); if(errc) n=NOT_CLEAN;
 free(drsc);
 return n;
}

#ifndef NO_GPU
__host__ static int cuda_stream_get(int gpu_num, int * cuda_stream_handle)
/** For GPU#gpu_num, returns a usable CUDA stream handle <cuda_stream_handle>.
Non-zero return status means an error, except the return status TRY_LATER means
no free resources are currently available (not an error). **/
{
 *cuda_stream_handle=-1;
 if(gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE){
  if(gpu_is_mine(gpu_num) > GPU_OFF){
   if(CUDAStreamFFE[gpu_num] > 0){ //number of free handles left on GPU#gpu_num
    *cuda_stream_handle=CUDAStreamFreeHandle[gpu_num][--CUDAStreamFFE[gpu_num]];
    if(*cuda_stream_handle < 0 || *cuda_stream_handle >= MAX_CUDA_TASKS){
     *cuda_stream_handle=-1; return 3; //invalid handle: corruption
    }
   }else{
    return TRY_LATER; //all handles are currently busy
   }
  }else{
   return 2;
  }
 }else{
  return 1;
 }
 return 0;
}

__host__ static int cuda_stream_release(int gpu_num, int cuda_stream_handle)
/** For GPU#gpu_num, releases a CUDA stream handle <cuda_stream_handle>.
Non-zero return status means an error. **/
{
 if(gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE){
  if(gpu_is_mine(gpu_num) > GPU_OFF){
   if(cuda_stream_handle >= 0 && cuda_stream_handle < MAX_CUDA_TASKS){
    if(CUDAStreamFFE[gpu_num] < 0 || CUDAStreamFFE[gpu_num] > MAX_CUDA_TASKS) return 5; //corrupted
    if(CUDAStreamFFE[gpu_num] < MAX_CUDA_TASKS){
     CUDAStreamFreeHandle[gpu_num][CUDAStreamFFE[gpu_num]++]=cuda_stream_handle;
    }else{
     return 4; //an attempt to release a non-existing handle
    }
   }else{
    return 3;
   }
  }else{
   return 2;
  }
 }else{
  return 1;
 }
 return 0;
}

__host__ static cudaStream_t * cuda_stream_ptr(int gpu_num, int cuda_stream_handle)
{
/** Returns a pointer to a valid CUDA stream handle. **/
 if(gpu_num < 0 || gpu_num >= MAX_GPUS_PER_NODE) return NULL;
 if(cuda_stream_handle < 0 || cuda_stream_handle >= MAX_CUDA_TASKS) return NULL;
 if(gpu_is_mine(gpu_num) > GPU_OFF) return &(CUDAStreamBank[gpu_num][cuda_stream_handle]);
 return NULL;
}

__host__ static int cuda_event_get(int gpu_num, int * cuda_event_handle)
/** For GPU#gpu_num, returns a usable CUDA event handle <cuda_event_handle>.
Non-zero return status means an error, except the return status TRY_LATER means
no free resources are currently available (not an error). **/
{
 *cuda_event_handle=-1;
 if(gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE){
  if(gpu_is_mine(gpu_num) > GPU_OFF){
   if(CUDAEventFFE[gpu_num] > 0){ //number of free handles left on GPU#gpu_num
    *cuda_event_handle=CUDAEventFreeHandle[gpu_num][--CUDAEventFFE[gpu_num]];
    if(*cuda_event_handle < 0 || *cuda_event_handle >= MAX_CUDA_EVENTS){
     *cuda_event_handle=-1; return 3; //invalid handle: corruption
    }
   }else{
    return TRY_LATER; //all handles are currently busy
   }
  }else{
   return 2;
  }
 }else{
  return 1;
 }
 return 0;
}

__host__ static int cuda_event_release(int gpu_num, int cuda_event_handle)
/** For GPU#gpu_num, releases a CUDA event handle <cuda_event_handle>.
Non-zero return status means an error. **/
{
 if(gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE){
  if(gpu_is_mine(gpu_num) > GPU_OFF){
   if(cuda_event_handle >= 0 && cuda_event_handle < MAX_CUDA_EVENTS){
    if(CUDAEventFFE[gpu_num] < 0 || CUDAEventFFE[gpu_num] > MAX_CUDA_EVENTS) return 5; //corrupted
    if(CUDAEventFFE[gpu_num] < MAX_CUDA_EVENTS){
     CUDAEventFreeHandle[gpu_num][CUDAEventFFE[gpu_num]++]=cuda_event_handle;
    }else{
     return 4; //an attempt to release a non-existing handle
    }
   }else{
    return 3;
   }
  }else{
   return 2;
  }
 }else{
  return 1;
 }
 return 0;
}

__host__ static cudaEvent_t * cuda_event_ptr(int gpu_num, int cuda_event_handle)
{
/** Returns a pointer to a valid CUDA event handle. **/
 if(gpu_num < 0 || gpu_num >= MAX_GPUS_PER_NODE) return NULL;
 if(cuda_event_handle < 0 || cuda_event_handle >= MAX_CUDA_EVENTS) return NULL;
 if(gpu_is_mine(gpu_num) > GPU_OFF) return &(CUDAEventBank[gpu_num][cuda_event_handle]);
 return NULL;
}

__host__ static void limit_cuda_blocks2d(int max_blocks, int *bx, int *by)
/** Limits the number of CUDA blocks in a 2d grid to <max_blocks>.
    No argument validity check! **/
{
 if(max_blocks > 1){
  double rdc = ((double)max_blocks)/(((double)(*bx))*((double)(*by)));
  if(rdc < 1.0){
   rdc=sqrt(rdc);
   if(*bx > *by){
    *by=(int)(rdc*((double)(*by))); if(*by < 1){*by=1; *bx=max_blocks; return;}
    *bx=(int)(rdc*((double)(*bx)));
   }else{
    *bx=(int)(rdc*((double)(*bx))); if(*bx < 1){*bx=1; *by=max_blocks; return;}
    *by=(int)(rdc*((double)(*by)));
   }
   if((*bx)*(*by) > max_blocks){
    if(*bx > *by){(*bx)--;}else{(*by)--;}
   }
  }
 }else{
  *bx=1; *by=1;
 }
 return;
}

__host__ static int tens_op_best_gpu(const tensBlck_t *tens0, const tensBlck_t *tens1, const tensBlck_t *tens2)
/** Returns the optimal GPU for a given set of tensor arguments (from the data locality point of view).
    A negative return status means an error. All arguments are optional. **/
{
 int gpu,dev_kind,gpu0,gpu1,gpu2,s0,s1,s2;

 gpu=-1;
 if(tens0 != NULL){
  if(tens0->src_rsc == NULL) return -1;
  gpu0=decode_device_id((tens0->src_rsc)->dev_id,&dev_kind);
  if(dev_kind != DEV_NVIDIA_GPU) gpu0=-1;
  if(tens1 != NULL){
   if(tens1->src_rsc == NULL) return -1;
   gpu1=decode_device_id((tens1->src_rsc)->dev_id,&dev_kind);
   if(dev_kind != DEV_NVIDIA_GPU) gpu1=-1;
   if(gpu1 >= 0 && gpu1 == gpu0){
    gpu=gpu1;
   }else{
    if(tens2 != NULL){
     if(tens2->src_rsc == NULL) return -1;
     gpu2=decode_device_id((tens2->src_rsc)->dev_id,&dev_kind);
     if(dev_kind != DEV_NVIDIA_GPU) gpu2=-1;
     if(gpu2 >= 0 && (gpu2 == gpu1 || gpu2 == gpu0)){
      gpu=gpu2;
     }else{
      s0=0; s1=0; s2=0;
      if(gpu0 >= 0) s0=gpu_stats[gpu0].tasks_submitted-(gpu_stats[gpu0].tasks_completed+gpu_stats[gpu0].tasks_deferred+gpu_stats[gpu0].tasks_failed);
      if(gpu1 >= 0) s1=gpu_stats[gpu1].tasks_submitted-(gpu_stats[gpu1].tasks_completed+gpu_stats[gpu1].tasks_deferred+gpu_stats[gpu1].tasks_failed);
      if(gpu2 >= 0) s2=gpu_stats[gpu2].tasks_submitted-(gpu_stats[gpu2].tasks_completed+gpu_stats[gpu2].tasks_deferred+gpu_stats[gpu2].tasks_failed);
      if(gpu0 >= 0 && (gpu1 < 0 || s0 <= s1) && (gpu2 < 0 || s0 <= s2)){
       gpu=gpu0;
      }else if(gpu1 >= 0 && (gpu0 < 0 || s1 <= s0) && (gpu2 < 0 || s1 <= s2)){
       gpu=gpu1;
      }else if(gpu2 >= 0 && (gpu1 < 0 || s2 <= s1) && (gpu0 < 0 || s2 <= s0)){
       gpu=gpu2;
      }
     }
    }else{
     s0=0; s1=0;
     if(gpu0 >= 0) s0=gpu_stats[gpu0].tasks_submitted-(gpu_stats[gpu0].tasks_completed+gpu_stats[gpu0].tasks_deferred+gpu_stats[gpu0].tasks_failed);
     if(gpu1 >= 0) s1=gpu_stats[gpu1].tasks_submitted-(gpu_stats[gpu1].tasks_completed+gpu_stats[gpu1].tasks_deferred+gpu_stats[gpu1].tasks_failed);
     if(gpu0 >= 0 && (gpu1 < 0 || s0 <= s1)){
      gpu=gpu0;
     }else if(gpu1 >= 0 && (gpu0 < 0 || s1 <= s0)){
      gpu=gpu1;
     }
    }
   }
  }else{
   gpu=gpu0;
  }
 }
 if(gpu < 0 || gpu >= MAX_GPUS_PER_NODE) gpu=gpu_busy_least();
 if(gpu_is_mine(gpu) <= GPU_OFF) gpu=-1; //for safety
 return gpu;
}

//NV-TAL INITIALIZATION/SHUTDOWN (internal use only):
__host__ int init_gpus(int gpu_beg, int gpu_end)
/** Initializes all GPU contexts for the current MPI process. Returned positive value is
the number of initialized GPUs. A negative return status means an error occured.
Each enabled GPU from the range [gpu_beg:gpu_end] will obtain its own cublasHandle as well.
The first GPU from the given range will be left active at the end. If <gpu_beg> > <gpu_end>,
no GPU will be initialized. **/
{
 int i,j,n,errc;
 void * base_ptr;
 cudaError_t err;
#ifndef NO_BLAS
 cublasStatus_t err_cublas;
#endif

 n=0; for(i=0;i<MAX_GPUS_PER_NODE;i++) gpu_up[i]=GPU_OFF; //initial GPU status
 if(gpu_beg >= 0 && gpu_end >= gpu_beg){
  err=cudaGetDeviceCount(&i); if(err != cudaSuccess) return -1;
  if(gpu_end >= MAX_GPUS_PER_NODE || gpu_end >= i) return -2;
//Initialize a mapped bank for tensor operation prefactors for GPU usage:
  errc=slab_clean(&prefactors); if(errc != 0) return -3;
  errc=slab_construct(&prefactors,sizeof(talshComplex8),(size_t)(MAX_GPUS_PER_NODE*MAX_CUDA_TASKS),sizeof(talshComplex8),1U); if(errc != 0) return -4;
  errc=slab_get_base_ptr(&prefactors,&base_ptr); if(errc != 0) return -5;
  err=cudaHostGetDevicePointer(&gpu_prefs_base_ptr,base_ptr,0); if(err != cudaSuccess) return -6;
//Initialize each GPU device:
  for(i=gpu_end;i>=gpu_beg;i--){
   err=cudaSetDevice(i);
   if(err == cudaSuccess){
    gpu_up[i]=GPU_MINE; err=cudaGetDeviceProperties(&(gpu_prop[i]),i); if(err != cudaSuccess) gpu_up[i]=GPU_OFF;
    if(gpu_up[i] > GPU_OFF){
//SHMEM width:
     errc=gpu_set_shmem_width(GPU_SHMEM_WIDTH);
     if(errc != 0 && VERBOSE) printf("#WARNING(tensor_algebra_gpu_nvidia:init_gpus): Unable to set GPU SHMEM width %d: Error %d \n",GPU_SHMEM_WIDTH,errc);
//cuBLAS.v2 context:
#ifndef NO_BLAS
     err_cublas=cublasCreate(&(cublas_handle[i]));
     if(err_cublas == CUBLAS_STATUS_SUCCESS){
      gpu_up[i]=GPU_MINE_CUBLAS;
      err_cublas=cublasSetPointerMode(cublas_handle[i],CUBLAS_POINTER_MODE_DEVICE);
      if(err_cublas != CUBLAS_STATUS_SUCCESS) gpu_up[i]=GPU_MINE;
     }
#endif
    }
//CUDA stream bank:
    if(gpu_up[i] > GPU_OFF){
     for(j=0;j<MAX_CUDA_TASKS;j++) CUDAStreamFreeHandle[i][j]=j; CUDAStreamFFE[i]=MAX_CUDA_TASKS;
     for(j=0;j<MAX_CUDA_TASKS;j++){
      err=cudaStreamCreate(&(CUDAStreamBank[i][j])); if(err != cudaSuccess){gpu_up[i]=GPU_OFF; break;};
     }
    }
//CUDA event bank:
    if(gpu_up[i] > GPU_OFF){
     for(j=0;j<MAX_CUDA_EVENTS;j++) CUDAEventFreeHandle[i][j]=j; CUDAEventFFE[i]=MAX_CUDA_EVENTS;
     for(j=0;j<MAX_CUDA_EVENTS;j++){
      err=cudaEventCreate(&(CUDAEventBank[i][j])); if(err != cudaSuccess){gpu_up[i]=GPU_OFF; break;};
     }
    }
//Last task:
    LastTask[i]=NULL;
//Clear GPU statistics:
    gpu_stats[i].tasks_submitted=0;
    gpu_stats[i].tasks_completed=0;
    gpu_stats[i].tasks_deferred=0;
    gpu_stats[i].tasks_failed=0;
    gpu_stats[i].flops=0.0;
    gpu_stats[i].traffic_in=0.0;
    gpu_stats[i].traffic_out=0.0;
    gpu_stats[i].time_active=0.0;
    gpu_stats[i].time_start=clock();
//Accept GPU as ready (active):
    if(gpu_up[i] > GPU_OFF) n++;
   }
  }
//Peer memory access (UVA based):
#ifdef UNIFIED_ADDRESSING
  for(i=gpu_end;i>=gpu_beg;i--){
   if(gpu_up[i] > GPU_OFF){
    if(gpu_prop[i].unifiedAddressing != 0){
     err=cudaSetDevice(i);
     if(err == cudaSuccess){
      for(j=gpu_end;j>=gpu_beg;j--){
       if(j != i && gpu_up[j] > GPU_OFF){
        if(gpu_prop[j].unifiedAddressing != 0){
         err=cudaDeviceEnablePeerAccess(j,0); //device i can access memory of device j
         if((err != cudaSuccess) && VERBOSE) printf("\n#MSG(tensor_algebra_gpu_nvidia): GPU peer no access: %d->%d\n",i,j);
        }else{
         if(VERBOSE) printf("\n#MSG(tensor_algebra_gpu_nvidia): GPU peer no access: %d->%d\n",i,j);
        }
       }
      }
     }else{
      gpu_up[i]=GPU_OFF; n--;
     }
    }
    err=cudaGetLastError(); //clear the GPU#i error status
   }
  }
#endif
 }
 return n; //number of initialized GPU's
}

__host__ int free_gpus(int gpu_beg, int gpu_end)
/** Destroys all GPU/CUBLAS contexts on all GPU devices belonging to the MPI process.
A positive value returned is the number of failed GPUs; a negative one is an error.
If <gpu_beg> > <gpu_end>, nothing wil be done. **/
{
 int i,j,n,failure;
 cudaError_t err;
#ifndef NO_BLAS
 cublasStatus_t err_cublas;
#endif
 failure=0; n=0;
 if(gpu_beg >= 0 && gpu_end >= gpu_beg){
  err=cudaGetDeviceCount(&i); if(err != cudaSuccess) return -1;
  if(gpu_end >= MAX_GPUS_PER_NODE || gpu_end >= i) return -2;
//Free the mapped bank of tensor operation prefactors:
  i=slab_destruct(&prefactors); if(i != 0) failure++;
  gpu_prefs_base_ptr=NULL;
//Free GPU devices:
  for(i=gpu_beg;i<=gpu_end;i++){
   if(gpu_up[i] > GPU_OFF){
    n++; err=cudaSetDevice(i);
    if(err == cudaSuccess){
#ifndef NO_BLAS
     if(gpu_up[i] >= GPU_MINE_CUBLAS){err_cublas=cublasDestroy(cublas_handle[i]); if(err_cublas == CUBLAS_STATUS_SUCCESS) gpu_up[i]=GPU_MINE;}
#endif
//CUDA stream bank:
     if(gpu_up[i] > GPU_OFF){
      for(j=0;j<MAX_CUDA_TASKS;j++) CUDAStreamFreeHandle[i][j]=j; CUDAStreamFFE[i]=MAX_CUDA_TASKS;
      for(j=0;j<MAX_CUDA_TASKS;j++){err=cudaStreamDestroy(CUDAStreamBank[i][j]); if(err != cudaSuccess) failure++;}
     }
//CUDA event bank:
     if(gpu_up[i] > GPU_OFF){
      for(j=0;j<MAX_CUDA_EVENTS;j++) CUDAEventFreeHandle[i][j]=j; CUDAEventFFE[i]=MAX_CUDA_EVENTS;
      for(j=0;j<MAX_CUDA_EVENTS;j++){err=cudaEventDestroy(CUDAEventBank[i][j]); if(err != cudaSuccess) failure++;}
     }
//Last task:
     LastTask[i]=NULL;
     n--; err=cudaDeviceReset();
    }
    gpu_up[i]=GPU_OFF; //GPU is taken out of use regardless of its status!
   }
  }
 }
 if(failure && VERBOSE) printf("#WARNING(tensor_algebra_gpu_nvidia:free_gpus): Resource deallocation was not fully successful!");
 return n;
}

__host__ int gpu_get_device_count(int * dev_count)
/** Returns the total number of NVIDIA GPUs found on the node. **/
{
 const char *err_msg;
 cudaError_t cuda_err = cudaGetDeviceCount(dev_count);
 if(cuda_err != cudaSuccess){
  err_msg=cudaGetErrorString(cuda_err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_get_device_count): %s\n",err_msg);
  *dev_count=-1; return 1;
 }
 return 0;
}

__host__ int gpu_is_mine(int gpu_num)
/** Positive return: GPU is mine; 0: GPU is not mine; -1: invalid <gpu_num>. **/
{if(gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE){return gpu_up[gpu_num];}else{return -1;}}

__host__ int gpu_busy_least()
/** Returns the ID of the least busy GPU (non-negative) or -1 (no GPU found). **/
{
 int i,j,m,n;
 m=-1; n=-1;
 for(i=0;i<MAX_GPUS_PER_NODE;i++){
  if(gpu_up[i] > GPU_OFF){
   j=gpu_stats[i].tasks_submitted-(gpu_stats[i].tasks_completed+gpu_stats[i].tasks_deferred+gpu_stats[i].tasks_failed);
   if(m >= 0){
    if(j < m){m=j; n=i;};
   }else{
    m=j; n=i;
   }
  }
 }
 return n;
}

__host__ int gpu_in_focus(int gpu_num)
/** If <gpu_num> is not passed here, returns the id of the current GPU in focus.
    If <gpu_num> is passed here, returns YEP if it is currently in focus, NOPE otherwise.
    In case of error, returns NVTAL_FAILURE (negative integer). **/
{
 int n;
 cudaError_t err;
 err=cudaGetDevice(&n); if(err != cudaSuccess) return NVTAL_FAILURE;
 if(gpu_num >= 0){if(n == gpu_num){return YEP;}else{return NOPE;}}
 if(n < 0 || n >= MAX_GPUS_PER_NODE) return NVTAL_FAILURE; //GPU id must not exceed the TALSH limit per node
 return n;
}

__host__ int gpu_activate(int gpu_num)
/** If GPU is enabled (mine), does cudaSetDevice; returns non-zero otherwise (error). **/
{
 int cur_gpu;
 cudaError_t err;
 if(gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE){
  if(gpu_up[gpu_num] > GPU_OFF){
   cur_gpu=gpu_in_focus();
   if(cur_gpu != gpu_num){
    err=cudaSetDevice(gpu_num);
    if(err != cudaSuccess){if(cur_gpu >= 0) err=cudaSetDevice(cur_gpu); return 3;}
   }
  }else{
   return 2; //GPU is not mine
  }
 }else{
  return 1; //invalid <gpu_num>
 }
 return 0;
}

__host__ size_t gpu_device_memory_size(int gpu_num)
/** Returns the total memory (bytes) for a given GPU device. **/
{
 size_t bytes;

 bytes=0;
 if(gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE){
  if(gpu_up[gpu_num] > GPU_OFF) bytes=gpu_prop[gpu_num].totalGlobalMem;
 }
 return bytes;
}

//NV-TAL INTERNAL CONTROL:
__host__ int gpu_set_shmem_width(int width){
/** Sets the GPU shared memory bank width:
    <width> = R4: 4 bytes;
    <width> = R8: 8 bytes. **/
 cudaError_t cerr;
 if(width == R8){
  cerr=cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
 }else if(width == R4){
  cerr=cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
 }else{
  return 1; //invalid <width> passed
 }
 if(cerr != cudaSuccess) return 2;
 return 0;
}

__host__ void gpu_set_transpose_algorithm(int alg){
/** Activates either the scatter or the shared-memory based tensor transpose algorithm.
    Invalid <alg> values will activate the basic shared-memory algorithm (default). **/
 if(alg == EFF_TRN_OFF){TRANS_SHMEM=EFF_TRN_OFF;}
#ifdef USE_CUTT
 else if(alg == EFF_TRN_ON_CUTT){TRANS_SHMEM=EFF_TRN_ON_CUTT;}
#endif
 else{TRANS_SHMEM=EFF_TRN_ON;} //any other value will result in the default setting
 return;
}

__host__ void gpu_set_matmult_algorithm(int alg){
/** Activates either cuBLAS (fast) or my own (slow) BLAS CUDA kernels. **/
#ifndef NO_BLAS
 if(alg == BLAS_ON){DISABLE_BLAS=BLAS_ON;}else{DISABLE_BLAS=BLAS_OFF;};
#endif
 return;
}

__host__ int gpu_print_stats(int gpu_num)
/** Prints GPU statistics for GPU#<gpu_num>. If <gpu_num>=-1,
    prints GPU statistics for all active GPUs.
    A negative return status means invalid <gpu_num>. **/
{
 int i,b,f;
 clock_t ctm;

 if(gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE){
  b=gpu_num; f=gpu_num; //select a specific GPU
 }else if(gpu_num == -1){
  b=0; f=MAX_GPUS_PER_NODE-1; //select all GPUs
 }else{
  return -1; //invalid GPU number
 }
 for(i=b;i<=f;i++){
  if(gpu_is_mine(i) != GPU_OFF){
   ctm=clock();
   gpu_stats[i].time_active=((double)(ctm-gpu_stats[i].time_start))/CLOCKS_PER_SEC;
   printf("\n#MSG(TAL-SH::NV-TAL): Statistics on GPU #%d:\n",i);
   printf(" Number of tasks submitted: %llu\n",gpu_stats[i].tasks_submitted);
   printf(" Number of tasks completed: %llu\n",gpu_stats[i].tasks_completed);
   printf(" Number of tasks deferred : %llu\n",gpu_stats[i].tasks_deferred);
   printf(" Number of tasks failed   : %llu\n",gpu_stats[i].tasks_failed);
   printf(" Number of Flops processed: %G\n",gpu_stats[i].flops);
   printf(" Number of Bytes to GPU   : %G\n",gpu_stats[i].traffic_in);
   printf(" Number of Bytes from GPU : %G\n",gpu_stats[i].traffic_out);
   printf(" Time active (sec)        : %f\n",gpu_stats[i].time_active);
   printf("#END_MSG\n");
//  }else{
//   printf("\n#MSG(TAL-SH::NV-TAL): Statistics on GPU #%d: GPU is OFF\n",i);
  }
 }
 return 0;
}
#endif /*NO_GPU*/

//TENSOR BLOCK API:
int tensShape_create(talsh_tens_shape_t ** tshape)
/** Creates a tensor shape and cleans it. **/
{
 int errc;
 if(tshape == NULL) return -1;
 (*tshape)=(talsh_tens_shape_t*)malloc(sizeof(talsh_tens_shape_t));
 if(*tshape == NULL) return TRY_LATER;
 errc=tensShape_clean(*tshape); if(errc) return 1;
 return 0;
}

int tensShape_clean(talsh_tens_shape_t * tshape)
/** Cleans a tensor shape. A clean (initialized to null) tensor shape has .num_dim=-1.
    A further defined tensor shape has .num_dim >= 0. **/
{
 if(tshape != NULL){
  tshape->num_dim=-1; //tensor rank
  tshape->dims=NULL;  //tensor dimension extents
  tshape->divs=NULL;  //tensor dimension dividers (segment sizes)
  tshape->grps=NULL;  //tensor dimension groups
 }else{
  return -1;
 }
 return 0;
}

int tensShape_construct(talsh_tens_shape_t * tshape, int pinned, int rank, const int * dims, const int * divs, const int * grps)
/** (Re-)defines a tensor shape. It is errorneous to pass an uninitialized tensor shape here,
    that is, the tensor shape *(tshape) must be either clean or previously defined. If <rank> > 0,
    <dims[rank]> must be supplied, whereas <divs[rank]> and <grps[rank]> are always optional.
    If <pinned> = YEP and the tensor shape is clean, then the multi-indices will be allocated
    via the multi-index bank (pinned), otherwise a regular malloc will be called. In case the
    tensor shape is already defined, the previous mutli-index storage entries will be reused,
    regardless whether they were pinned or not (argument <pinned> will not be respected!).
    TRY_LATER or DEVICE_UNABLE return statuses are not errors and in this case the input
    tensor shape will stay unchanged. A return status NOT_CLEAN indicates an unsuccessful
    resource release that can be tolerated in general (the construction will still occur). **/
{
 int i,errc;
 int *mi_dims,*mi_divs,*mi_grps;

 errc=0;
//Check arguments:
 if(tshape == NULL) return -1;
 if(rank < 0) return -2;
 if(dims != NULL){for(i=0;i<rank;i++){if(dims[i] < 0) return -3;}}
 if(divs != NULL){for(i=0;i<rank;i++){if(divs[i] < 0) return -4;}}
 if(grps != NULL){for(i=0;i<rank;i++){if(grps[i] < 0) return -5;}}
 if(rank > 0 && dims == NULL) return -6; //dimension extents must be present for rank>0
//Acquire/release resources if needed:
 mi_dims=NULL; mi_divs=NULL; mi_grps=NULL;
 if(rank > 0 && tshape->num_dim <= 0){ //acquire multi-index resources
  if(tshape->dims != NULL || tshape->divs != NULL || tshape->grps != NULL) return -7; //shape must be clean if .num_dim<0
  if(pinned == NOPE){
   mi_dims=(int*)malloc(3*MAX_TENSOR_RANK*sizeof(int));
   if(mi_dims == NULL) return TRY_LATER;
   mi_divs=mi_dims+MAX_TENSOR_RANK;
   mi_grps=mi_divs+MAX_TENSOR_RANK;
  }else{
 //Multi-index "Dimension extents":
   errc=mi_entry_get(&mi_dims); //acquire a mi resource
   if(errc != 0){
    if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return 1;}
   }
 //Multi-index "Dimension dividers":
   errc=mi_entry_get(&mi_divs); //acquire a mi resource
   if(errc != 0){
    i=mi_entry_release(mi_dims);
    if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return 2;}
   }
 //Multi-index "Dimension groups":
   errc=mi_entry_get(&mi_grps); //acquire a mi resource
   if(errc != 0){
    i=mi_entry_release(mi_divs); i=mi_entry_release(mi_dims);
    if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return 3;}
   }
  }
  tshape->dims=mi_dims; tshape->divs=mi_divs; tshape->grps=mi_grps;
  errc=0;
 }else if(rank == 0 && tshape->num_dim > 0){ //release multi-index resources
  errc=tensShape_destruct(tshape); if(errc != 0 && errc != NOT_CLEAN) return 4;
 }
//Define the new tensor shape:
 tshape->num_dim=rank;
 if(dims != NULL){
  for(i=0;i<rank;i++) tshape->dims[i]=dims[i];
 }
 if(divs != NULL){
  for(i=0;i<rank;i++) tshape->divs[i]=divs[i];
 }else{
  for(i=0;i<rank;i++) tshape->divs[i]=tshape->dims[i]; //default dividers (one segment per dimension)
 }
 if(grps != NULL){
  for(i=0;i<rank;i++) tshape->grps[i]=grps[i];
 }else{
  for(i=0;i<rank;i++) tshape->grps[i]=0; //default groups (all indices belong to the unrestricted group)
 }
 return errc; //either 0 or NOT_CLEAN
}

int tensShape_destruct(talsh_tens_shape_t * tshape)
/** Destructs a defined tensor shape (releases resources and cleans it).
    If the input tensor shape is initialized to null, nothing happens.
    In case of an unsuccessful resource release, a return status NOT_CLEAN
    will be returned, which can be considered as a tolerable error since
    the tensor shape will be cleaned anyway (although a leak can occur). **/
{
 int n,pinned,errc;

 n=0; //will be incremented upon an unsuccessful resource release
 if(tshape == NULL) return -1;
 if(tshape->num_dim > 0){ //need to release resources
  if(tshape->dims != NULL){
   pinned=mi_entry_pinned(tshape->dims);
   if(pinned == NOPE){
    free(tshape->dims); //will free all {dims,divs,grps}
    tshape->dims=NULL; tshape->divs=NULL; tshape->grps=NULL;
   }else{
    if(tshape->grps != NULL){errc=mi_entry_release(tshape->grps); if(errc != 0) n++; tshape->grps=NULL;} //release a mi resource
    if(tshape->divs != NULL){errc=mi_entry_release(tshape->divs); if(errc != 0) n++; tshape->divs=NULL;} //release a mi resource
    if(tshape->dims != NULL){errc=mi_entry_release(tshape->dims); if(errc != 0) n++; tshape->dims=NULL;} //release a mi resource
   }
  }else{
   return -2;
  }
 }
 if(n != 0) n=NOT_CLEAN;
 errc=tensShape_clean(tshape);
 return n; //either 0 or NOT_CLEAN
}

int tensShape_destroy(talsh_tens_shape_t * tshape)
/** Completely destroys a tensor shape. **/
{
 int errc,n;
 if(tshape == NULL) return -1;
 n=0; errc=tensShape_destruct(tshape); if(errc) n=NOT_CLEAN;
 free(tshape);
 return n; //either 0 (success) or NOT_CLEAN
}

size_t tensShape_volume(const talsh_tens_shape_t * tshape)
/** Returns the volume of a defined tensor shape, or 0 otherwise. **/
{
 int i;
 size_t vol;

 vol=0;
 if(tshape != NULL){
  if(tshape->num_dim >= 0 && tshape->num_dim <= MAX_TENSOR_RANK){
   vol=1;
   for(i=0;i<tshape->num_dim;i++){
    if(tshape->dims[i] > 0){
     vol*=tshape->dims[i];
    }else{
     return 0;
    }
   }
  }
 }
 return vol;
}

int tensShape_rank(const talsh_tens_shape_t * tshape)
/** Returns the tensor shape rank (number of dimensions). **/
{return tshape->num_dim;}

int tensBlck_create(tensBlck_t **ctens)
/** Creates an empty instance of tensBlck_t and initializes it to null (on Host). **/
{
 *ctens=(tensBlck_t*)malloc(sizeof(tensBlck_t)); if(*ctens == NULL) return TRY_LATER;
 return tensBlck_clean(*ctens);
}

int tensBlck_clean(tensBlck_t *ctens)
/** Cleans an undefined tensBlck_t object. **/
{
 if(ctens == NULL) return -1;
 ctens->data_kind=NO_TYPE;
 ctens->src_rsc=NULL; //source memory resource (where the tensor body is before the operation)
 ctens->dst_rsc=NULL; //destination memory resource (where the tensor body will be after the operation)
 ctens->tmp_rsc=NULL; //temporary memory resource (where the tensor body can be during the operation)
 return tensShape_clean(&(ctens->shape));
}

int tensBlck_destroy(tensBlck_t *ctens)
/** Destroys a defined instance of tensBlck_t (either nullified or shape-defined).
    A return status NOT_CLEAN indicates an unsuccessful resource release, which
    can be considered as a tolerable error (the object will still be destroyed). **/
{
 int n,errc;

 errc=0; n=0;
 if(ctens == NULL) return -1;
 errc=tensBlck_destruct(ctens); if(errc) n=NOT_CLEAN;
 if(ctens->tmp_rsc != NULL){errc=tensDevRsc_destroy(ctens->tmp_rsc); if(errc) n=NOT_CLEAN;}
 if(ctens->dst_rsc != NULL && ctens->dst_rsc != ctens->src_rsc){errc=tensDevRsc_destroy(ctens->dst_rsc); if(errc) n=NOT_CLEAN;}
 if(ctens->src_rsc != NULL){errc=tensDevRsc_destroy(ctens->src_rsc); if(errc) n=NOT_CLEAN;}
 ctens->src_rsc=NULL; ctens->dst_rsc=NULL; ctens->tmp_rsc=NULL;
 free(ctens);
 return n;
}

int tensBlck_construct(tensBlck_t *ctens, //pointer to defined tensor block (either nullified or defined to a value)
                       int pinned,        //YEP: tensor shape multi-indices will be pinned (for GPU), NOPE: regular malloc (not pinned)
                       int trank,         //tensor rank
                       const int *dims,   //tensor dimension extents (when trank > 0)
                       const int *divs,   //tensor dimension dividers (when trank > 0, optional)
                       const int *grps)   //tensor dimension groups (when trank > 0, optional)
/** Constructs (defines/redefines) a tensor block without attaching its body (only the shape).
    If the tensor block is to be used on Nvidia GPUs or other asynchronous devices,
    argument <pinned> must be set to YEP (NOPE will not use pinned memory).
    A return status NOT_CLEAN indicates an unsuccessful resource release, which,
    can be considered as a tolerable error (the object will still be constructed). **/
{
 int n,errc;

 n=0;
 if(ctens == NULL) return -1;
 if(trank < 0 || trank > MAX_TENSOR_RANK) return -2; //invalid tensor rank
 if(trank > 0 && dims == NULL) return -3; //dimension extents must be present for rank>0 tensors
 errc=tensBlck_destruct(ctens); if(errc != 0){if(errc == NOT_CLEAN){n=errc;}else{return 1;}}
 errc=tensShape_construct(&(ctens->shape),pinned,trank,dims,divs,grps);
 if(errc != 0){if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return 2;}}
 return n; //either 0 or NOT_CLEAN
}

int tensBlck_attach_body(tensBlck_t *ctens, //pointer to a shape-defined (constructed) tensor block
                         int data_kind,     //data kind (R4,R8,C4,C8)
                         int dev_id,        //flat device id where the body resides (or should reside): Defaults to Host
                         void *body_ptr,    //pointer to the tensor body (global memory of device <dev_id>)
                         int buf_entry)     //argument buffer entry handle corresponding to the <body_ptr> (optional)
/** Attaches a body to a shape-defined tensor block (with an empty body). If both <body_ptr> and <buf_entry> are absent,
    a resource will be allocated on device <dev_id> in the device argument buffer (if available). If <buf_entry> is absent,
    a defined <body_ptr> points to an external memory (either pinned or not). If both <body_ptr> and <buf_entry> are defined,
    the external memory is assumed to be within that argument buffer entry. In all cases, the memory resource will be
    associated with the .src_rsc component of tensBlck_t. It is forbidden to attempt allocating/attaching a memory resource
    when an existing memory resource is still in use (this will result in an error). A return status of TRY_LATER or
    DEVICE_UNABLE indicates the current or permanent shortage in the necessary resources and is not an error. **/
{
 int errc,dks;
 size_t vol,body_size;

 if(ctens == NULL) return -1;
 errc=tens_valid_data_kind(data_kind,&dks);
 if(errc != YEP || data_kind == NO_TYPE) return -2;
 if(ctens->shape.num_dim < 0 || ctens->shape.num_dim > MAX_TENSOR_RANK) return -3; //tensor block must be shape-defined
 if(body_ptr == NULL && buf_entry >= 0) return -4; //a defined argument buffer entry must be supplied with the corresponding pointer
 if(dev_id < 0){dev_id=encode_device_id(DEV_HOST,0); if(dev_id < 0 || dev_id >= DEV_MAX) return -5;} //dev_id defaults to Host
 if(ctens->src_rsc == NULL){
  errc=tensDevRsc_create(&(ctens->src_rsc)); if(errc != 0 || ctens->src_rsc == NULL) return 1;
 }else{
  if(tensDevRsc_is_empty(ctens->src_rsc) == NOPE) return 2; //source resource is not empty (release it first)
 }
 vol=tensShape_volume(&(ctens->shape)); //tensor body volume (number of elements)
 body_size=vol*((size_t)dks); //tensor body size in bytes
 if(body_ptr == NULL){ //allocate memory in the argument buffer
  errc=tensDevRsc_allocate_mem(ctens->src_rsc,dev_id,body_size,YEP);
  if(errc != 0){if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return 3;}}
 }else{ //associate memory
  errc=tensDevRsc_attach_mem(ctens->src_rsc,dev_id,body_ptr,buf_entry);
  if(errc != 0){if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return 4;}}
 }
 ctens->data_kind=data_kind;
 return 0;
}

int tensBlck_destruct(tensBlck_t *ctens, int release_body, int which_body)
/** Destructs a defined tensor block (releases all resources and initializes the tensor block to null).
    If <release_body> == YEP/NOPE, the global memory resources will be released/kept. Argument <which_body>
    can further regulate which tensor body to be released/kept (SOURCE, DESTINATION, TEMPORARY, EVERYTHING).
    A return status NOT_CLEAN indicates an unsuccessful resource release that may be considered as a
    tolerable error since the tensor block will be nullified anyway. Although device resources are
    released the resource objects themselves are not (they are only destroyed in _destroy method). **/
{
 int n,errc;

 n=0;
 if(ctens == NULL) return -1;
 if(ctens->shape.num_dim >= 0){ //shape-defined tensor block
  if(ctens->shape.num_dim > MAX_TENSOR_RANK) return -2;
//Release the TEMPORARY resource:
  if(ctens->tmp_rsc != NULL &&
     ((release_body == YEP && (which_body == EVERYTHING || which_body == TEMPORARY)) ||
      (release_body == NOPE && (which_body != EVERYTHING && which_body != TEMPORARY)))){
   errc=tensDevRsc_release_all(ctens->tmp_rsc); if(errc != 0) n=NOT_CLEAN; //Note: Resource object is not destroyed!
  }
  ctens->tmp_rsc=NULL;
//Release the DESTINATION resource (only if different from the SOURCE resource):
  if(ctens->dst_rsc != NULL &&
     ((release_body == YEP && (which_body == EVERYTHING || which_body == DESTINATION)) ||
      (release_body == NOPE && (which_body != EVERYTHING && which_body != DESTINATION)))){
   if(ctens->dst_rsc != ctens->src_rsc){
    errc=tensDevRsc_release_all(ctens->dst_rsc); if(errc != 0) n=NOT_CLEAN; //Note: Resource object is not destroyed!
   }else{
    ctens->dst_rsc=NULL; //destination resource simply pointed to the source resource
   }
  }
  ctens->dst_rsc=NULL;
//Release the SOURCE resource:
  if(ctens->src_rsc != NULL &&
     ((release_body == YEP && (which_body == EVERYTHING || which_body == SOURCE)) ||
      (release_body == NOPE && (which_body != EVERYTHING && which_body != SOURCE)))){
   errc=tensDevRsc_release_all(ctens->src_rsc); if(errc != 0) n=NOT_CLEAN; //Note: Resource object is not destroyed!
  }
  ctens->src_rsc=NULL;
  if(tens_valid_data_kind(ctens->data_kind) != YEP) n=NOT_CLEAN;
 }
 ctens->data_kind=NO_TYPE;
 errc=tensShape_destruct(&(ctens->shape)); if(errc){if(errc == NOT_CLEAN){n=NOT_CLEAN;}else{return 1;}}
 return n;
}

int tensBlck_src_dev_id(const tensBlck_t * ctens, int * dev_kind)
/** Returns the device id on which the source data (tensor body) resides.
    If <dev_kind> is provided (!=NULL), the device id will be kind-specific,
    belonging to the device kind <dev_kind>. Otherwise, it will be the flat id.
    A return status DEV_NULL indicates no current source data. A return
    status DEV_MAX indicates a failure (error). **/
{
 int dev_id;

 dev_id=DEV_NULL;
 if(dev_kind != NULL) *dev_kind=DEV_NULL;
 if(ctens == NULL) return DEV_MAX;
 if(ctens->src_rsc != NULL){
  if(dev_kind == NULL){
   dev_id=((*ctens).src_rsc)->dev_id;
  }else{
   dev_id=decode_device_id(((*ctens).src_rsc)->dev_id,dev_kind);
  }
 }
 return dev_id;
}

int tensBlck_present(const tensBlck_t * ctens, int dev_id, int dev_kind)
/** Returns YEP/NOPE if the tensor body is present/absent on the device specified by
    a device id <dev_id> and a device kind <dev_kind>. When <dev_id> is present,
    the presence of <dev_kind> determines whether <dev_id> is a flat or kind-specific.
    When <dev_id> is absent but <dev_kind> is present, the presence will be checked
    against the specified device kind. If both <dev_id> and <dev_kind> are absent,
    any presence will be checked (on any device). A return status NVTAL_FAILURE
    indicates invalid arguments. **/
{
 int src_dev,dst_dev,devn,devk;

 if(ctens == NULL) return NVTAL_FAILURE;
 if(ctens->src_rsc != NULL){src_dev=ctens->src_rsc->dev_id;}else{src_dev=DEV_NULL;}
 if(ctens->dst_rsc != NULL){dst_dev=ctens->dst_rsc->dev_id;}else{dst_dev=DEV_NULL;}
 if(dev_kind == DEV_NULL){
  if(dev_id == DEV_NULL){
   if(src_dev >= 0 || dst_dev >= 0) return YEP;
  }else{
   if(dev_id < 0 || dev_id >= DEV_MAX) return NVTAL_FAILURE;
   if(src_dev == dev_id || dst_dev == dev_id) return YEP;
  }
 }else{
  if(valid_device_kind(dev_kind) != YEP) return NVTAL_FAILURE;
  if(dev_id == DEV_NULL){
   devn=decode_device_id(src_dev,&devk);
   if(devn >= 0 && devk == dev_kind) return YEP;
   devn=decode_device_id(dst_dev,&devk);
   if(devn >= 0 && devk == dev_kind) return YEP;
  }else{
   devn=encode_device_id(dev_id,dev_kind);
   if(devn >= DEV_MAX) return NVTAL_FAILURE;
   if(src_dev == devn || dst_dev == devn) return YEP;
  }
 }
 return NOPE;
}

size_t tensBlck_volume(const tensBlck_t * ctens)
/** Returns the volume of a tensor block (number of elements)
    or zero in cases of an empty tensor block or an error. **/
{
 if(ctens == NULL) return 0;
 size_t tvol=tensShape_volume(&(ctens->shape));
 return tvol;
}

void tensBlck_print(const tensBlck_t * ctens)
/** Print info on a given tensor block. **/
{
 if(ctens != NULL){
  printf("\n#MESSAGE: Printing tensor block info:\n");
  printf(" Tensor block address   : %p\n",ctens);
  printf(" Tensor block data kind : %d\n",ctens->data_kind);
  printf(" Tensor block rank      : %d\n",ctens->shape.num_dim);
  if(ctens->shape.num_dim >= 0 && ctens->shape.num_dim <= MAX_TENSOR_RANK){
   printf(" Tensor block dimensions:"); for(int i=0;i<(ctens->shape.num_dim);i++) printf(" %d",ctens->shape.dims[i]);
   printf("\n Tensor block source resource: %p:\n",ctens->src_rsc);
   if(ctens->src_rsc != NULL){
    printf("  Device ID     : %d\n",ctens->src_rsc->dev_id);
    printf("  Memory address: %p\n",ctens->src_rsc->gmem_p);
    printf("  Buffer entry  : %d\n",ctens->src_rsc->buf_entry);
    printf("  External mem  : %d\n",ctens->src_rsc->mem_attached);
   }
   printf(" Tensor block destination resource: %p:\n",ctens->dst_rsc);
   if(ctens->dst_rsc != NULL){
    printf("  Device ID     : %d\n",ctens->dst_rsc->dev_id);
    printf("  Memory address: %p\n",ctens->dst_rsc->gmem_p);
    printf("  Buffer entry  : %d\n",ctens->dst_rsc->buf_entry);
    printf("  External mem  : %d\n",ctens->dst_rsc->mem_attached);
   }
   printf(" Tensor block temporary resource: %p:\n",ctens->tmp_rsc);
   if(ctens->tmp_rsc != NULL){
    printf("  Device ID     : %d\n",ctens->tmp_rsc->dev_id);
    printf("  Memory address: %p\n",ctens->tmp_rsc->gmem_p);
    printf("  Buffer entry  : %d\n",ctens->tmp_rsc->buf_entry);
    printf("  External mem  : %d\n",ctens->tmp_rsc->mem_attached);
   }
  }
  printf("#END OF MESSAGE\n");
 }else{
  printf("\n#WARNING(tensor_algebra_gpu_nvidia:tensBlck_print): NULL pointer!\n");
 }
 return;
}

int tensBlck_init_host(tensBlck_t * ctens, double init_val)
/** Initializes a tensor block on Host. **/
{
 int i,dev_kind;
 size_t vol;
 float fval;
 float *fp;
 double *dp;
 if(ctens == NULL) return -1;
 if(ctens->shape.num_dim < 0 || ctens->src_rsc == NULL) return -2;
 if(ctens->src_rsc->gmem_p == NULL) return -3;
 if(tens_valid_data_kind(ctens->data_kind) != YEP || ctens->data_kind == NO_TYPE) return -4;
 i=decode_device_id(ctens->src_rsc->dev_id,&dev_kind); if(dev_kind != DEV_HOST || i != 0) return 1;
 vol=tensBlck_volume(ctens); if(vol == 0) return -5;
 switch(ctens->data_kind){
  case R4:
   fval = (float)init_val;
   fp = (float*)(ctens->src_rsc->gmem_p);
#pragma omp parallel for shared(vol,fp,fval) schedule(guided)
   for(size_t l=0; l < vol; l++) fp[l]=fval;
   break;
  case R8:
   dp = (double*)(ctens->src_rsc->gmem_p);
#pragma omp parallel for shared(vol,dp,init_val) schedule(guided)
   for(size_t l=0; l < vol; l++) dp[l]=init_val;
   break;
  default:
   return 2;
 }
 return 0;
}

double tensBlck_norm2_host(const tensBlck_t * ctens)
/** Computes the squared 2-norm of the tensor block on Host. **/
{
 int i,dev_kind;
 size_t vol;
 double nrm2;
 float *fp;
 double *dp;
 if(ctens == NULL) return -1.;
 if(ctens->shape.num_dim < 0 || ctens->src_rsc == NULL) return -2.;
 if(ctens->src_rsc->gmem_p == NULL) return -3.;
 if(tens_valid_data_kind(ctens->data_kind) != YEP || ctens->data_kind == NO_TYPE) return -4.;
 i=decode_device_id(ctens->src_rsc->dev_id,&dev_kind); if(dev_kind != DEV_HOST || i != 0) return -5.;
 vol=tensBlck_volume(ctens); if(vol == 0) return -6.;
 nrm2=0.0;
 switch(ctens->data_kind){
  case R4:
   fp = (float*)(ctens->src_rsc->gmem_p);
#pragma omp parallel for shared(vol,fp) schedule(guided) reduction(+:nrm2)
   for(size_t l=0; l < vol; l++) nrm2+=(double)(fp[l]*fp[l]);
   break;
  case R8:
   dp = (double*)(ctens->src_rsc->gmem_p);
#pragma omp parallel for shared(vol,dp) schedule(guided) reduction(+:nrm2)
   for(size_t l=0; l < vol; l++) nrm2+=dp[l]*dp[l];
   break;
  default:
   return -7.;
 }
 return nrm2;
}

#ifndef NO_GPU
//CUDA TASK API:
__host__ int cuda_task_create(cudaTask_t **cuda_task)
/** Creates an empty instance of cudaTask_t. An unsuccessful attempt
    to allocate memory for the CUDA task returns status TRY_LATER. **/
{
 int errc = 0;
//if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:cuda_task_create): New CUDA task: sizeof(cudaTask_t) = %d",sizeof(cudaTask_t)); //debug
 *cuda_task=(cudaTask_t*)malloc(sizeof(cudaTask_t)); if(*cuda_task == NULL) return TRY_LATER;
 errc=cuda_task_clean(*cuda_task); errc=0;
 return errc;
}

__host__ int cuda_task_clean(cudaTask_t *cuda_task)
/** Cleans (initializes to null) a freshly allocated CUDA task. **/
{
 if(cuda_task == NULL) return -1;
 cuda_task->task_error=-1; cuda_task->gpu_id=-1; cuda_task->num_args=0;
 cuda_task->stream_hl=-1;
 cuda_task->event_start_hl=-1; cuda_task->event_comput_hl=-1;
 cuda_task->event_output_hl=-1; cuda_task->event_finish_hl=-1;
#ifdef GPU_FINE_TIMING
 cuda_task->event_mmbeg_hl=-1; cuda_task->event_mmend_hl=-1;
#endif
 for(int i=0;i<MAX_TENSOR_OPERANDS;++i){
  cuda_task->tens_args[i].tens_p=NULL;
  cuda_task->tens_args[i].prmn_p=NULL;
  cuda_task->tens_args[i].const_mem_entry=-1;
 }
 cuda_task->pref_ptr=NULL;
 return 0;
}

__host__ int cuda_task_construct(cudaTask_t *cuda_task, int gpu_id)
/** Constructs a CUDA task ready for recording on GPU#gpu_id (acquires resources).
    If <gpu_id> is not passed here (negative), the currently active GPU will be used.
    Returns TRY_LATER or DEVICE_UNABLE in case of temporary or permanent
    shortage of GPU resources, respectively (CUDA task is left clean). **/
{
 int i,errc;

 errc=0;
 if(cuda_task == NULL) return -1;
 if(cuda_task->task_error >= 0 || cuda_task->gpu_id >= 0 || cuda_task->num_args > 0) return 1; //CUDA task is not clean: Destruct/clean it first
 i=cuda_task_clean(cuda_task); //just in case
 if(gpu_id < 0) gpu_id=gpu_in_focus();
 if(gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE) return 2; //gpu_id is out of range
 if(gpu_is_mine(gpu_id) > GPU_OFF){
  errc=cuda_stream_get(gpu_id,&(cuda_task->stream_hl));
  if(errc != 0){
   cuda_task->stream_hl=-1; if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=3;
  }else{
   errc=cuda_event_get(gpu_id,&(cuda_task->event_start_hl));
   if(errc != 0){
    cuda_task->event_start_hl=-1; if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=4;
   }else{
    errc=cuda_event_get(gpu_id,&(cuda_task->event_comput_hl));
    if(errc != 0){
     cuda_task->event_comput_hl=-1; if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=5;
    }else{
     errc=cuda_event_get(gpu_id,&(cuda_task->event_output_hl));
     if(errc != 0){
      cuda_task->event_output_hl=-1; if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=6;
     }else{
      errc=cuda_event_get(gpu_id,&(cuda_task->event_finish_hl));
      if(errc != 0){
       cuda_task->event_finish_hl=-1; if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=7;
#ifdef GPU_FINE_TIMING
      }else{
       errc=cuda_event_get(gpu_id,&(cuda_task->event_mmbeg_hl));
       if(errc != 0){
        cuda_task->event_mmbeg_hl=-1; if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=8;
       }else{
        errc=cuda_event_get(gpu_id,&(cuda_task->event_mmend_hl));
        if(errc != 0){
         cuda_task->event_mmend_hl=-1; if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=9;
        }
       }
#endif
      }
     }
    }
   }
  }
  if(errc == 0){
   cuda_task->task_error=-1; cuda_task->gpu_id=gpu_id;
  }else{
#ifdef GPU_FINE_TIMING
   i=cuda_event_release(gpu_id,cuda_task->event_mmbeg_hl); cuda_task->event_mmbeg_hl=-1;
   i=cuda_event_release(gpu_id,cuda_task->event_mmend_hl); cuda_task->event_mmend_hl=-1;
#endif
   i=cuda_event_release(gpu_id,cuda_task->event_finish_hl); cuda_task->event_finish_hl=-1;
   i=cuda_event_release(gpu_id,cuda_task->event_output_hl); cuda_task->event_output_hl=-1;
   i=cuda_event_release(gpu_id,cuda_task->event_comput_hl); cuda_task->event_comput_hl=-1;
   i=cuda_event_release(gpu_id,cuda_task->event_start_hl); cuda_task->event_start_hl=-1;
   i=cuda_stream_release(gpu_id,cuda_task->stream_hl); cuda_task->stream_hl=-1;
   i=cuda_task_clean(cuda_task);
  }
 }else{
  return DEVICE_UNABLE;
 }
 return errc;
}

__host__ int cuda_task_destruct(cudaTask_t *cuda_task)
/** Destructs a defined completed CUDA task or does nothing. If the CUDA task
    is defined but not completed, a return status TRY_LATER is returned.
    If any of the resources used by the CUDA task cannot be released cleanly,
    a return status NOT_CLEAN is returned. Nevertheless, the CUDA task will be
    clean at the end. **/
{
 int n,errc;

 if(cuda_task == NULL) return -1;
 errc=cuda_task_completed(cuda_task); //CUDA task is finalized there (if completed or failed)
 if(errc == CUDA_TASK_EMPTY) return 0;
 n=0; //number of unsuccessful resource releases
 if(errc == CUDA_TASK_COMPLETED || errc == CUDA_TASK_ERROR){
  if(cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return -2; //GPU id is out of allowed range
  if(cuda_task == LastTask[cuda_task->gpu_id]) LastTask[cuda_task->gpu_id]=NULL; //clear task dependency
// Release CUDA resources:
  errc=cuda_stream_release(cuda_task->gpu_id,cuda_task->stream_hl); cuda_task->stream_hl=-1; if(errc != 0) n++;
  errc=cuda_event_release(cuda_task->gpu_id,cuda_task->event_start_hl); cuda_task->event_start_hl=-1; if(errc != 0) n++;
  errc=cuda_event_release(cuda_task->gpu_id,cuda_task->event_comput_hl); cuda_task->event_comput_hl=-1; if(errc != 0) n++;
  errc=cuda_event_release(cuda_task->gpu_id,cuda_task->event_output_hl); cuda_task->event_output_hl=-1; if(errc != 0) n++;
  errc=cuda_event_release(cuda_task->gpu_id,cuda_task->event_finish_hl); cuda_task->event_finish_hl=-1; if(errc != 0) n++;
#ifdef GPU_FINE_TIMING
  errc=cuda_event_release(cuda_task->gpu_id,cuda_task->event_mmbeg_hl); cuda_task->event_mmbeg_hl=-1; if(errc != 0) n++;
  errc=cuda_event_release(cuda_task->gpu_id,cuda_task->event_mmend_hl); cuda_task->event_mmend_hl=-1; if(errc != 0) n++;
#endif
// Release prefactor entry, if needed:
  if(cuda_task->pref_ptr != NULL){
   errc=slab_entry_release(&prefactors,cuda_task->pref_ptr); if(errc != 0) n++;
  }
// Clean the CUDA task:
  errc=cuda_task_clean(cuda_task);
 }else{
  return TRY_LATER; //CUDA task is still in progress
 }
 if(n != 0) n=NOT_CLEAN;
 return n;
}

__host__ int cuda_task_destroy(cudaTask_t *cuda_task)
/** Destroys an instance of cudaTask_t if the CUDA task has completed or empty.
    If the CUDA task is still in progress, a return status TRY_LATER is returned.
    If any of the CUDA task resources could not be released cleanly, a return
    status NOT_CLEAN will be returned but the CUDA task will still be destroyed. **/
{
 int n,errc;

 n=0;
 if(cuda_task == NULL) return -1;
 errc=cuda_task_completed(cuda_task); //CUDA task is finalized there (if completed or failed)
 if(errc == CUDA_TASK_COMPLETED || errc == CUDA_TASK_ERROR){
  errc=cuda_task_destruct(cuda_task); if(errc != 0) n=NOT_CLEAN;
 }else{
  if(errc != CUDA_TASK_EMPTY) return TRY_LATER; //CUDA task is still in progress
 }
 free(cuda_task);
 return n;
}

__host__ int cuda_task_gpu_id(const cudaTask_t *cuda_task)
/** Returns the GPU id associated with a CUDA task. A negative
    return value means a null or empty task was passed here. **/
{
 if(cuda_task == NULL) return -2;
 if(cuda_task->gpu_id >= 0 && cuda_task->gpu_id < MAX_GPUS_PER_NODE) return cuda_task->gpu_id;
 return -1;
}

__host__ int cuda_task_status(cudaTask_t *cuda_task)
/** Checks the status of a CUDA task. Possible status values are listed in tensor_algebra.h
    and tensor_algebra.inc (keep them consistent!). Both CUDA_TASK_COMPLETED (no errors) and
    CUDA_TASK_ERROR (error occurred) suggest a completion of the CUDA task. An unsuccessful
    attempt to find out the status of the CUDA task results in a return status NVTAL_FAILURE. **/
{
 int task_stat,cur_gpu,errc;
 cudaEvent_t *evnt_p;
 cudaError_t err;

 if(cuda_task == NULL) return CUDA_TASK_EMPTY; //NULL task pointer is treated as an empty task here
 if(cuda_task->task_error < 0 && cuda_task->gpu_id < 0) return CUDA_TASK_EMPTY; //empty CUDA task
 if(cuda_task->task_error >= 0 && cuda_task->gpu_id < 0) return NVTAL_FAILURE; //completed task without an assigned GPU
 if(cuda_task->task_error == 0) return CUDA_TASK_COMPLETED; //CUDA task had completed successfully
 if(cuda_task->task_error > 0) return CUDA_TASK_ERROR; //CUDA task error had been registered
 cur_gpu=gpu_in_focus(); if(cur_gpu < 0 || cur_gpu >= MAX_GPUS_PER_NODE) return NVTAL_FAILURE; //get current GPU
 errc=gpu_activate(cuda_task->gpu_id); if(errc != 0) return NVTAL_FAILURE; //could not activate the CUDA task GPU
 evnt_p=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_finish_hl); if(evnt_p == NULL) return NVTAL_FAILURE;
 err=cudaEventQuery(*evnt_p);
 if(err == cudaSuccess){
  cuda_task->task_error=0; errc=cuda_task_finalize(cuda_task); //release unneeded memory resources occupied by the task arguments
  if(errc == 0){
   cuda_task->task_error=0; task_stat=CUDA_TASK_COMPLETED; //CUDA task completed, memory released cleanly
  }else{
   if(VERBOSE) printf("#ERROR(NV-TAL:cuda_task_status): cuda_task_finalize error %d\n",errc);
   cuda_task->task_error=127; task_stat=CUDA_TASK_ERROR; //CUDA task completed, memory could not be released cleanly
  }
  gpu_stats[cuda_task->gpu_id].tasks_completed++;
 }else{
  evnt_p=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_output_hl); if(evnt_p == NULL) return NVTAL_FAILURE;
  err=cudaEventQuery(*evnt_p);
  if(err == cudaSuccess){
   task_stat=CUDA_TASK_OUTPUT_THERE; //computing kernel has finished
  }else{
   evnt_p=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_comput_hl); if(evnt_p == NULL) return NVTAL_FAILURE;
   err=cudaEventQuery(*evnt_p);
   if(err == cudaSuccess){
    task_stat=CUDA_TASK_INPUT_THERE; //computation started, input data is on device (can be reused later)
   }else{
    evnt_p=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_start_hl); if(evnt_p == NULL) return NVTAL_FAILURE;
    err=cudaEventQuery(*evnt_p);
    if(err == cudaSuccess){
     task_stat=CUDA_TASK_STARTED; //task started
    }else{
     task_stat=CUDA_TASK_SCHEDULED; //task has not started yet
    }
   }
  }
 }
 errc=gpu_activate(cur_gpu);
 return task_stat;
}

__host__ int cuda_task_completed(cudaTask_t *cuda_task)
/** Returns CUDA_TASK_COMPLETED or CUDA_TASK_ERROR if an existing CUDA task <cuda_task>
    has completed successfully or due to a scheduling/execution failure, respectively.
    Note that having had successfully checked the CUDA task for completion before will immediately
    suggest completion later (without further querying)! Other possible outputs: CUDA_TASK_EMPTY, CUDA_TASK_SCHEDULED.
    An inability to check the completion status of the CUDA task results in return status NVTAL_FAILURE. **/
{
 int cur_gpu,ret_stat,errc;
 cudaStream_t *strm_p;
 cudaError_t err;

 if(cuda_task == NULL) return CUDA_TASK_EMPTY; //null CUDA task is treated as empty
 if(cuda_task->gpu_id < 0) return CUDA_TASK_EMPTY;
 if(cuda_task->task_error == 0) return CUDA_TASK_COMPLETED; //successful completion had occurred
 if(cuda_task->task_error > 0) return CUDA_TASK_ERROR; //completion due to an error had occurred
 cur_gpu=gpu_in_focus(); if(cur_gpu < 0 || cur_gpu >= MAX_GPUS_PER_NODE) return NVTAL_FAILURE;
 errc=gpu_activate(cuda_task->gpu_id); if(errc != 0) return NVTAL_FAILURE;
 strm_p=cuda_stream_ptr(cuda_task->gpu_id,cuda_task->stream_hl); if(strm_p == NULL) return NVTAL_FAILURE;
 err=cudaStreamQuery(*strm_p);
 if(err != cudaSuccess && err != cudaErrorInvalidResourceHandle){ //task is still in progress
  ret_stat=CUDA_TASK_SCHEDULED;
 }else{ //task completed successfully or has never been scheduled
  if(err == cudaErrorInvalidResourceHandle){ //stream does not exist
   ret_stat=CUDA_TASK_EMPTY;
  }else{
   ret_stat=CUDA_TASK_COMPLETED;
   if(cuda_task->task_error < 0){cuda_task->task_error=0; gpu_stats[cuda_task->gpu_id].tasks_completed++;}
  }
 }
 if(ret_stat == CUDA_TASK_COMPLETED){
  errc=cuda_task_finalize(cuda_task);
  if(errc != 0){
   if(VERBOSE) printf("#ERROR(NV-TAL:cuda_task_completed): cuda_task_finalize error %d\n",errc);
   cuda_task->task_error=127; //resources could not be released properly
  }
 }
 errc=gpu_activate(cur_gpu);
 return ret_stat;
}

__host__ int cuda_task_wait(cudaTask_t *cuda_task)
/** Waits upon completion of a CUDA task: Returns the output of cuda_task_completed(..).
    Possible returns are CUDA_TASK_COMPLETED, CUDA_TASK_ERROR, CUDA_TASK_SCHEDULED, CUDA_TASK_EMPTY.
    In case the completion of a CUDA task cannot be determined, a return status NVTAL_FAILURE is returned. **/
{
 int i,j;

 i=CUDA_TASK_SCHEDULED; j=1;
 while(j>0){
  i=cuda_task_completed(cuda_task); if(i != CUDA_TASK_SCHEDULED) j--;
 }
 return i;
}

__host__ int cuda_tasks_wait(unsigned int num_tasks, cudaTask_t **cuda_tasks, int *task_stats)
/** Waits upon completion of a series of CUDA tasks. Returns zero on success, non-zero on error.
    On success, <task_stats> will contain the completion status for each task. Note that
    <cuda_tasks> points to an array of CUDA task pointers. **/
{
 int i,j,n;

 if(num_tasks > 0){
  if(cuda_tasks != NULL && task_stats != NULL){
   for(i=0;i<num_tasks;i++){task_stats[i]=CUDA_TASK_SCHEDULED;}
   n=num_tasks;
   while(n>0){
    for(i=0;i<num_tasks;i++){
     if(task_stats[i] == CUDA_TASK_SCHEDULED){
      if(cuda_tasks[i] != NULL){
       j=cuda_task_completed(cuda_tasks[i]); task_stats[i]=j;
       if(j != CUDA_TASK_SCHEDULED) n--;
      }else{
       return 1;
      }
     }
    }
   }
  }else{
   return 2;
  }
 }
 return 0;
}

__host__ int cuda_task_error_code(const cudaTask_t *cuda_task)
/** Returns the current .task_error member variable. **/
{return cuda_task->task_error;}

__host__ int cuda_task_dev_rsc_copy(const cudaTask_t *cuda_task, unsigned int arg_num, char which, talsh_dev_rsc_t *dev_rsc)
/** Clones the device resource object from a tensor argument of a CUDA task into <dev_rsc>:
    <which> selects bewteen 's':source, 't':temporary, 'd':destination (resource). **/
{
 int errc;
 tensBlck_t * ctens;

 if(cuda_task == NULL) return -1;
 if(dev_rsc == NULL) return -2;
 if(arg_num >= cuda_task->num_args) return 1;
 ctens=cuda_task->tens_args[arg_num].tens_p;
 if(ctens){
  switch(which){
   case 's': errc=tensDevRsc_clone(ctens->src_rsc,dev_rsc); break;
   case 't': errc=tensDevRsc_clone(ctens->tmp_rsc,dev_rsc); break;
   case 'd': errc=tensDevRsc_clone(ctens->dst_rsc,dev_rsc); break;
   default: errc=2;
  }
 }else{
  errc=3;
 }
 return errc;
}

__host__ int cuda_task_dev_rsc_move(cudaTask_t *cuda_task, unsigned int arg_num, char which, talsh_dev_rsc_t *dev_rsc)
/** Moves the device resource object from a tensor argument of a CUDA task into <dev_rsc>:
    <which> selects bewteen 's':source, 't':temporary, 'd':destination (resource). **/
{
 int errc;
 tensBlck_t * ctens;

 if(cuda_task == NULL) return -1;
 if(dev_rsc == NULL) return -2;
 if(arg_num >= cuda_task->num_args) return 1;
 ctens=cuda_task->tens_args[arg_num].tens_p;
 if(ctens){
  switch(which){
   case 's': errc=tensDevRsc_clone(ctens->src_rsc,dev_rsc); if(errc == 0){free(ctens->src_rsc); ctens->src_rsc=NULL;} break;
   case 't': errc=tensDevRsc_clone(ctens->tmp_rsc,dev_rsc); if(errc == 0){free(ctens->tmp_rsc); ctens->tmp_rsc=NULL;} break;
   case 'd': errc=tensDevRsc_clone(ctens->dst_rsc,dev_rsc); if(errc == 0){free(ctens->dst_rsc); ctens->dst_rsc=NULL;} break;
   default: errc=2;
  }
 }else{
  errc=3;
 }
 return errc;
}

__host__ int cuda_task_arg_has_resource(cudaTask_t *cuda_task, unsigned int arg_num, char which, int *ierr)
/** Queries the existence of a CUDA task resource for tensor argument <arg_num>.
    <which> selects bewteen 's':source, 't':temporary, 'd':destination (resource). **/
{
 int ans;
 tensBlck_t * ctens;

 ans=NOPE; *ierr=0;
 if(cuda_task == NULL){*ierr=-1; return ans;}
 if(arg_num >= cuda_task->num_args){*ierr=1; return ans;}
 ctens=cuda_task->tens_args[arg_num].tens_p;
 if(ctens == NULL){*ierr=2; return ans;}
 switch(which){
  case 's': if(ctens->src_rsc != NULL) ans=YEP; break;
  case 't': if(ctens->tmp_rsc != NULL) ans=YEP; break;
  case 'd': if(ctens->dst_rsc != NULL) ans=YEP; break;
  default: *ierr=3;
 }
 return ans;
}

__host__ int cuda_task_arg_destroy(cudaTask_t *cuda_task, int arg_num) //internal use only
/** Destroys a specific <tensBlck_t> argument in a CUDA task. If <arg_num> is not
    specified (negative), all arguments of the CUDA task will be destroyed. **/
{
 int i,errc;

 errc=0;
 if(cuda_task == NULL) return -1;
 if(arg_num >= cuda_task->num_args) return 1;
 if(arg_num < 0){ //destroy all tensor arguments
  while(cuda_task->num_args > 0){
   i=tensBlck_destroy(cuda_task->tens_args[cuda_task->num_args-1].tens_p);
   if((i == 0 || i == NOT_CLEAN) && errc == 0){errc=i;}else{errc=2;}
   cuda_task->tens_args[--(cuda_task->num_args)].tens_p=NULL;
  }
 }else{ //destroy a specific tensor argument
  i=tensBlck_destroy(cuda_task->tens_args[arg_num].tens_p);
  if((i == 0 || i == NOT_CLEAN) && errc == 0){errc=i;}else{errc=3;}
  cuda_task->tens_args[arg_num].tens_p=NULL;
 }
 return errc;
}

__host__ float cuda_task_time(const cudaTask_t *cuda_task, float *in_copy, float *out_copy, float *comp, float *mmul)
/** Returns the time (in seconds) the CUDA task took to complete. Also, <in_copy> is the input copying time,
    <out_copy> is the output copying time, <comp> is the computing time, and <mmul> is the matrix
    multiplication time in seconds. A negative return value means an error occurred. **/
{
 int cur_gpu,errc;
 float time_ms;
 cudaEvent_t *evnt0_p,*evnt1_p,*evnt2_p,*evnt3_p;
#ifdef GPU_FINE_TIMING
 cudaEvent_t *evnt4_p,*evnt5_p;
#endif
 cudaError_t err;

 if(cuda_task != NULL){
  if(cuda_task->task_error < 0) return -10.0f; //unfinished or empty task
  if(cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return -9.0f;
  cur_gpu=gpu_in_focus(); if(cur_gpu < 0 || cur_gpu >= MAX_GPUS_PER_NODE) return -8.0f;
  errc=gpu_activate(cuda_task->gpu_id); if(errc != 0) return -7.0f;
  evnt0_p=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_start_hl); if(evnt0_p == NULL) return -6.0f;
  evnt1_p=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_comput_hl); if(evnt1_p == NULL) return -5.0f;
  evnt2_p=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_output_hl); if(evnt2_p == NULL) return -4.0f;
  evnt3_p=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_finish_hl); if(evnt3_p == NULL) return -3.0f;
#ifdef GPU_FINE_TIMING
  evnt4_p=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmbeg_hl); if(evnt4_p == NULL) return -2.0f;
  evnt5_p=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmend_hl); if(evnt5_p == NULL) return -1.0f;
#endif
  if(in_copy != NULL){
   err=cudaEventElapsedTime(&time_ms,*evnt0_p,*evnt1_p); //time in miliseconds
   if(err == cudaSuccess){*in_copy=time_ms/1000.0f;}else{*in_copy=-1.0f;}
  }
  if(comp != NULL){
   err=cudaEventElapsedTime(&time_ms,*evnt1_p,*evnt2_p); //time in miliseconds
   if(err == cudaSuccess){*comp=time_ms/1000.0f;}else{*comp=-1.0f;}
  }
  if(out_copy != NULL){
   err=cudaEventElapsedTime(&time_ms,*evnt2_p,*evnt3_p); //time in miliseconds
   if(err == cudaSuccess){*out_copy=time_ms/1000.0f;}else{*out_copy=-1.0f;}
  }
#ifdef GPU_FINE_TIMING
  if(mmul != NULL){
   err=cudaEventElapsedTime(&time_ms,*evnt4_p,*evnt5_p); //time in miliseconds
   if(err == cudaSuccess){*mmul=time_ms/1000.0f;}else{*mmul=-1.0f;}
  }
#endif
  err=cudaEventElapsedTime(&time_ms,*evnt0_p,*evnt3_p); //time in miliseconds
  if(err == cudaSuccess){time_ms/=1000.0f;}else{time_ms=-1.0f;} //time in seconds
  errc=gpu_activate(cur_gpu);
  return time_ms;
 }else{
  return -13.666f; //null task
 }
}

__host__ float cuda_task_time_(const cudaTask_t *cuda_task, float *in_copy, float *out_copy, float *comp, float *mmul)
{
 return cuda_task_time(cuda_task,in_copy,out_copy,comp,mmul);
}

void cuda_task_print(const cudaTask_t *cuda_task)
/** Prints CUDA task info. **/
{
 if(cuda_task != NULL){
  printf("\n#MESSAGE: Printing CUDA task info:\n");
  printf(" CUDA task status             : %d\n",cuda_task->task_error);
  printf(" CUDA task GPU id             : %d\n",cuda_task->gpu_id);
  printf(" CUDA task stream handle      : %d\n",cuda_task->stream_hl);
  printf(" CUDA task event_start handle : %d\n",cuda_task->event_start_hl);
  printf(" CUDA task event_comput handle: %d\n",cuda_task->event_comput_hl);
  printf(" CUDA task event_output handle: %d\n",cuda_task->event_output_hl);
  printf(" CUDA task event_finish handle: %d\n",cuda_task->event_finish_hl);
#ifdef GPU_FINE_TIMING
  printf(" CUDA task event_mmbeg handle : %d\n",cuda_task->event_mmbeg_hl);
  printf(" CUDA task event_mmend handle : %d\n",cuda_task->event_mmend_hl);
#endif
  printf(" CUDA task coherence_var      : %u\n",cuda_task->coherence);
  printf(" CUDA task num_args           : %u\n",cuda_task->num_args);
  if(cuda_task->num_args <= MAX_TENSOR_OPERANDS){
   for(int i=0; i < cuda_task->num_args; ++i){
    printf("  Tensor argument #%d address: %p\n",i,cuda_task->tens_args[i].tens_p);
    tensBlck_print(cuda_task->tens_args[i].tens_p);
   }
  }else{
   printf(" ERROR: Invalid number of arguments!!!\n");
  }
  printf("#END OF MESSAGE\n");
 }else{
  printf("\n#WARNING(tensor_algebra_gpu_nvidia:cuda_task_print): NULL pointer!\n");
 }
 return;
}

__host__ static int cuda_task_set_arg(cudaTask_t *cuda_task, unsigned int arg_num, tensBlck_t *tens_p)
/** Sets a specific tensor argument in a CUDA task. The tensor argument is associated with
    the provided tensor block and the required temporary multi-index entries are acquired.
    If the multi-index resources cannot be acquired at this time, TRY_LATER is returned. **/
{
 int cae,errc;
 if(cuda_task == NULL) return -1;
 if(cuda_task->task_error >= 0 || cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return -2; //finished or empty CUDA task
 if(arg_num >= MAX_TENSOR_OPERANDS) return -3; //[0..MAX_TENSOR_OPERANDS-1]
 if(tens_p == NULL) return -4;
 if(gpu_is_mine(cuda_task->gpu_id) > GPU_OFF){
//Associate the tensor block:
  cuda_task->tens_args[arg_num].tens_p=tens_p; //no checks, just do it
//Acquire a multi-index entry in pinned Host memory:
  errc=mi_entry_get(&(cuda_task->tens_args[arg_num].prmn_p));
  if(errc){cuda_task->tens_args[arg_num].prmn_p=NULL; cuda_task->tens_args[arg_num].tens_p=NULL; return TRY_LATER;}
//Acquire a paired multi-index entry in GPU constant memory:
  errc=const_args_entry_get(cuda_task->gpu_id,&cae);
  if(errc == 0){
   cuda_task->tens_args[arg_num].const_mem_entry=cae;
  }else{
   cuda_task->tens_args[arg_num].prmn_p=NULL; cuda_task->tens_args[arg_num].tens_p=NULL;
   return TRY_LATER;
  }
  cuda_task->num_args=MAX(cuda_task->num_args,arg_num+1); //it is user's responsibility to set all preceding arguments
 }else{
  return 1;
 }
 return 0;
}

__host__ static int cuda_task_set_prefactor(cudaTask_t *cuda_task, talshComplex4 prefactor)
/** Sets a complex prefactor for the tensor operation in a CUDA task (single precision). **/
{
 int errc;
 void * pref_p;

 if(cuda_task == NULL) return -1;
 if(cuda_task->task_error >= 0 || cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return -2; //finished or empty CUDA task
 errc=slab_entry_get(&prefactors,&pref_p); if(errc != 0) return -3;
 cuda_task->pref_ptr=pref_p;
 *((talshComplex4*)(cuda_task->pref_ptr))=prefactor;
 return 0;
}

__host__ static int cuda_task_set_prefactor(cudaTask_t *cuda_task, talshComplex8 prefactor)
/** Sets a complex prefactor for the tensor operation in a CUDA task (double precision). **/
{
 int errc;
 void * pref_p;

 if(cuda_task == NULL) return -1;
 if(cuda_task->task_error >= 0 || cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return -2; //finished or empty CUDA task
 errc=slab_entry_get(&prefactors,&pref_p); if(errc != 0) return -3;
 cuda_task->pref_ptr=pref_p;
 *((talshComplex8*)(cuda_task->pref_ptr))=prefactor;
 return 0;
}

__host__ static int cuda_task_record(cudaTask_t *cuda_task, unsigned int coh_ctrl, unsigned int err_code)
/** Records a scheduled CUDA task. A successfully scheduled CUDA task has <err_code>=0,
    otherwise a positive <err_code> indicates a task scheduling failure. In the latter
    case, the CUDA task will be finalized here. Special error code NVTAL_DEFERRED is a
    non-critical task scheduling failure, not considered as an error.  **/
{
 int i,errc;

 if(cuda_task == NULL) return -1;
 if(cuda_task->task_error >= 0) return -2; //CUDA task is not clean
 if(cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return -3; //GPU ID is out of range or CUDA task is not clean
 if(cuda_task->num_args == 0 || cuda_task->num_args > MAX_TENSOR_OPERANDS) return -4; //no operands associated with the task
 for(i=0;i<cuda_task->num_args;i++){if(cuda_task->tens_args[i].tens_p == NULL) return -5;} //all tensor arguments must be set
 if(err_code == 0){ //successfully scheduled CUDA task
  if(gpu_is_mine(cuda_task->gpu_id) > GPU_OFF){
   cuda_task->task_error=-1; cuda_task->coherence=coh_ctrl;
  }else{
   cuda_task->task_error=13; cuda_task->coherence=coh_ctrl; //GPU is not mine
   errc=cuda_task_finalize(cuda_task); gpu_stats[cuda_task->gpu_id].tasks_failed++;
  }
 }else{ //CUDA task that failed scheduling
  cuda_task->task_error=err_code; cuda_task->coherence=coh_ctrl;
  errc=cuda_task_finalize(cuda_task);
  if(err_code == NVTAL_DEFERRED){
   gpu_stats[cuda_task->gpu_id].tasks_deferred++;
  }else{
   gpu_stats[cuda_task->gpu_id].tasks_failed++;
  }
 }
 return 0;
}

__host__ static int cuda_task_finalize(cudaTask_t *cuda_task) //do not call this function in tensor operations
/** Releases unneeded (temporary and other) memory resources right after a CUDA task
    has completed or failed. In case the resources cannot be released cleanly,
    it returns NOT_CLEAN just as a warning, but the CUDA task is finalized anyway.
    It also applies the coherence control protocol (for successfully completed tasks only).
    Note that the CUDA task is not destructed here, namely CUDA stream/event resources and the
    .tens_p component of .tens_args[] are unmodified (.prmn_p and .const_mem_entry are released). **/
{
 const unsigned int TWO_BITS_SET = 3; //two right bits are set: {0:D,1:M,2:T,3:K}
 unsigned int bts,coh,s_d_same;
 int i,ret_stat,errc;
 cudaTensArg_t *tens_arg;

 if(cuda_task == NULL) return -1;
 if(cuda_task->task_error < 0) return 1; //unfinished or empty CUDA task cannot be finalized
 if(cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return 2; //invalid GPU id or empty
 if(cuda_task->num_args > MAX_TENSOR_OPERANDS) return 3; //invalid number of tensor arguments
 ret_stat=0; coh=cuda_task->coherence;
//Release resources for tensor arguments:
 for(i=cuda_task->num_args-1;i>=0;i--){ //last argument corresponds to the first (minor) two bits
  bts=(coh)&(TWO_BITS_SET);
  tens_arg=&(cuda_task->tens_args[i]);
  if(tens_arg->tens_p != NULL){ //pointer to the tensor block associated with this argument
   if(tens_arg->tens_p->src_rsc == NULL) return -2; //source must always be present
   if(tens_arg->tens_p->dst_rsc != NULL){
    if(tens_arg->tens_p->src_rsc->dev_id == tens_arg->tens_p->dst_rsc->dev_id){s_d_same=YEP;}else{s_d_same=NOPE;};
   }else{
    if(cuda_task->task_error == 0) return -3; //destination resource must be present for successfully completed CUDA tasks
    s_d_same=NOPE; //no destination resource (failed CUDA tasks only)
   }
// Release temporary resources (always):
   if(tens_arg->tens_p->tmp_rsc != NULL){
    errc=tensDevRsc_release_all(tens_arg->tens_p->tmp_rsc);
    if(errc){
     if(VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): tmp_rsc resource release error %d\n",errc);
     ret_stat=NOT_CLEAN;
    }
   }
// Release source/destination resources if needed:
   if(tens_arg->tens_p->dst_rsc == tens_arg->tens_p->src_rsc) tens_arg->tens_p->dst_rsc=NULL;
   if(cuda_task->task_error == 0){ //coherence control for successfully completed CUDA tasks
    if(bts < 2){
     if(s_d_same == NOPE){
      errc=tensDevRsc_release_all(tens_arg->tens_p->src_rsc);
      if(errc){
       if(VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): src_rsc resource release error %d\n",errc);
       ret_stat=NOT_CLEAN;
      }
     }
     if(bts == 0 && tens_arg->tens_p->dst_rsc != NULL){
      errc=tensDevRsc_release_all(tens_arg->tens_p->dst_rsc);
      if(errc){
       if(VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): dst_rsc resource release error %d\n",errc);
       ret_stat=NOT_CLEAN;
      }
     }
    }else if(bts == 2){
     if(s_d_same == NOPE && tens_arg->tens_p->dst_rsc != NULL){
      errc=tensDevRsc_release_all(tens_arg->tens_p->dst_rsc);
      if(errc){
       if(VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): dst_rsc resource release error %d\n",errc);
       ret_stat=NOT_CLEAN;
      }
     }
    }
   }else{ //failed CUDA task
    if(tens_arg->tens_p->dst_rsc != NULL){
     errc=tensDevRsc_release_all(tens_arg->tens_p->dst_rsc);
     if(errc){
      if(VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): dst_rsc resource release error %d\n",errc);
      ret_stat=NOT_CLEAN;
     }
    }
   }
// Release multi-index entries if any:
   if(tens_arg->prmn_p != NULL){ //if .prmn_p is not from the internal pinned slab nothing will be done:
    if(mi_entry_pinned(tens_arg->prmn_p) == YEP){
     errc=mi_entry_release(tens_arg->prmn_p);
     if(errc){
      if(VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): permutation entry release error %d\n",errc);
      ret_stat=NOT_CLEAN;
     }
     tens_arg->prmn_p=NULL;
    }
   }
   if(tens_arg->const_mem_entry >= 0){
    errc=const_args_entry_free(cuda_task->gpu_id,tens_arg->const_mem_entry);
    if(errc){
     if(VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): constant memory resource release error %d\n",errc);
     ret_stat=NOT_CLEAN;
    }
    tens_arg->const_mem_entry=0;
   }
   //printf("\n#DEBUG(NV-TAL::cuda_task_finalize): tensBlck_t argument %d end state:\n",i); tensBlck_print(tens_arg->tens_p); //debug
  }else{
   if(cuda_task->task_error == 0) return -4; //successfully completed CUDA tasks must have all tensor arguments associated
  }
  coh=coh>>2; //select the 2-bits for the next argument
 }
//Release prefactor resource, if needed:
 if(cuda_task->pref_ptr != NULL){
  errc=slab_entry_release(&prefactors,cuda_task->pref_ptr);
  if(errc){
   if(VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): prefactor release error %d\n",errc);
   ret_stat=NOT_CLEAN;
  }
  cuda_task->pref_ptr=NULL;
 }
 return ret_stat;
}
//-------------------------------------------------
//EXPORTED FUNCTIONS (callable from C/C++/Fortran):
//---------------------------------------------------------------------------
// MATRIX MULTIPLICATION 'TN' (blocking, slow):
template <typename T>
__host__ int gpu_matrix_multiply_tn(size_t ll, size_t lr, size_t lc,
                                    const T * lmat, const T * rmat, T * dmat)
/** dmat(0:ll-1,0:lr-1)+=lmat(0:lc-1,0:ll-1)*rmat(0:lc-1,0:lr-1)
All matrices are in Host memory. Executed on the currently set GPU device. **/
{
 size_t dsize,lsize,rsize;
 T *dptr,*lptr,*rptr;
 int bx,by,err_code;
 const char *err_msg;
 cudaError_t err;

 if(lc > 0 && ll > 0 && lr > 0 && lmat != NULL && rmat != NULL && dmat != NULL){
  err=cudaGetLastError(); err=cudaSuccess;
  dsize=ll*lr*sizeof(T); lsize=lc*ll*sizeof(T); rsize=lc*lr*sizeof(T);
  err_code=gpu_mem_alloc((void**)&dptr,dsize); if(err_code != 0) return 1;
  err_code=gpu_mem_alloc((void**)&lptr,lsize); if(err_code != 0) return 2;
  err_code=gpu_mem_alloc((void**)&rptr,rsize); if(err_code != 0) return 3;
  err=cudaMemcpy((void*)dptr,(void*)dmat,dsize,cudaMemcpyHostToDevice); if(err != cudaSuccess) return 4;
  err=cudaMemcpy((void*)lptr,(void*)lmat,lsize,cudaMemcpyHostToDevice); if(err != cudaSuccess) return 5;
  err=cudaMemcpy((void*)rptr,(void*)rmat,rsize,cudaMemcpyHostToDevice); if(err != cudaSuccess) return 6;
  err_code=gpu_get_error_count();
  bx=1+(ll-1)/MAT_MULT_TILE_DIMX; by=1+(lr-1)/MAT_MULT_TILE_DIMY; limit_cuda_blocks2d(MAX_CUDA_BLOCKS,&bx,&by);
  dim3 blcks(bx,by); dim3 thrds(MAT_MULT_TILE_DIMX,MAT_MULT_TILE_DIMY);
  //if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_matrix_multiply_tn): Running GPU kernel ..."); //debug
  gpu_matrix_multiply_tn__<<<blcks,thrds>>>(ll,lr,lc,lptr,rptr,dptr,(T)(1.0));
  err=cudaDeviceSynchronize(); if(err != cudaSuccess) return 7;
  err=cudaGetLastError();
  if(err!=cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_matrix_multiply_tn): Kernel error: %s\n",err_msg);
   return 8;
  }
  if(gpu_get_error_count() > err_code) return 9;
  //if(DEBUG) printf("Done: %d",err); //debug
  err=cudaMemcpy((void*)dmat,(void*)dptr,dsize,cudaMemcpyDeviceToHost); if(err != cudaSuccess) return 10;
  err=cudaDeviceSynchronize(); if(err != cudaSuccess) return 11;
  err_code=gpu_mem_free((void*)rptr); if(err_code != 0) return 12;
  err_code=gpu_mem_free((void*)lptr); if(err_code != 0) return 13;
  err_code=gpu_mem_free((void*)dptr); if(err_code != 0) return 14;
  err=cudaDeviceSynchronize(); if(err != cudaSuccess) return 15;
 }else{
  return 16;
 }
 return 0;
}
//-----------------------------------------------------------------------------------------------------------------------------
// TENSOR BODY CLONING (non-blocking):
__host__ int gpu_tensor_block_place(tensBlck_t *ctens, int gpu_id, unsigned int coh_ctrl, cudaTask_t *cuda_task, void *dev_mem)
/** Copies/moves the tensor body to a different GPU (gpu_id >= 0) or Host (gpu_id < 0).
    If <dev_mem> is a valid target device memory pointer, it will be used for storage, otherwise buffer memory will be allocated.
    A non-zero return status indicates an error. If the error code is negative, the CUDA task was not recorded.
    For positive error codes, the CUDA task was recorded. If the source device where the tensor body resides
    coincides with the destination device, no transfer will be scheduled.
    The source tensor body must reside either on Host or on Nvidia GPU. **/
{
 int j,tds,gpu_ex,src_gpu,devk,cur_gpu,devid,nclean,errc;
 size_t tvol,tsize;
 cudaStream_t *cuda_stream;
 cudaEvent_t *cuda_start,*cuda_comput,*cuda_output,*cuda_finish,*dep_event;
 cudaError_t err;
 const char *err_msg;

 errc=0; nclean=0;
 //Argument check:
 if(ctens == NULL) return -1;
 if(cuda_task == NULL) return -2;
 if(cuda_task_gpu_id(cuda_task) >= 0) return -3; //CUDA task is not clean (destruct/clean it first)
 if(tens_valid_data_kind(ctens->data_kind,&tds) != YEP) return -4;
 if(tensBlck_present(ctens,DEV_NULL,DEV_NVIDIA_GPU) == YEP || tensBlck_present(ctens,DEV_NULL,DEV_HOST) == YEP){
  //Determine the id of the transfer executing GPU:
  cur_gpu=gpu_in_focus(); //save the current GPU
  gpu_ex=DEV_NULL; //executing GPU
  src_gpu=tensBlck_src_dev_id(ctens,&devk); //source GPU (or Host)
  if(devk == DEV_HOST){src_gpu=DEV_NULL;}else{if(devk != DEV_NVIDIA_GPU) return -5;} //src_gpu: source GPU (-1:Host)
  if(gpu_id >= 0 && gpu_id < MAX_GPUS_PER_NODE){ //destination is a GPU
   gpu_ex=gpu_id; if(gpu_is_mine(gpu_ex) <= GPU_OFF) return -6;
  }else if(gpu_id < 0){ //destination is Host
   if(src_gpu >= 0){gpu_ex=src_gpu; if(gpu_is_mine(gpu_ex) <= GPU_OFF) return -7;}
  }else{
   return -8; //invalid gpu_id
  }
  //Construct the CUDA task:
  if(gpu_ex < 0){ //Host-to-self transfer requested (no transfer)
   errc=cuda_task_construct(cuda_task);
  }else{ //Host-to-GPU, GPU-to-Host, GPU-to-GPU
   gpu_stats[gpu_ex].tasks_submitted++;
   //Check peer access if appropriate:
   if(src_gpu >= 0 && src_gpu != gpu_ex){
    err=cudaDeviceCanAccessPeer(&j,gpu_ex,src_gpu);
    if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
   }
   //Activate the transfer executing GPU:
   if(gpu_ex != cur_gpu){j=gpu_activate(gpu_ex); if(j){j=gpu_activate(cur_gpu); return -9;}} //activate the target GPU
   err=cudaGetLastError();
   if(err != cudaSuccess){
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Previous error detected: %s\n",cudaGetErrorString(err));
    ++nclean; err=cudaSuccess; //clear the GPU error status (sets NOT_CLEAN on exit)
   }
   errc=cuda_task_construct(cuda_task,gpu_ex); if(errc) j=gpu_activate(cur_gpu);
  }
  if(errc){if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return -10;}}

  // *** From this point all error codes must be positive and the CUDA task must be recorded! ***
  //Set the CUDA task argument(s):
  errc=cuda_task_set_arg(cuda_task,0,ctens);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    j=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); j=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
    return errc;
   }else{
    j=cuda_task_record(cuda_task,coh_ctrl,1); j=gpu_activate(cur_gpu);
    return 1;
   }
  }
  //Determine the volume/size of the tensor block:
  tvol=tensBlck_volume(ctens); tsize=tvol*tds;
  if(tvol == 0){errc=cuda_task_record(cuda_task,coh_ctrl,2); errc=gpu_activate(cur_gpu); return 2;}
  //Associate CUDA stream and event pointers locally for convenience:
  cuda_stream=cuda_stream_ptr(cuda_task->gpu_id,cuda_task->stream_hl);
  if(cuda_stream == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,3); errc=gpu_activate(cur_gpu); return 3;}
  cuda_start=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_start_hl);
  if(cuda_start == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,4); errc=gpu_activate(cur_gpu); return 4;}
  cuda_comput=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_comput_hl);
  if(cuda_comput == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,5); errc=gpu_activate(cur_gpu); return 5;}
  cuda_output=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_output_hl);
  if(cuda_output == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,6); errc=gpu_activate(cur_gpu); return 6;}
  cuda_finish=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_finish_hl);
  if(cuda_finish == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,7); errc=gpu_activate(cur_gpu); return 7;}
  //Acquire global memory resources (destination resource):
  if(gpu_id >= 0){devid=encode_device_id(DEV_NVIDIA_GPU,gpu_id);}else{devid=encode_device_id(DEV_HOST,0);} //flat device id of the destination
  if(ctens->dst_rsc == ctens->src_rsc) ctens->dst_rsc=NULL;
  if(gpu_ex >= 0 && gpu_id != src_gpu){ //data is on a different GPU device or Host
   if(ctens->dst_rsc == NULL){
    errc=tensDevRsc_create(&(ctens->dst_rsc)); if(errc){j=cuda_task_record(cuda_task,coh_ctrl,8); j=gpu_activate(cur_gpu); return 8;}
   }else{
    if(tensDevRsc_is_empty(ctens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(ctens->dst_rsc); if(errc) ++nclean;}
   }
   if(dev_mem == NULL){
    errc=tensDevRsc_allocate_mem(ctens->dst_rsc,devid,tsize,YEP); //device memory is allocated in the device argument buffer
   }else{
    errc=tensDevRsc_attach_mem(ctens->dst_rsc,devid,dev_mem); //externally provided device memory will be used for storage
   }
   if(errc){
    if(errc == TRY_LATER || errc == DEVICE_UNABLE){
     j=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); j=gpu_activate(cur_gpu);
     return errc;
    }else{
     j=cuda_task_record(cuda_task,coh_ctrl,9); j=gpu_activate(cur_gpu);
     return 9;
    }
   }
  }else{
   if(ctens->dst_rsc != NULL){
    if(tensDevRsc_is_empty(ctens->dst_rsc) == NOPE){j=tensDevRsc_release_all(ctens->dst_rsc); if(j) ++nclean;}
   }
   ctens->dst_rsc=ctens->src_rsc; //destination and source resources are the same (because the data is already on the executing GPU or Host)
  }
  //Record the start event:
  err=cudaEventRecord(*cuda_start,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Unable to record the start event: %s\n",err_msg);
   j=cuda_task_record(cuda_task,coh_ctrl,10); j=gpu_activate(cur_gpu); return 10;
  }
  //Schedule the data transfer:
  if(gpu_ex >= 0 && gpu_id != src_gpu){
   //Make sure the data transfer does not begin before the data transfer from the previous task has finished:
   if(LastTask[gpu_ex] != NULL){ //`This should be done atomically for thread safety
    dep_event=cuda_event_ptr(LastTask[gpu_ex]->gpu_id,LastTask[gpu_ex]->event_comput_hl);
    err=cudaStreamWaitEvent(*cuda_stream,*dep_event,0); //input transfers should only begin after the previous task input transfers have completed
    if(err != cudaSuccess){
     err_msg=cudaGetErrorString(err);
     if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Unable to create a task dependency: %s\n",err_msg);
     j=cuda_task_record(cuda_task,coh_ctrl,11); j=gpu_activate(cur_gpu); return 11;
    }
   }
   //Transfer:
   err=cudaMemcpyAsync(ctens->dst_rsc->gmem_p,ctens->src_rsc->gmem_p,tsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Tensor body transfer failed: %s\n",err_msg);
    j=cuda_task_record(cuda_task,coh_ctrl,12); j=gpu_activate(cur_gpu); return 12;
   }
   if(gpu_id >= 0){ //incoming traffic
    gpu_stats[gpu_ex].traffic_in+=tsize;
   }else{ //outgoing traffic (to Host)
    gpu_stats[gpu_ex].traffic_out+=tsize;
   }
  }
  //Record other events:
  err=cudaEventRecord(*cuda_comput,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Unable to record the compute event: %s\n",err_msg);
   j=cuda_task_record(cuda_task,coh_ctrl,13); j=gpu_activate(cur_gpu); return 13;
  }
  err=cudaEventRecord(*cuda_output,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Unable to record the output event: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,14); errc=gpu_activate(cur_gpu); return 14;
  }
  err=cudaEventRecord(*cuda_finish,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Unable to record the finish event: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,15); errc=gpu_activate(cur_gpu); return 15;
  }
  //Record the successfully scheduled CUDA task, update the Last Task, and restore the original GPU:
  errc=cuda_task_record(cuda_task,coh_ctrl,0);
  if(gpu_ex >= 0 && gpu_ex != src_gpu) LastTask[gpu_ex]=cuda_task;
  if(gpu_ex >= 0 && gpu_ex != cur_gpu) j=gpu_activate(cur_gpu);
 }else{
  return -11; //tensor block is neither present on Host nor on any Nvidia GPU
 }
 if(nclean > 0 && errc == 0) errc=NOT_CLEAN;
 return errc;
}
//-------------------------------------------------------------------------------------------------------------------------
// TENSOR INITIALIZATION (non-blocking):
__host__ int gpu_tensor_block_init(tensBlck_t *dtens, double val, unsigned int coh_ctrl, cudaTask_t *cuda_task, int gpu_id)
/**
dtens(:)=scalar_value
INPUT:
 # val - initialization value;
 # coh_ctrl - one of the COPY_X parameters regulating the data presence for each tensor argument;
 # cuda_task - pointer to an empty (clean) CUDA task;
 # gpu_id - suggested GPU ID on which the operation is to be scheduled (-1: defaults to the optimal one);
OUTPUT:
 # dtens - initialized destination tensor;
 # cuda_task - recorded CUDA task (either successfully scheduled or failed).
NOTES:
 # If the tensor operation has been scheduled successfully, a recorded (active) CUDA task
   will be returned along with zero return status. A scheduling error results in either
   a negative (at early stages) or positive (at later stages) return status. In the former case
   the CUDA task is left clean, while at the latter case it will be recorded as failed (error).
 # Special return statuses TRY_LATER and DEVICE_UNABLE are not errors but merely indicators
   of the current or permanent lack of resources, respectively. However, the CUDA task status
   in these cases will still be set to an error (always check the function return status!).
 # If <gpu_id> is out of the legitimate GPU range, it will be replaced by an optimal one,
   based on argument residence and the current load of GPU(s).
**/
{
 int i,j,drank,tds_d,gpu_d,gpu_num,cur_gpu,targ_dev,bx,errc,stat;
 size_t vol_d,dsize;
 unsigned int coh;
 const unsigned int TWO_BITS_SET = 3; //two right bits are set
 void *darg;
 float fval;
 cudaStream_t *cuda_stream;
 cudaEvent_t *cuda_start,*cuda_comput,*cuda_output,*cuda_finish,*dep_event;
#ifdef GPU_FINE_TIMING
 cudaEvent_t *cuda_mmbeg,*cuda_mmend;
#endif
 cudaError_t err;
 const char *err_msg;

 //if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): GPU Tensor Initialization:\n"); //debug
 stat=0; //return status in case of successful scheduling
//Check function arguments:
 if(dtens == NULL || cuda_task == NULL) return -1;
 if(tensBlck_present(dtens) != YEP) return -2; //tensor block must reside in some device memory
 if(cuda_task_gpu_id(cuda_task) >= 0) return -3; //CUDA task is not clean (destruct/clean it first)
//Check tensor arguments:
 drank=(dtens->shape).num_dim; //destination tensor rank
 if(drank < 0 || drank > MAX_TENSOR_RANK) return -4;
 if(tens_valid_data_kind(dtens->data_kind,&tds_d) != YEP) return -5; //tds_d: destination tensor element size in bytes
 if(dtens->data_kind <= 0) return -6; //tensor must have been previsously allocated with a certain data kind
 if(dtens->src_rsc == NULL) return -7; //source resource must always be present
 if(tensDevRsc_is_empty(dtens->src_rsc) != NOPE) return -8; //source resource must be present (tensor body)
//Activate the right GPU:
 if(gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE){gpu_num=tens_op_best_gpu(dtens);}else{gpu_num=gpu_id;}
 if(gpu_is_mine(gpu_num) <= GPU_OFF) return -28; //GPU is not mine or error
 gpu_stats[gpu_num].tasks_submitted++;
 gpu_d=decode_device_id(dtens->src_rsc->dev_id,&j); if(gpu_d < 0) return -29; //destination tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_d != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_d); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_d=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 cur_gpu=gpu_in_focus(); //save the current GPU
 if(gpu_num != cur_gpu){errc=gpu_activate(gpu_num); if(errc){errc=gpu_activate(cur_gpu); return -32;}} //activate the target GPU
 err=cudaGetLastError(); err=cudaSuccess; //clear the GPU error status
 targ_dev=encode_device_id(DEV_NVIDIA_GPU,gpu_num); //flat device id
//Construct a CUDA task (acquire CUDA resources) for the target GPU:
 errc=cuda_task_construct(cuda_task,gpu_num);
 if(errc){i=gpu_activate(cur_gpu); if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return -33;}}

// *** From this point all error codes must be positive and the CUDA task must be recorded! ***
//Set up tensor arguments (allocates additional resources for each tensor argument):
// Destination argument:
 errc=cuda_task_set_arg(cuda_task,0,dtens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,1); i=gpu_activate(cur_gpu);
   return 1;
  }
 }
//Associate CUDA stream and event pointers locally for convenience:
 cuda_stream=cuda_stream_ptr(cuda_task->gpu_id,cuda_task->stream_hl);
 if(cuda_stream == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,4); errc=gpu_activate(cur_gpu); return 4;}
 cuda_start=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_start_hl);
 if(cuda_start == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,5); errc=gpu_activate(cur_gpu); return 5;}
 cuda_comput=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_comput_hl);
 if(cuda_comput == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,6); errc=gpu_activate(cur_gpu); return 6;}
 cuda_output=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_output_hl);
 if(cuda_output == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,7); errc=gpu_activate(cur_gpu); return 7;}
 cuda_finish=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_finish_hl);
 if(cuda_finish == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
#ifdef GPU_FINE_TIMING
 cuda_mmbeg=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmbeg_hl);
 if(cuda_mmbeg == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
 cuda_mmend=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmend_hl);
 if(cuda_mmend == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
#endif
 vol_d=tensBlck_volume(dtens); //tensor block volume
 dsize=vol_d*tds_d;            //tensor argument size in bytes
//Acquire global memory resources for tensor arguments if needed:
// Set up destination memory resources in all tensors:
//  Destination tensor:
 if(dtens->dst_rsc == dtens->src_rsc) dtens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_d != gpu_num){ //data is on a different GPU device or Host
  if(dtens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(dtens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,11); i=gpu_activate(cur_gpu); return 11;}
  }else{
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(dtens->dst_rsc,targ_dev,dsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,12); i=gpu_activate(cur_gpu);
    return 12;
   }
  }
 }else{
  if(dtens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  dtens->dst_rsc=dtens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
#ifdef DEBUG_GPU
//DEBUG begin:
 if(DEBUG){
  printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_init):\n");
  printf(" Const args (d)     : %d\n",cuda_task->tens_args[0].const_mem_entry); //debug
  printf(" Block sizes (d)    : %lu\n",dsize); //debug
  printf(" Block ranks (d)    : %d\n",dtens->shape.num_dim); //debug
  printf("\n#END OF DEBUG\n");
 }
//DEBUG end.
#endif /*DEBUG_GPU*/
//Start scheduling CUDA calls:
 err=cudaEventRecord(*cuda_start,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to record the start event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,23); errc=gpu_activate(cur_gpu); return 23;
 }
 if(LastTask[gpu_num] != NULL){ //`This should be done atomically for thread safety
  dep_event=cuda_event_ptr(LastTask[gpu_num]->gpu_id,LastTask[gpu_num]->event_comput_hl);
  err=cudaStreamWaitEvent(*cuda_stream,*dep_event,0); //input transfers should only begin after the previous task input transfers have completed
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to create a task dependency: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,24); errc=gpu_activate(cur_gpu); return 24;
  }
 }
//Schedule forward data transfers for all tensors if needed:
// Destination tensor:
 if(cuda_task->tens_args[0].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(dtens->shape.dims),sizeof(int)*((size_t)drank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[0].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Destination tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,33); errc=gpu_activate(cur_gpu); return 33;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)drank);
  if(gpu_d != gpu_num){ //data is not on the computing GPU
   err=cudaMemcpyAsync(dtens->dst_rsc->gmem_p,dtens->src_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Destination tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,35); errc=gpu_activate(cur_gpu); return 35;
   }
   gpu_stats[gpu_num].traffic_in+=dsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,36); errc=gpu_activate(cur_gpu); return 36;
 }
// Record a CUDA event:
 err=cudaEventRecord(*cuda_comput,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to record the compute event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,37); errc=gpu_activate(cur_gpu); return 37;
 }
//Destination tensor argument does not need transposing:
 darg=dtens->dst_rsc->gmem_p;
//Schedule the appropriate computation kernel:
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmbeg,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to record the mmbeg event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,66); errc=gpu_activate(cur_gpu); return 66;
 }
#endif
// Addition kernel:
 bx=1+(vol_d-1)/THRDS_ARRAY_INIT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
 switch(dtens->data_kind){
  case R4:
   fval=(float)val;
   gpu_array_init__<<<bx,THRDS_ARRAY_INIT,0,*cuda_stream>>>(vol_d,(float*)darg,fval);
   break;
  case R8:
   gpu_array_init__<<<bx,THRDS_ARRAY_INIT,0,*cuda_stream>>>(vol_d,(double*)darg,val);
   break;
  default:
   errc=cuda_task_record(cuda_task,coh_ctrl,48); errc=gpu_activate(cur_gpu); return 48;
 }
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmend,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to record the mmend event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,67); errc=gpu_activate(cur_gpu); return 67;
 }
#endif
//Record a CUDA event (output ready on GPU):
 err=cudaEventRecord(*cuda_output,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to record the output event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,56); errc=gpu_activate(cur_gpu); return 56;
 }
//Transfer back the updated destination tensor if needed ("T","K" coherence control):
 coh=(coh_ctrl)&(TWO_BITS_SET); //select bits 0,1 (destination tensor coherence)
 if(gpu_d != gpu_num && coh >= 2){ //data is not on the computing GPU and coherence control = 2("T") or (3)"K":
  err=cudaMemcpyAsync(dtens->src_rsc->gmem_p,dtens->dst_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Destination tensor body back copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,57); errc=gpu_activate(cur_gpu); return 57;
  }
  gpu_stats[gpu_num].traffic_out+=dsize;
 }
//Record a CUDA event (task finished):
 err=cudaEventRecord(*cuda_finish,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to record the finish event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,58); errc=gpu_activate(cur_gpu); return 58;
 }
//Record the successfully scheduled CUDA task and update the Last Task:
 errc=cuda_task_record(cuda_task,coh_ctrl,0);
 LastTask[gpu_num]=cuda_task;
 if(gpu_num != cur_gpu) errc=gpu_activate(cur_gpu);
 return stat; //either 0 (success) or NOT_CLEAN (warning)
}
//-----------------------------------------------------------------------------------------------------------------------
// TENSOR ADDITION (non-blocking):
__host__ int gpu_tensor_block_add(const int *cptrn, tensBlck_t *ltens, tensBlck_t *dtens, unsigned int coh_ctrl,
                                  cudaTask_t *cuda_task, int gpu_id, double scale_real, double scale_imag, int conj_bits)
/**
dtens(:)+=ltens(:)*scalar
INPUT:
 # cptrn(1:lrank) - contraction pattern: Position correspondence:
                    Uncontracted indices are positive, no contracted indices;
 # ltens - left tensor argument (initialized!);
 # dtens - destination tensor argument (initialized!);
 # coh_ctrl - one of the COPY_XX parameters regulating the data presence for each tensor argument;
 # cuda_task - pointer to an empty (clean) CUDA task;
 # gpu_id - suggested GPU ID on which the operation is to be scheduled (-1: defaults to the optimal one);
 # scale_real - real part of the GEMM alpha coefficient (defaults to 1.0);
 # scale_imag - imaginary part of the GEMM alpha coefficient (defaults to 0.0);
 # conj_bits - tensor argument complex conjugation bits, one bit per argument: {0:D,1:L};
OUTPUT:
 # dtens - updated destination tensor;
 # cuda_task - recorded CUDA task (either successfully scheduled or failed).
NOTES:
 # If the tensor operation has been scheduled successfully, a recorded (active) CUDA task
   will be returned along with zero return status. A scheduling error results in either
   a negative (at early stages) or positive (at later stages) return status. In the former case
   the CUDA task is left clean, while at the latter case it will be recorded as failed (error).
 # Special return statuses TRY_LATER and DEVICE_UNABLE are not errors but merely indicators
   of the current or permanent lack of resources, respectively. However, the CUDA task status
   in these cases will still be set to an error (always check the function return status!).
 # If <gpu_id> is out of the legitimate GPU range, it will be replaced by an optimal one,
   based on argument residence and the current load of GPU(s).
**/
{
 int i,j,drank,lrank,tds_d,tds_l,gpu_d,gpu_l,perm_d,perm_l,ncd,nlu,nru,gpu_num,cur_gpu,targ_dev,bx,errc,stat,conj_l;
 int dprm[1+MAX_TENSOR_RANK],lprm[1+MAX_TENSOR_RANK],rprm[1]; //the 1st element is the sign of the permutation
 size_t vol_d,vol_l,dsize,lsize;
 unsigned int coh;
 const unsigned int TWO_BITS_SET = 3; //two right bits are set
 void *darg,*larg;
 talshComplex4 scale_cmplx4;
 talshComplex8 scale_cmplx8;
 cudaStream_t *cuda_stream;
 cudaEvent_t *cuda_start,*cuda_comput,*cuda_output,*cuda_finish,*dep_event;
#ifdef GPU_FINE_TIMING
 cudaEvent_t *cuda_mmbeg,*cuda_mmend;
#endif
 cudaError_t err;
 const char *err_msg;
#ifdef USE_CUTT
 cuttHandle cutt_d,cutt_l;
 cuttResult cutt_err;
#endif

 //if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): GPU Tensor Addition:\n"); //debug
 stat=0; //return status in case of successful scheduling
//Check function arguments:
 if(cptrn == NULL || dtens == NULL || ltens == NULL || cuda_task == NULL) return -1;
 if(tensBlck_present(dtens) != YEP || tensBlck_present(ltens) != YEP) return -2; //tensor blocks must reside in some device memory
 if(cuda_task_gpu_id(cuda_task) >= 0) return -3; //CUDA task is not clean (destruct/clean it first)
//Check tensor arguments:
 drank=(dtens->shape).num_dim; //destination tensor rank
 lrank=(ltens->shape).num_dim; //left tensor rank
 if(drank < 0 || drank > MAX_TENSOR_RANK ||
    lrank < 0 || lrank > MAX_TENSOR_RANK) return -4;
 if(tens_valid_data_kind(dtens->data_kind,&tds_d) != YEP ||          //tds_d: destination tensor element size in bytes
    tens_valid_data_kind(ltens->data_kind,&tds_l) != YEP) return -5; //tds_l: left tensor element size in bytes
 if(!(dtens->data_kind > 0 && ltens->data_kind == dtens->data_kind)) return -6; //data kind mismatch
 if(dtens->src_rsc == NULL || ltens->src_rsc == NULL) return -7; //source resource must always be present
 if(tensDevRsc_is_empty(dtens->src_rsc) != NOPE) return -8; //source resource must be present (tensor body)
 if(tensDevRsc_is_empty(ltens->src_rsc) != NOPE) return -9; //source resource must be present (tensor body)
//Check the contraction pattern and dimension extent correspondence:
 for(i=0;i<drank;i++) dprm[i]=0; for(i=0;i<lrank;i++) lprm[i]=0;
 for(i=0;i<lrank;i++){ //position in ltens
  j=cptrn[i];
  if(j > 0){ //position in dtens
   if(j > drank) return -11;
   if((dtens->shape).dims[j-1] != (ltens->shape).dims[i]) return -12;
   if(dprm[j-1] == 0){dprm[j-1]=1;}else{return -13;}
  }else{
   return -18;
  }
 }
 for(i=0;i<drank;i++) if(dprm[i] != 1) return -27;
//Check argument complex conjugation bits:
 conj_bits = conj_bits & 3; //keep only first two bits, one per tensor argument {0:D,1:L}
 if(conj_bits & 1){ //destination tensor argument conjugation = inverse conjugation of the left argument
  conj_bits = conj_bits ^ 3; //XOR with 0b111 will invert bits
 }
 conj_l = 0; if((conj_bits & 2) != 0) conj_l = 1; //left argument conjugation flag
//Activate the right GPU:
 if(gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE){gpu_num=tens_op_best_gpu(dtens,ltens);}else{gpu_num=gpu_id;}
 if(gpu_is_mine(gpu_num) <= GPU_OFF) return -28; //GPU is not mine or error
 gpu_stats[gpu_num].tasks_submitted++;
 gpu_d=decode_device_id(dtens->src_rsc->dev_id,&j); if(gpu_d < 0) return -29; //destination tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_d != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_d); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_d=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 gpu_l=decode_device_id(ltens->src_rsc->dev_id,&j); if(gpu_l < 0) return -30; //left tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_l != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_l); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_l=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 cur_gpu=gpu_in_focus(); //save the current GPU
 if(gpu_num != cur_gpu){errc=gpu_activate(gpu_num); if(errc){errc=gpu_activate(cur_gpu); return -32;}} //activate the target GPU
 err=cudaGetLastError(); err=cudaSuccess; //clear the GPU error status
 targ_dev=encode_device_id(DEV_NVIDIA_GPU,gpu_num); //flat device id
//Construct a CUDA task (acquire CUDA resources) for the target GPU:
 errc=cuda_task_construct(cuda_task,gpu_num);
 if(errc){i=gpu_activate(cur_gpu); if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return -33;}}

// *** From this point all error codes must be positive and the CUDA task must be recorded! ***
//Set up tensor arguments (allocates additional resources for each tensor argument):
// Destination argument:
 errc=cuda_task_set_arg(cuda_task,0,dtens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,1); i=gpu_activate(cur_gpu);
   return 1;
  }
 }
// Left argument:
 errc=cuda_task_set_arg(cuda_task,1,ltens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,2); i=gpu_activate(cur_gpu);
   return 2;
  }
 }
//Associate CUDA stream and event pointers locally for convenience:
 cuda_stream=cuda_stream_ptr(cuda_task->gpu_id,cuda_task->stream_hl);
 if(cuda_stream == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,4); errc=gpu_activate(cur_gpu); return 4;}
 cuda_start=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_start_hl);
 if(cuda_start == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,5); errc=gpu_activate(cur_gpu); return 5;}
 cuda_comput=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_comput_hl);
 if(cuda_comput == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,6); errc=gpu_activate(cur_gpu); return 6;}
 cuda_output=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_output_hl);
 if(cuda_output == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,7); errc=gpu_activate(cur_gpu); return 7;}
 cuda_finish=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_finish_hl);
 if(cuda_finish == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
#ifdef GPU_FINE_TIMING
 cuda_mmbeg=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmbeg_hl);
 if(cuda_mmbeg == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
 cuda_mmend=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmend_hl);
 if(cuda_mmend == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
#endif
//Determine the volume and required matricization permutation for each tensor argument:
 get_contr_permutations(lrank,0,cptrn,0,dprm,lprm,rprm,&ncd,&nlu,&nru,&errc); //permutations and numbers of dimensions
 if(errc){i=cuda_task_record(cuda_task,coh_ctrl,9); i=gpu_activate(cur_gpu); return 9;}
 for(i=0;i<drank;i++) cuda_task->tens_args[0].prmn_p[i]=dprm[1+i]; //ignore the permutaion sign
 perm_d=non_trivial_prmn(drank,cuda_task->tens_args[0].prmn_p);    //trivial or not
 for(i=0;i<lrank;i++) cuda_task->tens_args[1].prmn_p[i]=lprm[1+i]; //ignore the permutaion sign
 perm_l=non_trivial_prmn(lrank,cuda_task->tens_args[1].prmn_p);    //trivial or not
 vol_d=tensBlck_volume(dtens); vol_l=tensBlck_volume(ltens);       //tensor block volumes
 dsize=vol_d*tds_d; lsize=vol_l*tds_l;                             //tensor argument sizes in bytes
//Acquire global memory resources for tensor arguments if needed:
// Set up destination memory resources in all tensors:
//  Destination tensor:
 if(dtens->dst_rsc == dtens->src_rsc) dtens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_d != gpu_num){ //data is on a different GPU device or Host
  if(dtens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(dtens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,11); i=gpu_activate(cur_gpu); return 11;}
  }else{
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(dtens->dst_rsc,targ_dev,dsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,12); i=gpu_activate(cur_gpu);
    return 12;
   }
  }
 }else{
  if(dtens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  dtens->dst_rsc=dtens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
//  Left tensor:
 if(ltens->dst_rsc == ltens->src_rsc) ltens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_l != gpu_num){ //data is on a different GPU device or Host
  if(ltens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(ltens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,13); i=gpu_activate(cur_gpu); return 13;}
  }else{
   if(tensDevRsc_is_empty(ltens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(ltens->dst_rsc,targ_dev,lsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,14); i=gpu_activate(cur_gpu);
    return 14;
   }
  }
 }else{
  if(ltens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(ltens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  ltens->dst_rsc=ltens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
// Set up temporary memory resources in all tensors if needed (because of out-of-place tensor transpose):
//  Destination tensor:
 if(perm_d == YEP){
  if(dtens->tmp_rsc == NULL){
   errc=tensDevRsc_create(&(dtens->tmp_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,17); i=gpu_activate(cur_gpu); return 17;}
  }else{
   if(tensDevRsc_is_empty(dtens->tmp_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->tmp_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(dtens->tmp_rsc,targ_dev,dsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,18); i=gpu_activate(cur_gpu);
    return 18;
   }
  }
 }
//  Left tensor:
 if(perm_l == YEP){
  if(ltens->tmp_rsc == NULL){
   errc=tensDevRsc_create(&(ltens->tmp_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,19); i=gpu_activate(cur_gpu); return 19;}
  }else{
   if(tensDevRsc_is_empty(ltens->tmp_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->tmp_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(ltens->tmp_rsc,targ_dev,lsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,20); i=gpu_activate(cur_gpu);
    return 20;
   }
  }
 }
#ifdef DEBUG_GPU
//DEBUG begin:
 if(DEBUG){
  printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_add):\n");
  printf(" Const args (d,l)   : %d %d\n",cuda_task->tens_args[0].const_mem_entry,
                                         cuda_task->tens_args[1].const_mem_entry); //debug
  printf(" Block sizes (d,l)  : %lu %lu\n",dsize,lsize); //debug
  printf(" Block ranks (d,l)  : %d %d\n",dtens->shape.num_dim,ltens->shape.num_dim); //debug
  printf(" Contraction pattern:"); for(i=0;i<(ltens->shape.num_dim);i++) printf(" %d",cptrn[i]); //debug
  printf("\n Contr/uncontr/lens : %d %d %d",ncd,nlu,nru); //debug
  printf("\n D-permutation      :"); for(i=0;i<dtens->shape.num_dim;i++) printf(" %d",cuda_task->tens_args[0].prmn_p[i]); //debug
  printf("\n L-permutation      :"); for(i=0;i<ltens->shape.num_dim;i++) printf(" %d",cuda_task->tens_args[1].prmn_p[i]); //debug
  printf("\n#END OF DEBUG\n");
 }
//DEBUG end.
#endif /*DEBUG_GPU*/
//Start scheduling CUDA calls:
 err=cudaEventRecord(*cuda_start,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to record the start event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,23); errc=gpu_activate(cur_gpu); return 23;
 }
 if(LastTask[gpu_num] != NULL){ //`This should be done atomically for thread safety
  dep_event=cuda_event_ptr(LastTask[gpu_num]->gpu_id,LastTask[gpu_num]->event_comput_hl);
  err=cudaStreamWaitEvent(*cuda_stream,*dep_event,0); //input transfers should only begin after the previous task input transfers have completed
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to create a task dependency: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,24); errc=gpu_activate(cur_gpu); return 24;
  }
 }
//Schedule forward data transfers for all tensors if needed:
// Left tensor:
 if(cuda_task->tens_args[1].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(ltens->shape.dims),sizeof(int)*((size_t)lrank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[1].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Left tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,25); errc=gpu_activate(cur_gpu); return 25;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)lrank);
  if(perm_l == YEP){
   err=cudaMemcpyToSymbolAsync(const_args_prmn,(void*)(cuda_task->tens_args[1].prmn_p),sizeof(int)*((size_t)lrank),
                 sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[1].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor matricization permutation
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Left tensor prmn H2D copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,26); errc=gpu_activate(cur_gpu); return 26;
   }
   gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)lrank);
  }
  if(gpu_l != gpu_num){ //data is not on the computing GPU
   err=cudaMemcpyAsync(ltens->dst_rsc->gmem_p,ltens->src_rsc->gmem_p,lsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Left tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,27); errc=gpu_activate(cur_gpu); return 27;
   }
   gpu_stats[gpu_num].traffic_in+=lsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,28); errc=gpu_activate(cur_gpu); return 28;
 }
// Destination tensor:
 if(cuda_task->tens_args[0].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(dtens->shape.dims),sizeof(int)*((size_t)drank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[0].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Destination tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,33); errc=gpu_activate(cur_gpu); return 33;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)drank);
  if(perm_d == YEP){
   err=cudaMemcpyToSymbolAsync(const_args_prmn,(void*)(cuda_task->tens_args[0].prmn_p),sizeof(int)*((size_t)drank),
                 sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[0].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor matricization permutation
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Destination tensor prmn H2D copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,34); errc=gpu_activate(cur_gpu); return 34;
   }
   gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)drank);
  }
  if(gpu_d != gpu_num){ //data is not on the computing GPU
   err=cudaMemcpyAsync(dtens->dst_rsc->gmem_p,dtens->src_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Destination tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,35); errc=gpu_activate(cur_gpu); return 35;
   }
   gpu_stats[gpu_num].traffic_in+=dsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,36); errc=gpu_activate(cur_gpu); return 36;
 }
//Schedule tensor transposes if needed:
// Record a CUDA event:
 err=cudaEventRecord(*cuda_comput,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to record the compute event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,37); errc=gpu_activate(cur_gpu); return 37;
 }
// Destination tensor transpose (it should not happen actually):
 if(perm_d == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->dst_rsc->gmem_p),(float*)(dtens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->dst_rsc->gmem_p),(double*)(dtens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->dst_rsc->gmem_p),(talshComplex4*)(dtens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->dst_rsc->gmem_p),(talshComplex8*)(dtens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,38); errc=gpu_activate(cur_gpu); return 38;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   for(i=0;i<drank;++i) dprm[i]=cuda_task->tens_args[0].prmn_p[i]-1;
   cutt_err=cuttPlan(&cutt_d,drank,(dtens->shape).dims,dprm,((size_t)tds_d),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_d,dtens->dst_rsc->gmem_p,dtens->tmp_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,63); errc=gpu_activate(cur_gpu); return 63;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,64); errc=gpu_activate(cur_gpu); return 64;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,65); errc=gpu_activate(cur_gpu); return 65;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->dst_rsc->gmem_p),(float*)(dtens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->dst_rsc->gmem_p),(double*)(dtens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->dst_rsc->gmem_p),(talshComplex4*)(dtens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->dst_rsc->gmem_p),(talshComplex8*)(dtens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,39); errc=gpu_activate(cur_gpu); return 39;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,59); errc=gpu_activate(cur_gpu); return 59;
  }
  darg=dtens->tmp_rsc->gmem_p;
 }else{
  darg=dtens->dst_rsc->gmem_p;
 }
// Left tensor transpose:
 if(perm_l == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_l-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(ltens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(float*)(ltens->dst_rsc->gmem_p),(float*)(ltens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(double*)(ltens->dst_rsc->gmem_p),(double*)(ltens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex4*)(ltens->dst_rsc->gmem_p),(talshComplex4*)(ltens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex8*)(ltens->dst_rsc->gmem_p),(talshComplex8*)(ltens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,40); errc=gpu_activate(cur_gpu); return 40;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   errc=prmn_convert(lrank,cuda_task->tens_args[1].prmn_p,lprm); for(i=0;i<lrank;++i) --(lprm[i]);
   cutt_err=cuttPlan(&cutt_l,lrank,(ltens->shape).dims,lprm,((size_t)tds_l),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_l,ltens->dst_rsc->gmem_p,ltens->tmp_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,66); errc=gpu_activate(cur_gpu); return 66;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,67); errc=gpu_activate(cur_gpu); return 67;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,68); errc=gpu_activate(cur_gpu); return 68;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_l-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(ltens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(float*)(ltens->dst_rsc->gmem_p),(float*)(ltens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(double*)(ltens->dst_rsc->gmem_p),(double*)(ltens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex4*)(ltens->dst_rsc->gmem_p),(talshComplex4*)(ltens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex8*)(ltens->dst_rsc->gmem_p),(talshComplex8*)(ltens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,41); errc=gpu_activate(cur_gpu); return 41;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,60); errc=gpu_activate(cur_gpu); return 60;
  }
  larg=ltens->tmp_rsc->gmem_p;
 }else{
  larg=ltens->dst_rsc->gmem_p;
 }
//Schedule the appropriate computation kernel:
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmbeg,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to record the mmbeg event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,66); errc=gpu_activate(cur_gpu); return 66;
 }
#endif
// Addition kernel:
 bx=1+(vol_d-1)/THRDS_ARRAY_ADD; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
 switch(dtens->data_kind){
  case R4:
   gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(float*)darg,(float*)larg,(float)scale_real);
   gpu_stats[gpu_num].flops+=2.0*((double)(dsize)); //1 mul, 1 add SP
   break;
  case R8:
   gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(double*)darg,(double*)larg,scale_real);
   gpu_stats[gpu_num].flops+=2.0*((double)(dsize)); //1 mul, 1 add DP
   break;
  case C4:
   scale_cmplx4 = talshComplex4Set((float)scale_real,(float)scale_imag);
   gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(talshComplex4*)darg,(talshComplex4*)larg,scale_cmplx4,conj_l);
   gpu_stats[gpu_num].flops+=8.0*((double)(dsize)); //4 mul, 4 add SP
   break;
  case C8:
   scale_cmplx8 = talshComplex8Set(scale_real,scale_imag);
   gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(talshComplex8*)darg,(talshComplex8*)larg,scale_cmplx8,conj_l);
   gpu_stats[gpu_num].flops+=8.0*((double)(dsize)); //4 mul, 4 add DP
   break;
  default:
   errc=cuda_task_record(cuda_task,coh_ctrl,48); errc=gpu_activate(cur_gpu); return 48;
 }
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmend,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to record the mmend event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,67); errc=gpu_activate(cur_gpu); return 67;
 }
#endif
//Schedule the inverse tensor transpose for the destination tensor (should not happen actually):
 if(perm_d == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->tmp_rsc->gmem_p),(float*)(dtens->dst_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->tmp_rsc->gmem_p),(double*)(dtens->dst_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->tmp_rsc->gmem_p),(talshComplex4*)(dtens->dst_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->tmp_rsc->gmem_p),(talshComplex8*)(dtens->dst_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,54); errc=gpu_activate(cur_gpu); return 54;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   errc=prmn_convert(drank,cuda_task->tens_args[0].prmn_p,dprm); for(i=0;i<drank;++i) --(dprm[i]);
   for(i=0;i<drank;++i) rprm[i]=(dtens->shape).dims[drank-i-1]; //inversed dimension order
   cutt_err=cuttPlan(&cutt_d,drank,rprm,dprm,((size_t)tds_d),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_d,dtens->tmp_rsc->gmem_p,dtens->dst_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,63); errc=gpu_activate(cur_gpu); return 63;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,64); errc=gpu_activate(cur_gpu); return 64;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,65); errc=gpu_activate(cur_gpu); return 65;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->tmp_rsc->gmem_p),(float*)(dtens->dst_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->tmp_rsc->gmem_p),(double*)(dtens->dst_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->tmp_rsc->gmem_p),(talshComplex4*)(dtens->dst_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->tmp_rsc->gmem_p),(talshComplex8*)(dtens->dst_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,55); errc=gpu_activate(cur_gpu); return 55;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,62); errc=gpu_activate(cur_gpu); return 62;
  }
 }
//Record a CUDA event (output ready on GPU):
 err=cudaEventRecord(*cuda_output,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to record the output event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,56); errc=gpu_activate(cur_gpu); return 56;
 }
//Transfer back the updated destination tensor if needed ("T","K" coherence control):
 coh=(coh_ctrl>>2)&(TWO_BITS_SET); //select bits 2,3 (destination tensor coherence)
 if(gpu_d != gpu_num && coh >= 2){ //data is not on the computing GPU and coherence control = 2("T") or (3)"K":
  err=cudaMemcpyAsync(dtens->src_rsc->gmem_p,dtens->dst_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Destination tensor body back copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,57); errc=gpu_activate(cur_gpu); return 57;
  }
  gpu_stats[gpu_num].traffic_out+=dsize;
 }
//Record a CUDA event (task finished):
 err=cudaEventRecord(*cuda_finish,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to record the finish event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,58); errc=gpu_activate(cur_gpu); return 58;
 }
//Record the successfully scheduled CUDA task and update the Last Task:
 errc=cuda_task_record(cuda_task,coh_ctrl,0);
 LastTask[gpu_num]=cuda_task;
 if(gpu_num != cur_gpu) errc=gpu_activate(cur_gpu);
 return stat; //either 0 (success) or NOT_CLEAN (warning)
}
//-------------------------------------------------------------------------------------------------------------------
// TENSOR CONTRACTION (non-blocking):
__host__ int gpu_tensor_block_contract_dlf(const int *cptrn, tensBlck_t *ltens, tensBlck_t *rtens, tensBlck_t *dtens,
                                           unsigned int coh_ctrl, cudaTask_t *cuda_task, int gpu_id,
                                           double scale_real, double scale_imag, int conj_bits)
/**
dtens(:)+=ltens(:)*rtens(:)
INPUT:
 # cptrn(1:lrank+rrank) - contraction pattern: Position correspondence:
                          Uncontracted indices are positive, contracted are negative;
 # ltens - left tensor argument (initialized!);
 # rtens - right tensor argument (initialized!);
 # dtens - destination tensor argument (initialized!);
 # coh_ctrl - one of the COPY_XXX parameters regulating the data presence for each tensor argument;
 # cuda_task - pointer to an empty (clean) CUDA task;
 # gpu_id - suggested GPU ID on which the operation is to be scheduled (-1: defaults to the optimal one);
 # scale_real - real part of the GEMM alpha coefficient (defaults to 1.0);
 # scale_imag - imaginary part of the GEMM alpha coefficient (defaults to 0.0);
 # conj_bits - tensor argument complex conjugation bits, one bit per argument: {0:D,1:L,2:R};
OUTPUT:
 # dtens - updated destination tensor;
 # cuda_task - recorded CUDA task (either successfully scheduled or failed).
NOTES:
 # If the tensor operation has been scheduled successfully, a recorded (active) CUDA task
   will be returned along with zero return status. A scheduling error results in either
   a negative (at early stages) or positive (at later stages) return status. In the former case
   the CUDA task is left clean, while at the latter case it will be recorded as failed (error).
 # Special return statuses TRY_LATER and DEVICE_UNABLE are not errors but merely indicators
   of the current or permanent lack of resources, respectively. However, the CUDA task status
   in these cases will still be set to an error (always check the function return status!).
 # If <gpu_id> is out of the legitimate GPU range, it will be replaced by an optimal one,
   based on argument residence and the current load of GPU(s).
**/
{
 int i,j,drank,lrank,rrank,tds_d,tds_l,tds_r,gpu_d,gpu_l,gpu_r,perm_d,perm_l,perm_r;
 int ncd,nlu,nru,gpu_num,cur_gpu,targ_dev,bx,by,errc,stat,conj_l,conj_r;
 int dprm[1+MAX_TENSOR_RANK],lprm[1+MAX_TENSOR_RANK],rprm[1+MAX_TENSOR_RANK]; //the 1st element is the sign of the permutation
 size_t vol_d,vol_l,vol_r,dsize,lsize,rsize,lc,ll,lr,pofs;
 unsigned int coh;
 const unsigned int TWO_BITS_SET = 3; //two right bits are set
 void *darg,*larg,*rarg,*alpha_p,*beta_p;
 talshComplex4 scale_cmplx4;
 talshComplex8 scale_cmplx8;
 cudaStream_t *cuda_stream;
 cudaEvent_t *cuda_start,*cuda_comput,*cuda_output,*cuda_finish,*dep_event;
#ifdef GPU_FINE_TIMING
 cudaEvent_t *cuda_mmbeg,*cuda_mmend;
#endif
 cudaError_t err;
 const char *err_msg;
#ifndef NO_BLAS
 cublasStatus_t err_cublas;
 cublasOperation_t left_conj,right_conj;
#endif
#ifdef USE_CUTT
 cuttHandle cutt_d,cutt_l,cutt_r;
 cuttResult cutt_err;
#endif

 //if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): GPU Tensor Contraction:\n"); //debug
 stat=0; //return status in case of successful scheduling
//Check function arguments:
 if(cptrn == NULL || dtens == NULL || ltens == NULL || rtens == NULL || cuda_task == NULL) return -1;
 if(tensBlck_present(dtens) != YEP || tensBlck_present(ltens) != YEP || tensBlck_present(rtens) != YEP) return -2; //tensor blocks must reside in some device memory
 if(cuda_task_gpu_id(cuda_task) >= 0) return -3; //CUDA task is not clean (destruct/clean it first)
//Check tensor arguments:
 drank=(dtens->shape).num_dim; //destination tensor rank
 lrank=(ltens->shape).num_dim; //left tensor rank
 rrank=(rtens->shape).num_dim; //right tensor rank
 if(drank < 0 || drank > MAX_TENSOR_RANK ||
    lrank < 0 || lrank > MAX_TENSOR_RANK ||
    rrank < 0 || rrank > MAX_TENSOR_RANK) return -4;
 if(tens_valid_data_kind(dtens->data_kind,&tds_d) != YEP ||          //tds_d: destination tensor element size in bytes
    tens_valid_data_kind(ltens->data_kind,&tds_l) != YEP ||          //tds_l: left tensor element size in bytes
    tens_valid_data_kind(rtens->data_kind,&tds_r) != YEP) return -5; //tds_r: right tensor element size in bytes
 if(!(dtens->data_kind > 0 && ltens->data_kind == dtens->data_kind && rtens->data_kind == dtens->data_kind)) return -6; //data kind mismatch
 if(dtens->src_rsc == NULL || ltens->src_rsc == NULL || rtens->src_rsc == NULL) return -7; //source resource must always be present
 if(tensDevRsc_is_empty(dtens->src_rsc) != NOPE) return -8;  //source resource must be present (tensor body)
 if(tensDevRsc_is_empty(ltens->src_rsc) != NOPE) return -9;  //source resource must be present (tensor body)
 if(tensDevRsc_is_empty(rtens->src_rsc) != NOPE) return -10; //source resource must be present (tensor body)
//Check the contraction pattern and tensor dimension extent correspondence:
 for(i=0;i<drank;i++) dprm[i]=0; for(i=0;i<lrank;i++) lprm[i]=0; for(i=0;i<rrank;i++) rprm[i]=0;
 for(i=0;i<lrank;i++){ //position in ltens
  j=cptrn[i];
  if(j > 0){ //position in dtens
   if(j > drank) return -11;
   if((dtens->shape).dims[j-1] != (ltens->shape).dims[i]) return -12;
   if(dprm[j-1] == 0){dprm[j-1]=1;}else{return -13;}
  }else if(j < 0){ //position in rtens
   if(-j > rrank) return -14;
   if((rtens->shape).dims[-j-1] != (ltens->shape).dims[i]) return -15;
   if(cptrn[lrank+(-j-1)] != -(i+1)) return -16;
   if(rprm[-j-1] == 0){rprm[-j-1]=1;}else{return -17;}
  }else{
   return -18;
  }
 }
 for(i=0;i<rrank;i++){ //position in rtens
  j=cptrn[lrank+i];
  if(j > 0){ //position in dtens
   if(j > drank) return -19;
   if((dtens->shape).dims[j-1] != (rtens->shape).dims[i]) return -20;
   if(dprm[j-1] == 0){dprm[j-1]=1;}else{return -21;}
  }else if(j < 0){ //position in ltens
   if(-j > lrank) return -22;
   if((ltens->shape).dims[-j-1] != (rtens->shape).dims[i]) return -23;
   if(cptrn[-j-1] != -(i+1)) return -24;
   if(lprm[-j-1] == 0){lprm[-j-1]=1;}else{return -25;}
  }else{
   return -26;
  }
 }
 for(i=0;i<drank;i++) if(dprm[i] != 1) return -27;
//Check argument complex conjugation bits:
#ifndef NO_BLAS
 left_conj=CUBLAS_OP_T; right_conj=CUBLAS_OP_N; //default is TN GEMM
#endif
 conj_bits = conj_bits & 7; //keep only first three bits, one per tensor argument {0:D,1:L,2:R}
 if(conj_bits & 1){ //destination tensor argument conjugation = inverse conjugation of left and right tensor arguments
  conj_bits = conj_bits ^ 7; //XOR with 0b111 will invert bits
 }
 if(dtens->data_kind == C4 || dtens->data_kind == C8){ //conjugation may apply to complex data kinds
  conj_l = 0; if((conj_bits & 2) != 0) conj_l=1; //left tensor argument conjugation flag
  conj_r = 0; if((conj_bits & 4) != 0) conj_r=1; //right tensor argument conjugation flag
#ifndef NO_BLAS
  if(conj_l != 0) left_conj = CUBLAS_OP_C;
  if(conj_r != 0) right_conj = CUBLAS_OP_C;
#endif
 }else{
  conj_bits = 0; conj_l = 0; conj_r = 0; //no conjugation for real data kinds
 }
//Activate the right GPU:
 if(gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE){gpu_num=tens_op_best_gpu(dtens,ltens,rtens);}else{gpu_num=gpu_id;}
 if(gpu_is_mine(gpu_num) <= GPU_OFF) return -28; //GPU is not mine or error
 gpu_stats[gpu_num].tasks_submitted++;
 gpu_d=decode_device_id(dtens->src_rsc->dev_id,&j); if(gpu_d < 0) return -29; //destination tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_d != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_d); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_d=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 gpu_l=decode_device_id(ltens->src_rsc->dev_id,&j); if(gpu_l < 0) return -30; //left tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_l != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_l); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_l=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 gpu_r=decode_device_id(rtens->src_rsc->dev_id,&j); if(gpu_r < 0) return -31; //right tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_r != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_r); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_r=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 cur_gpu=gpu_in_focus(); //save the current GPU
 if(gpu_num != cur_gpu){errc=gpu_activate(gpu_num); if(errc){errc=gpu_activate(cur_gpu); return -32;}} //activate the target GPU
 err=cudaGetLastError(); err=cudaSuccess; //clear the GPU error status
 targ_dev=encode_device_id(DEV_NVIDIA_GPU,gpu_num); //flat device id
//Construct a CUDA task (acquire CUDA resources) for the target GPU:
 errc=cuda_task_construct(cuda_task,gpu_num);
 if(errc){i=gpu_activate(cur_gpu); if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return -33;}}

// *** From this point all error codes must be positive and the CUDA task must be recorded! ***
//Set up tensor arguments (allocates additional resources for each tensor argument):
// Destination argument:
 errc=cuda_task_set_arg(cuda_task,0,dtens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,1); i=gpu_activate(cur_gpu);
   return 1;
  }
 }
// Left argument:
 errc=cuda_task_set_arg(cuda_task,1,ltens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,2); i=gpu_activate(cur_gpu);
   return 2;
  }
 }
// Right argument:
 errc=cuda_task_set_arg(cuda_task,2,rtens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,3); i=gpu_activate(cur_gpu);
   return 3;
  }
 }
//Associate CUDA stream and event pointers locally for convenience:
 cuda_stream=cuda_stream_ptr(cuda_task->gpu_id,cuda_task->stream_hl);
 if(cuda_stream == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,4); errc=gpu_activate(cur_gpu); return 4;}
 cuda_start=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_start_hl);
 if(cuda_start == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,5); errc=gpu_activate(cur_gpu); return 5;}
 cuda_comput=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_comput_hl);
 if(cuda_comput == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,6); errc=gpu_activate(cur_gpu); return 6;}
 cuda_output=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_output_hl);
 if(cuda_output == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,7); errc=gpu_activate(cur_gpu); return 7;}
 cuda_finish=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_finish_hl);
 if(cuda_finish == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
#ifdef GPU_FINE_TIMING
 cuda_mmbeg=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmbeg_hl);
 if(cuda_mmbeg == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,9); errc=gpu_activate(cur_gpu); return 9;}
 cuda_mmend=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmend_hl);
 if(cuda_mmend == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,10); errc=gpu_activate(cur_gpu); return 10;}
#endif
//Determine the volume and required matricization permutation for each tensor argument:
 if(drank > 0 && lrank > 0 && rrank > 0 && drank < (lrank + rrank)){ //GEMM mapped tensor contraction: {TN,NT,NN,TT}
  get_contr_permutations(lrank,rrank,cptrn,conj_bits,dprm,lprm,rprm,&ncd,&nlu,&nru,&errc); //permutations and numbers of dimensions
 }else{ //custom kernel mapped tensor contraction (complex conjugation does not require modified permutations)
  get_contr_permutations(lrank,rrank,cptrn,0,dprm,lprm,rprm,&ncd,&nlu,&nru,&errc); //permutations and numbers of dimensions
 }
 if(errc){i=cuda_task_record(cuda_task,coh_ctrl,11); i=gpu_activate(cur_gpu); return 11;}
 for(i=0;i<drank;i++) cuda_task->tens_args[0].prmn_p[i]=dprm[1+i]; //ignore the permutaion sign
 perm_d=non_trivial_prmn(drank,cuda_task->tens_args[0].prmn_p);    //trivial or not
 for(i=0;i<lrank;i++) cuda_task->tens_args[1].prmn_p[i]=lprm[1+i]; //ignore the permutaion sign
 perm_l=non_trivial_prmn(lrank,cuda_task->tens_args[1].prmn_p);    //trivial or not
 for(i=0;i<rrank;i++) cuda_task->tens_args[2].prmn_p[i]=rprm[1+i]; //ignore the permutaion sign
 perm_r=non_trivial_prmn(rrank,cuda_task->tens_args[2].prmn_p);    //trivial or not
 vol_d=tensBlck_volume(dtens); vol_l=tensBlck_volume(ltens); vol_r=tensBlck_volume(rtens); //tensor block volumes
 lc=1; ll=1;
 for(i=0;i<lrank;i++){
  if(cuda_task->tens_args[1].prmn_p[i] <= ncd){lc*=((ltens->shape).dims[i]);}else{ll*=((ltens->shape).dims[i]);}
 }
 lr=vol_d/ll;
 if(vol_l <= 0 || vol_r <= 0 || vol_d <= 0 || vol_d%ll != 0 || vol_r%lr != 0 || vol_r/lr != lc){
  i=cuda_task_record(cuda_task,coh_ctrl,12); i=gpu_activate(cur_gpu); return 12; //invalid matrix dimensions obtained
 }
 dsize=vol_d*tds_d; lsize=vol_l*tds_l; rsize=vol_r*tds_r; //tensor argument sizes in bytes
//Acquire global memory resources for tensor arguments if needed:
// Set up destination memory resources in all tensors:
//  Destination tensor:
 if(dtens->dst_rsc == dtens->src_rsc) dtens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_d != gpu_num){ //data is on a different GPU device or Host
  if(dtens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(dtens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,13); i=gpu_activate(cur_gpu); return 13;}
  }else{
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(dtens->dst_rsc,targ_dev,dsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,14); i=gpu_activate(cur_gpu);
    return 14;
   }
  }
 }else{
  if(dtens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  dtens->dst_rsc=dtens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
//  Left tensor:
 if(ltens->dst_rsc == ltens->src_rsc) ltens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_l != gpu_num){ //data is on a different GPU device or Host
  if(ltens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(ltens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,15); i=gpu_activate(cur_gpu); return 15;}
  }else{
   if(tensDevRsc_is_empty(ltens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(ltens->dst_rsc,targ_dev,lsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,16); i=gpu_activate(cur_gpu);
    return 16;
   }
  }
 }else{
  if(ltens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(ltens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  ltens->dst_rsc=ltens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
//  Right tensor:
 if(rtens->dst_rsc == rtens->src_rsc) rtens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_r != gpu_num){ //data is on a different GPU device or Host
  if(rtens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(rtens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,17); i=gpu_activate(cur_gpu); return 17;}
  }else{
   if(tensDevRsc_is_empty(rtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(rtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(rtens->dst_rsc,targ_dev,rsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,18); i=gpu_activate(cur_gpu);
    return 18;
   }
  }
 }else{
  if(rtens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(rtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(rtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  rtens->dst_rsc=rtens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
// Set up temporary memory resources in all tensors if needed (because of out-of-place tensor transpose):
//  Destination tensor:
 if(perm_d == YEP){
  if(dtens->tmp_rsc == NULL){
   errc=tensDevRsc_create(&(dtens->tmp_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,19); i=gpu_activate(cur_gpu); return 19;}
  }else{
   if(tensDevRsc_is_empty(dtens->tmp_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->tmp_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(dtens->tmp_rsc,targ_dev,dsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,20); i=gpu_activate(cur_gpu);
    return 20;
   }
  }
 }
//  Left tensor:
 if(perm_l == YEP){
  if(ltens->tmp_rsc == NULL){
   errc=tensDevRsc_create(&(ltens->tmp_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,21); i=gpu_activate(cur_gpu); return 21;}
  }else{
   if(tensDevRsc_is_empty(ltens->tmp_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->tmp_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(ltens->tmp_rsc,targ_dev,lsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,22); i=gpu_activate(cur_gpu);
    return 22;
   }
  }
 }
//  Right tensor:
 if(perm_r == YEP){
  if(rtens->tmp_rsc == NULL){
   errc=tensDevRsc_create(&(rtens->tmp_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,23); i=gpu_activate(cur_gpu); return 23;}
  }else{
   if(tensDevRsc_is_empty(rtens->tmp_rsc) == NOPE){errc=tensDevRsc_release_all(rtens->tmp_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(rtens->tmp_rsc,targ_dev,rsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,24); i=gpu_activate(cur_gpu);
    return 24;
   }
  }
 }
#ifdef DEBUG_GPU
//DEBUG begin:
 if(DEBUG){
  printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf):\n");
  printf(" Const args (d,l,r) : %d %d %d\n",cuda_task->tens_args[0].const_mem_entry,
                                            cuda_task->tens_args[1].const_mem_entry,
                                            cuda_task->tens_args[2].const_mem_entry); //debug
  printf(" Block sizes (d,l,r): %lu %lu %lu\n",dsize,lsize,rsize); //debug
  printf(" Block ranks (d,l,r): %d %d %d\n",dtens->shape.num_dim,ltens->shape.num_dim,rtens->shape.num_dim); //debug
  printf(" Contraction pattern:"); for(i=0;i<(ltens->shape.num_dim+rtens->shape.num_dim);i++) printf(" %d",cptrn[i]); //debug
  printf("\n Contr/uncontr/lens : %d %d %d: %lu %lu %lu\n",ncd,nlu,nru,lc,ll,lr); //debug
  printf(" D-permutation      :"); for(i=0;i<dtens->shape.num_dim;i++) printf(" %d",cuda_task->tens_args[0].prmn_p[i]); //debug
  printf("\n L-permutation      :"); for(i=0;i<ltens->shape.num_dim;i++) printf(" %d",cuda_task->tens_args[1].prmn_p[i]); //debug
  printf("\n R-permutation      :"); for(i=0;i<rtens->shape.num_dim;i++) printf(" %d",cuda_task->tens_args[2].prmn_p[i]); //debug
  printf("\n#END OF DEBUG\n");
 }
//DEBUG end.
#endif /*DEBUG_GPU*/
//Start scheduling CUDA calls:
 err=cudaEventRecord(*cuda_start,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to record the start event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,25); errc=gpu_activate(cur_gpu); return 25;
 }
 if(LastTask[gpu_num] != NULL){ //`This should be done atomically for thread safety
  dep_event=cuda_event_ptr(LastTask[gpu_num]->gpu_id,LastTask[gpu_num]->event_comput_hl);
  err=cudaStreamWaitEvent(*cuda_stream,*dep_event,0); //input transfers should only begin after the previous task input transfers have completed
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to create a task dependency: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,26); errc=gpu_activate(cur_gpu); return 26;
  }
 }
//Schedule forward data transfers for all tensors if needed:
// Left tensor:
 if(cuda_task->tens_args[1].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(ltens->shape.dims),sizeof(int)*((size_t)lrank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[1].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Left tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,27); errc=gpu_activate(cur_gpu); return 27;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)lrank);
  if(perm_l == YEP){
   err=cudaMemcpyToSymbolAsync(const_args_prmn,(void*)(cuda_task->tens_args[1].prmn_p),sizeof(int)*((size_t)lrank),
                 sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[1].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor matricization permutation
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Left tensor prmn H2D copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,28); errc=gpu_activate(cur_gpu); return 28;
   }
   gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)lrank);
  }
  if(gpu_l != gpu_num){ //data is not on the computing GPU
   err=cudaMemcpyAsync(ltens->dst_rsc->gmem_p,ltens->src_rsc->gmem_p,lsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Left tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,29); errc=gpu_activate(cur_gpu); return 29;
   }
   gpu_stats[gpu_num].traffic_in+=lsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,30); errc=gpu_activate(cur_gpu); return 30;
 }
// Right tensor:
 if(cuda_task->tens_args[2].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(rtens->shape.dims),sizeof(int)*((size_t)rrank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[2].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Right tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,31); errc=gpu_activate(cur_gpu); return 31;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)rrank);
  if(perm_r == YEP){
   err=cudaMemcpyToSymbolAsync(const_args_prmn,(void*)(cuda_task->tens_args[2].prmn_p),sizeof(int)*((size_t)rrank),
                 sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[2].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor matricization permutation
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Right tensor prmn H2D copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,32); errc=gpu_activate(cur_gpu); return 32;
   }
   gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)rrank);
  }
  if(gpu_r != gpu_num){ //data is not on the computing GPU
   err=cudaMemcpyAsync(rtens->dst_rsc->gmem_p,rtens->src_rsc->gmem_p,rsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Right tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,33); errc=gpu_activate(cur_gpu); return 33;
   }
   gpu_stats[gpu_num].traffic_in+=rsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,34); errc=gpu_activate(cur_gpu); return 34;
 }
// Destination tensor:
 if(cuda_task->tens_args[0].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(dtens->shape.dims),sizeof(int)*((size_t)drank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[0].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Destination tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,35); errc=gpu_activate(cur_gpu); return 35;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)drank);
  if(perm_d == YEP){
   err=cudaMemcpyToSymbolAsync(const_args_prmn,(void*)(cuda_task->tens_args[0].prmn_p),sizeof(int)*((size_t)drank),
                 sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[0].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor matricization permutation
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Destination tensor prmn H2D copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,36); errc=gpu_activate(cur_gpu); return 36;
   }
   gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)drank);
  }
  if(gpu_d != gpu_num){ //data is not on the computing GPU
   err=cudaMemcpyAsync(dtens->dst_rsc->gmem_p,dtens->src_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Destination tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,37); errc=gpu_activate(cur_gpu); return 37;
   }
   gpu_stats[gpu_num].traffic_in+=dsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,38); errc=gpu_activate(cur_gpu); return 38;
 }
//Schedule tensor transposes if needed:
// Record a CUDA event:
 err=cudaEventRecord(*cuda_comput,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to record the compute event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,39); errc=gpu_activate(cur_gpu); return 39;
 }
// Destination tensor transpose:
 if(perm_d == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->dst_rsc->gmem_p),(float*)(dtens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->dst_rsc->gmem_p),(double*)(dtens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->dst_rsc->gmem_p),(talshComplex4*)(dtens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->dst_rsc->gmem_p),(talshComplex8*)(dtens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,40); errc=gpu_activate(cur_gpu); return 40;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   for(i=0;i<drank;++i) dprm[i]=cuda_task->tens_args[0].prmn_p[i]-1;
   cutt_err=cuttPlan(&cutt_d,drank,(dtens->shape).dims,dprm,((size_t)tds_d),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_d,dtens->dst_rsc->gmem_p,dtens->tmp_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,41); errc=gpu_activate(cur_gpu); return 41;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,42); errc=gpu_activate(cur_gpu); return 42;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,43); errc=gpu_activate(cur_gpu); return 43;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->dst_rsc->gmem_p),(float*)(dtens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->dst_rsc->gmem_p),(double*)(dtens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->dst_rsc->gmem_p),(talshComplex4*)(dtens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->dst_rsc->gmem_p),(talshComplex8*)(dtens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,44); errc=gpu_activate(cur_gpu); return 44;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,45); errc=gpu_activate(cur_gpu); return 45;
  }
  darg=dtens->tmp_rsc->gmem_p;
 }else{
  darg=dtens->dst_rsc->gmem_p;
 }
// Left tensor transpose:
 if(perm_l == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_l-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(ltens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(float*)(ltens->dst_rsc->gmem_p),(float*)(ltens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(double*)(ltens->dst_rsc->gmem_p),(double*)(ltens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex4*)(ltens->dst_rsc->gmem_p),(talshComplex4*)(ltens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex8*)(ltens->dst_rsc->gmem_p),(talshComplex8*)(ltens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,46); errc=gpu_activate(cur_gpu); return 46;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   errc=prmn_convert(lrank,cuda_task->tens_args[1].prmn_p,lprm); for(i=0;i<lrank;++i) --(lprm[i]);
   cutt_err=cuttPlan(&cutt_l,lrank,(ltens->shape).dims,lprm,((size_t)tds_l),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_l,ltens->dst_rsc->gmem_p,ltens->tmp_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,47); errc=gpu_activate(cur_gpu); return 47;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,48); errc=gpu_activate(cur_gpu); return 48;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,49); errc=gpu_activate(cur_gpu); return 49;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_l-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(ltens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(float*)(ltens->dst_rsc->gmem_p),(float*)(ltens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(double*)(ltens->dst_rsc->gmem_p),(double*)(ltens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex4*)(ltens->dst_rsc->gmem_p),(talshComplex4*)(ltens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex8*)(ltens->dst_rsc->gmem_p),(talshComplex8*)(ltens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,50); errc=gpu_activate(cur_gpu); return 50;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,51); errc=gpu_activate(cur_gpu); return 51;
  }
  larg=ltens->tmp_rsc->gmem_p;
 }else{
  larg=ltens->dst_rsc->gmem_p;
 }
// Right tensor transpose:
 if(perm_r == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_r-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(rtens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,(float*)(rtens->dst_rsc->gmem_p),(float*)(rtens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,(double*)(rtens->dst_rsc->gmem_p),(double*)(rtens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,
       (talshComplex4*)(rtens->dst_rsc->gmem_p),(talshComplex4*)(rtens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,
       (talshComplex8*)(rtens->dst_rsc->gmem_p),(talshComplex8*)(rtens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,52); errc=gpu_activate(cur_gpu); return 52;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   errc=prmn_convert(rrank,cuda_task->tens_args[2].prmn_p,rprm); for(i=0;i<rrank;++i) --(rprm[i]);
   cutt_err=cuttPlan(&cutt_r,rrank,(rtens->shape).dims,rprm,((size_t)tds_r),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_r,rtens->dst_rsc->gmem_p,rtens->tmp_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,53); errc=gpu_activate(cur_gpu); return 53;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,54); errc=gpu_activate(cur_gpu); return 54;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,55); errc=gpu_activate(cur_gpu); return 55;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_r-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(rtens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,(float*)(rtens->dst_rsc->gmem_p),(float*)(rtens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,(double*)(rtens->dst_rsc->gmem_p),(double*)(rtens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,
       (talshComplex4*)(rtens->dst_rsc->gmem_p),(talshComplex4*)(rtens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,
       (talshComplex8*)(rtens->dst_rsc->gmem_p),(talshComplex8*)(rtens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,56); errc=gpu_activate(cur_gpu); return 56;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,57); errc=gpu_activate(cur_gpu); return 57;
  }
  rarg=rtens->tmp_rsc->gmem_p;
 }else{
  rarg=rtens->dst_rsc->gmem_p;
 }
//Schedule the appropriate computation kernel:
// Set up the prefactor (in mapped Host memory):
 errc=0;
 switch(dtens->data_kind){
  case R4:
   if(scale_real != 1.0 || scale_imag != 0.0){
    errc=cuda_task_set_prefactor(cuda_task,talshComplex4Set((float)scale_real,(float)scale_imag));
    if(errc){j=cuda_task_record(cuda_task,coh_ctrl,58); j=gpu_activate(cur_gpu); return 58;}
    j=slab_get_entry_offset(&prefactors,cuda_task->pref_ptr,&pofs); if(j != 0) errc++;
    alpha_p=(void*)&(((char*)(gpu_prefs_base_ptr))[pofs]);
   }else{
    err=cudaGetSymbolAddress(&alpha_p,sgemm_alpha); if(err != cudaSuccess) errc++;
   }
   err=cudaGetSymbolAddress(&beta_p,sgemm_beta); if(err != cudaSuccess) errc++;
   break;
  case R8:
   if(scale_real != 1.0 || scale_imag != 0.0){
    errc=cuda_task_set_prefactor(cuda_task,talshComplex8Set(scale_real,scale_imag));
    if(errc){j=cuda_task_record(cuda_task,coh_ctrl,59); j=gpu_activate(cur_gpu); return 59;}
    j=slab_get_entry_offset(&prefactors,cuda_task->pref_ptr,&pofs); if(j != 0) errc++;
    alpha_p=(void*)&(((char*)(gpu_prefs_base_ptr))[pofs]);
   }else{
    err=cudaGetSymbolAddress(&alpha_p,dgemm_alpha); if(err != cudaSuccess) errc++;
   }
   err=cudaGetSymbolAddress(&beta_p,dgemm_beta); if(err != cudaSuccess) errc++;
   break;
  case C4:
   if(scale_real != 1.0 || scale_imag != 0.0){
    errc=cuda_task_set_prefactor(cuda_task,talshComplex4Set((float)scale_real,(float)scale_imag));
    if(errc){j=cuda_task_record(cuda_task,coh_ctrl,60); j=gpu_activate(cur_gpu); return 60;}
    j=slab_get_entry_offset(&prefactors,cuda_task->pref_ptr,&pofs); if(j != 0) errc++;
    alpha_p=(void*)&(((char*)(gpu_prefs_base_ptr))[pofs]);
   }else{
    err=cudaGetSymbolAddress(&alpha_p,cgemm_alpha); if(err != cudaSuccess) errc++;
   }
   err=cudaGetSymbolAddress(&beta_p,cgemm_beta); if(err != cudaSuccess) errc++;
   break;
  case C8:
   if(scale_real != 1.0 || scale_imag != 0.0){
    errc=cuda_task_set_prefactor(cuda_task,talshComplex8Set(scale_real,scale_imag));
    if(errc){j=cuda_task_record(cuda_task,coh_ctrl,61); j=gpu_activate(cur_gpu); return 61;}
    j=slab_get_entry_offset(&prefactors,cuda_task->pref_ptr,&pofs); if(j != 0) errc++;
    alpha_p=(void*)&(((char*)(gpu_prefs_base_ptr))[pofs]);
   }else{
    err=cudaGetSymbolAddress(&alpha_p,zgemm_alpha); if(err != cudaSuccess) errc++;
   }
   err=cudaGetSymbolAddress(&beta_p,zgemm_beta); if(err != cudaSuccess) errc++;
   break;
  default:
   errc++;
 }
 if(errc){errc=cuda_task_record(cuda_task,coh_ctrl,62); errc=gpu_activate(cur_gpu); return 62;}
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmbeg,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to record the mmbeg event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,63); errc=gpu_activate(cur_gpu); return 63;
 }
#endif
// Scalar multiplication:
 if(drank == 0 && lrank == 0 && rrank == 0){
  switch(dtens->data_kind){
   case R4:
    gpu_scalar_multiply__<<<1,1,0,*cuda_stream>>>((float*)larg,(float*)rarg,(float*)darg,(float)scale_real);
    break;
   case R8:
    gpu_scalar_multiply__<<<1,1,0,*cuda_stream>>>((double*)larg,(double*)rarg,(double*)darg,scale_real);
    break;
   case C4:
    scale_cmplx4 = talshComplex4Set((float)scale_real,(float)scale_imag);
    gpu_scalar_multiply__<<<1,1,0,*cuda_stream>>>((talshComplex4*)larg,(talshComplex4*)rarg,(talshComplex4*)darg,
                                                  scale_cmplx4,conj_l,conj_r);
    break;
   case C8:
    scale_cmplx8 = talshComplex8Set(scale_real,scale_imag);
    gpu_scalar_multiply__<<<1,1,0,*cuda_stream>>>((talshComplex8*)larg,(talshComplex8*)rarg,(talshComplex8*)darg,
                                                  scale_cmplx8,conj_l,conj_r);
    break;
   default:
    errc=cuda_task_record(cuda_task,coh_ctrl,64); errc=gpu_activate(cur_gpu); return 64;
  }
// Right tensor rescaled addition:
 }else if(lrank == 0 && rrank > 0){
  bx=1+(vol_d-1)/THRDS_ARRAY_ADD; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
  switch(dtens->data_kind){
   case R4:
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(float*)(darg),(float*)(rarg),(float*)(larg),(float)scale_real);
    break;
   case R8:
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(double*)(darg),(double*)(rarg),(double*)(larg),scale_real);
    break;
   case C4:
    scale_cmplx4 = talshComplex4Set((float)scale_real,(float)scale_imag);
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(talshComplex4*)(darg),(talshComplex4*)(rarg),
                                                           (talshComplex4*)(larg),scale_cmplx4,conj_r);
    break;
   case C8:
    scale_cmplx8 = talshComplex8Set(scale_real,scale_imag);
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(talshComplex8*)(darg),(talshComplex8*)(rarg),
                                                           (talshComplex8*)(larg),scale_cmplx8,conj_r);
    break;
   default:
    errc=cuda_task_record(cuda_task,coh_ctrl,65); errc=gpu_activate(cur_gpu); return 65;
  }
// Left tensor rescaled addition:
 }else if(lrank > 0 && rrank == 0){
  bx=1+(vol_d-1)/THRDS_ARRAY_ADD; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
  switch(dtens->data_kind){
   case R4:
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(float*)(darg),(float*)(larg),(float*)(rarg),(float)scale_real);
    break;
   case R8:
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(double*)(darg),(double*)(larg),(double*)(rarg),scale_real);
    break;
   case C4:
    scale_cmplx4 = talshComplex4Set((float)scale_real,(float)scale_imag);
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(talshComplex4*)(darg),(talshComplex4*)(larg),
                                                           (talshComplex4*)(rarg),scale_cmplx4,conj_l);
    break;
   case C8:
    scale_cmplx8 = talshComplex8Set(scale_real,scale_imag);
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(talshComplex8*)(darg),(talshComplex8*)(larg),
                                                           (talshComplex8*)(rarg),scale_cmplx8,conj_l);
    break;
   default:
    errc=cuda_task_record(cuda_task,coh_ctrl,66); errc=gpu_activate(cur_gpu); return 66;
  }
// Full tensor contraction (via vector dot-product):
 }else if(drank == 0 && lrank > 0 && rrank == lrank){
  bx=1+(vol_l-1)/THRDS_ARRAY_SCALE; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
  switch(ltens->data_kind){
   case R4:
    gpu_array_dot_product__<<<bx,THRDS_ARRAY_SCALE,THRDS_ARRAY_SCALE*sizeof(float),*cuda_stream>>>
                             (vol_l,(float*)larg,(float*)rarg,(float*)darg,(float)scale_real);
    break;
   case R8:
    gpu_array_dot_product__<<<bx,THRDS_ARRAY_SCALE,THRDS_ARRAY_SCALE*sizeof(double),*cuda_stream>>>
                             (vol_l,(double*)larg,(double*)rarg,(double*)darg,scale_real);
    break;
   case C4:
    scale_cmplx4 = talshComplex4Set((float)scale_real,(float)scale_imag);
    gpu_array_dot_product__<<<bx,THRDS_ARRAY_SCALE,THRDS_ARRAY_SCALE*sizeof(talshComplex4),*cuda_stream>>>
                             (vol_l,(talshComplex4*)larg,(talshComplex4*)rarg,(talshComplex4*)darg,scale_cmplx4,conj_l,conj_r);
    break;
   case C8:
    scale_cmplx8 = talshComplex8Set(scale_real,scale_imag);
    gpu_array_dot_product__<<<bx,THRDS_ARRAY_SCALE,THRDS_ARRAY_SCALE*sizeof(talshComplex8),*cuda_stream>>>
                             (vol_l,(talshComplex8*)larg,(talshComplex8*)rarg,(talshComplex8*)darg,scale_cmplx8,conj_l,conj_r);
    break;
   default:
    errc=cuda_task_record(cuda_task,coh_ctrl,67); errc=gpu_activate(cur_gpu); return 67;
  }
// Tensor product (no contracted indices):
 }else if(drank > 0 && drank == lrank + rrank){
  bx=1+(vol_l-1)/THRDS_ARRAY_PRODUCT; by=1+(vol_r-1)/THRDS_ARRAY_PRODUCT;
  limit_cuda_blocks2d(MAX_CUDA_BLOCKS,&bx,&by); dim3 blcks(bx,by);
  switch(dtens->data_kind){
   case R4:
    gpu_array_product__<<<blcks,THRDS_ARRAY_PRODUCT,0,*cuda_stream>>>
                          (vol_l,(float*)larg,vol_r,(float*)rarg,(float*)darg,(float)scale_real);
    break;
   case R8:
    gpu_array_product__<<<blcks,THRDS_ARRAY_PRODUCT,0,*cuda_stream>>>
                          (vol_l,(double*)larg,vol_r,(double*)rarg,(double*)darg,scale_real);
    break;
   case C4:
    scale_cmplx4 = talshComplex4Set((float)scale_real,(float)scale_imag);
    gpu_array_product__<<<blcks,THRDS_ARRAY_PRODUCT,0,*cuda_stream>>>
                          (vol_l,(talshComplex4*)larg,vol_r,(talshComplex4*)rarg,(talshComplex4*)darg,scale_cmplx4,conj_l,conj_r);
    break;
   case C8:
    scale_cmplx8 = talshComplex8Set(scale_real,scale_imag);
    gpu_array_product__<<<blcks,THRDS_ARRAY_PRODUCT,0,*cuda_stream>>>
                          (vol_l,(talshComplex8*)larg,vol_r,(talshComplex8*)rarg,(talshComplex8*)darg,scale_cmplx8,conj_l,conj_r);
    break;
   default:
    errc=cuda_task_record(cuda_task,coh_ctrl,68); errc=gpu_activate(cur_gpu); return 68;
  }
// Partial tensor contraction (via TN matrix multiplication):
 }else{
#ifndef NO_BLAS
  if(DISABLE_BLAS == 0 && gpu_is_mine(gpu_num) >= GPU_MINE_CUBLAS){ //BLAS is enabled
   err_cublas=cublasSetStream(cublas_handle[gpu_num],*cuda_stream);
   if(err_cublas != CUBLAS_STATUS_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,69); errc=gpu_activate(cur_gpu); return 69;}
   switch(dtens->data_kind){
    case R4:
     err_cublas=cublasSgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                (float*)alpha_p,(float*)larg,(int)lc,(float*)rarg,(int)lc,(float*)beta_p,(float*)darg,(int)ll);
     break;
    case R8:
     err_cublas=cublasDgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                (double*)alpha_p,(double*)larg,(int)lc,(double*)rarg,(int)lc,(double*)beta_p,(double*)darg,(int)ll);
     break;
    case C4:
     if(conj_r){
      err_cublas=cublasCgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                 (talshComplex4*)alpha_p,(talshComplex4*)larg,(int)lc,(talshComplex4*)rarg,(int)lr,(talshComplex4*)beta_p,
                 (talshComplex4*)darg,(int)ll);
     }else{
      err_cublas=cublasCgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                 (talshComplex4*)alpha_p,(talshComplex4*)larg,(int)lc,(talshComplex4*)rarg,(int)lc,(talshComplex4*)beta_p,
                 (talshComplex4*)darg,(int)ll);
     }
     break;
    case C8:
     if(conj_r){
      err_cublas=cublasZgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                 (talshComplex8*)alpha_p,(talshComplex8*)larg,(int)lc,(talshComplex8*)rarg,(int)lr,(talshComplex8*)beta_p,
                 (talshComplex8*)darg,(int)ll);
     }else{
      err_cublas=cublasZgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                 (talshComplex8*)alpha_p,(talshComplex8*)larg,(int)lc,(talshComplex8*)rarg,(int)lc,(talshComplex8*)beta_p,
                 (talshComplex8*)darg,(int)ll);
     }
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,70); errc=gpu_activate(cur_gpu); return 70;
   }
   if(err_cublas != CUBLAS_STATUS_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,71); errc=gpu_activate(cur_gpu); return 71;}
  }else{ //BLAS is disabled
#endif /*NO_BLAS*/
   bx=1+(vol_l-1)/MAT_MULT_TILE_DIMX; by=1+(vol_r-1)/MAT_MULT_TILE_DIMY; limit_cuda_blocks2d(MAX_CUDA_BLOCKS,&bx,&by);
   //if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): CUDA exec conf: %d %d %d %d\n",bx,by,MAT_MULT_TILE_DIMX,MAT_MULT_TILE_DIMY); //debug
   dim3 blcks(bx,by); dim3 thrds(MAT_MULT_TILE_DIMX,MAT_MULT_TILE_DIMY);
   switch(dtens->data_kind){
    case R4:
     gpu_matrix_multiply_tn__<<<blcks,thrds,0,*cuda_stream>>>(ll,lr,lc,(float*)larg,(float*)rarg,(float*)darg,(float)scale_real);
     break;
    case R8:
     gpu_matrix_multiply_tn__<<<blcks,thrds,0,*cuda_stream>>>(ll,lr,lc,(double*)larg,(double*)rarg,(double*)darg,scale_real);
     break;
    default: //`Add complex cases with and without conjugation
     errc=cuda_task_record(cuda_task,coh_ctrl,72); errc=gpu_activate(cur_gpu); return 72;
   }
#ifndef NO_BLAS
  }
#endif
 }
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmend,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to record the mmend event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,73); errc=gpu_activate(cur_gpu); return 73;
 }
#endif
 switch(dtens->data_kind){
  case R4:
   gpu_stats[gpu_num].flops+=2.0*((double)(lc))*((double)(ll))*((double)(lr));
   break;
  case R8:
   gpu_stats[gpu_num].flops+=2.0*((double)(lc))*((double)(ll))*((double)(lr));
   break;
  case C4:
   gpu_stats[gpu_num].flops+=8.0*((double)(lc))*((double)(ll))*((double)(lr));
   break;
  case C8:
   gpu_stats[gpu_num].flops+=8.0*((double)(lc))*((double)(ll))*((double)(lr));
   break;
 }
//Schedule the inverse tensor transpose for the destination tensor:
 if(perm_d == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->tmp_rsc->gmem_p),(float*)(dtens->dst_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->tmp_rsc->gmem_p),(double*)(dtens->dst_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->tmp_rsc->gmem_p),(talshComplex4*)(dtens->dst_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->tmp_rsc->gmem_p),(talshComplex8*)(dtens->dst_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,74); errc=gpu_activate(cur_gpu); return 74;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   errc=prmn_convert(drank,cuda_task->tens_args[0].prmn_p,dprm); for(i=0;i<drank;++i) --(dprm[i]);
   for(i=0;i<drank;++i) rprm[i]=(dtens->shape).dims[drank-i-1]; //inversed dimension order
   cutt_err=cuttPlan(&cutt_d,drank,rprm,dprm,((size_t)tds_d),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_d,dtens->tmp_rsc->gmem_p,dtens->dst_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,75); errc=gpu_activate(cur_gpu); return 75;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,76); errc=gpu_activate(cur_gpu); return 76;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,77); errc=gpu_activate(cur_gpu); return 77;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->tmp_rsc->gmem_p),(float*)(dtens->dst_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->tmp_rsc->gmem_p),(double*)(dtens->dst_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->tmp_rsc->gmem_p),(talshComplex4*)(dtens->dst_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->tmp_rsc->gmem_p),(talshComplex8*)(dtens->dst_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,78); errc=gpu_activate(cur_gpu); return 78;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,79); errc=gpu_activate(cur_gpu); return 79;
  }
 }
//Record a CUDA event (output ready on GPU):
 err=cudaEventRecord(*cuda_output,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to record the output event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,80); errc=gpu_activate(cur_gpu); return 80;
 }
//Transfer back the updated destination tensor if needed ("T","K" coherence control):
 coh=(coh_ctrl>>4)&(TWO_BITS_SET); //select bits 4,5 (destination tensor coherence)
 if(gpu_d != gpu_num && coh >= 2){ //data is not on the computing GPU and coherence control = 2("T") or (3)"K":
  err=cudaMemcpyAsync(dtens->src_rsc->gmem_p,dtens->dst_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Destination tensor body back copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,81); errc=gpu_activate(cur_gpu); return 81;
  }
  gpu_stats[gpu_num].traffic_out+=dsize;
 }
//Record a CUDA event (task finished):
 err=cudaEventRecord(*cuda_finish,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to record the finish event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,82); errc=gpu_activate(cur_gpu); return 82;
 }
//Record the successfully scheduled CUDA task and update the Last Task:
 errc=cuda_task_record(cuda_task,coh_ctrl,0);
 LastTask[gpu_num]=cuda_task;
 if(gpu_num != cur_gpu) errc=gpu_activate(cur_gpu);
 return stat; //either 0 (success) or NOT_CLEAN (warning)
}

#endif /*NO_GPU*/
