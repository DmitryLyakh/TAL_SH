/** ExaTensor::TAL-SH: Lower-level header:
    Parameters, derived types, and function prototypes
    used at the lower level of TAL-SH (device agnostic).
REVISION: 2020/02/19

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

#ifndef TENSOR_ALGEBRA_H_
#define TENSOR_ALGEBRA_H_

//GLOBAL PARAMETERS:
#define MAX_TENSOR_RANK 32 //max allowed tensor rank: Must be multiple of 4
#define MAX_TENSOR_OPERANDS 4 //max allowed number of tensor operands in a tensor operation
#define MAX_CONTRACTION_PATTERN_LEN 1024 //max allowed length of a symbolic tensor contraction pattern
#define MAX_MLNDS_PER_TENS 4 //max number of multi-indices per tensor block (dims, divs, grps, prmn)

//DATA KINDS (keep consistent with tensor_algebra.F90):
#define NO_TYPE 0 //null type
//#define R2 2    //half-precision float data kind
#define R4 4      //single-precision float data kind
#define R8 8      //double-precision float data kind
//#define R16 10  //quadruple-precision float data kind
//#define C2 12   //half-precision float complex data kind
#define C4 14     //single-precision float complex data kind
#define C8 18     //double-precision float complex data kind
//#define C16 20  //quadruple-precision float complex data kind

//DEVICE KINDS (keep consistent with tensor_algebra.F90):
#define MAX_GPUS_PER_NODE 8  //max allowed number of NVidia GPUs on a node (peer memory access will not work for more than tha
#define MAX_MICS_PER_NODE 8  //max allowed number of Intel MICs on a node
#define MAX_AMDS_PER_NODE 8  //max allowed number of AMD GPUs on a node
#define DEV_NULL -1          //abstract null device
#define DEV_DEFAULT DEV_NULL //will allow runtime to choose the device
#define DEV_HOST 0           //multicore CPU Host (includes any self-hosted many-core system)
#define DEV_NVIDIA_GPU 1     //NVidia GPU (as an accelerator)
#define DEV_INTEL_MIC 2      //Intel Xeon Phi (as an accelerator)
#define DEV_AMD_GPU 3        //AMD GPU (as an accelerator)
#define DEV_MAX (1+MAX_GPUS_PER_NODE+MAX_MICS_PER_NODE+MAX_AMDS_PER_NODE)

//HOST MEMORY ALLOCATION POLICY (keep consistent with tensor_algebra.F90):
#define MEM_ALLOC_REGULAR 0
#define MEM_ALLOC_TMP_BUF 1
#define MEM_ALLOC_ALL_BUF 2

//ALIASES (keep consistent with tensor_algebra.F90):
#define NOPE 0
#define YEP 1
#define BLAS_ON 0
#define BLAS_OFF 1
#define EFF_TRN_OFF 0
#define EFF_TRN_ON 1
#define EFF_TRN_ON_CUTT 2
#define DEVICE_UNABLE -546372819
#define TRY_LATER -918273645
#define NOT_CLEAN -192837465

#define EVERYTHING 0
#define SOURCE 1
#define DESTINATION 2
#define TEMPORARY 3

#define DEV_OFF 0
#define DEV_ON 1
#define DEV_ON_BLAS 2
#define GPU_OFF 0
#define GPU_MINE 1
#define GPU_MINE_CUBLAS 2
#define NO_COPY_BACK 0
#define COPY_BACK 1

//COHERENCE (COPY) CONTROL PARAMETERS (Senior bits: D -> L -> R: Junior bits):
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
#define ABS(x) ((x)>=0?(x):(-x))
#define ENTERED(x) printf("\n#DEBUG: ENTERED %s\n",#x);
#define EXITED(x) printf("\n#DEBUG: EXITED %s\n",#x);

//DERIVED TYPES (keep consistent with tensor_algebra.F90):
// Tensor signature (interoperable):
typedef struct{
 int num_dim;      //tensor rank (number of dimensions): >=0; -1:empty
 size_t * offsets; //tensor signature: An array of size <num_dim> (long integers)
} talsh_tens_signature_t;

// Tensor shape (interoperable):
typedef struct{
 int num_dim; //tensor rank (number of dimensions): >=0; -1:empty
 int * dims;  //tensor dimension extents (either in regular RAM or pinned)
 int * divs;  //tensor dimension dividers (either in regular RAM or pinned)
 int * grps;  //tensor dimension groups (either in regular RAM or pinned)
} talsh_tens_shape_t;

// Tensor data descriptor (interoperable):
typedef struct{
 void * base;
 size_t volume;
 int data_kind;
} talsh_tens_data_t;

// Dense tensor block (interoperable):
typedef struct{
 int num_dim;                   //tensor rank (number of dimensions)
 int data_kind;                 //data kind
 void * body;                   //pointer to the tensor data storage
 size_t bases[MAX_TENSOR_RANK]; //signature of the tensor block
 size_t dims[MAX_TENSOR_RANK];  //dimension extents
} talsh_tens_dense_t;

// Device resource (occupied by a tensor block):
typedef struct{
 int dev_id;        //flat device id (>=0) the following resources belong to (-1:None)
 void * gmem_p;     //pointer to global memory where the tensor body resides (NULL:None)
 int buf_entry;     //argument buffer entry handle (>=0) corresponding to <gmem_p> (-1:None)
 int mem_attached;  //0:memory was allocated; 1:memory was attached (external memory)
} talsh_dev_rsc_t;

// Interface for a user-defined tensor block initialization function:
typedef int (*talsh_tens_init_i)(const talsh_tens_data_t * tens_data,
                                 const talsh_tens_shape_t * tens_shape,
                                 const talsh_tens_signature_t * tens_signature);
// Dummy function that does no initialization on a tensor:
int talsh_tens_no_init(const talsh_tens_data_t *, const talsh_tens_shape_t *, const talsh_tens_signature_t *);

// Device statistics:
typedef struct{
 unsigned long long int tasks_submitted; //number of TAL-SH tasks submitted to the device
 unsigned long long int tasks_completed; //number of TAL-SH tasks successfully completed on the device
 unsigned long long int tasks_deferred;  //number of TAL-SH tasks deferred (TRY_LATER or DEVICE_UNABLE)
 unsigned long long int tasks_failed;    //number of TAL-SH tasks failed (with an error)
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
//Generic:
 int tens_valid_data_kind(int datk, int * datk_size = NULL);
 int tens_valid_data_kind_(int datk, int * datk_size);
 void get_contr_pattern_sym(const int * rank_left, const int * rank_right, const int * conj_bits, const int * cptrn_dig, char * cptrn_sym, int * cpl, int * ierr);
 int get_contr_pattern_cutensor(const int * dig_ptrn, int drank, int * ptrn_d, int lrank, int * ptrn_l, int rrank, int * ptrn_r);
 size_t tens_elem_offset_f(unsigned int num_dim, const unsigned int * dims, const unsigned int * mlndx);
 void tens_elem_mlndx_f(size_t offset, unsigned int num_dim, const unsigned int * dims, unsigned int * mlndx);
 unsigned int argument_coherence_get_value(unsigned int coh_ctrl, unsigned int tot_args, unsigned int arg_num);
 int argument_coherence_set_value(unsigned int * coh_ctrl, unsigned int tot_args, unsigned int arg_num, unsigned int coh_val);
// Device id conversion:
 int valid_device_kind(int dev_kind);
 int encode_device_id(int dev_kind, int dev_num);
 int decode_device_id(int dev_id, int * dev_kind = NULL);
// Device resource management:
 int tensDevRsc_create(talsh_dev_rsc_t **drsc);
 int tensDevRsc_clean(talsh_dev_rsc_t * drsc);
 int tensDevRsc_is_empty(talsh_dev_rsc_t * drsc);
 int tensDevRsc_same(const talsh_dev_rsc_t * drsc0, const talsh_dev_rsc_t * drsc1);
 int tensDevRsc_clone(const talsh_dev_rsc_t * drsc_in, talsh_dev_rsc_t * drsc_out);
 int tensDevRsc_attach_mem(talsh_dev_rsc_t * drsc, int dev_id, void * mem_p, int buf_entry = -1);
 int tensDevRsc_detach_mem(talsh_dev_rsc_t * drsc);
 int tensDevRsc_allocate_mem(talsh_dev_rsc_t * drsc, int dev_id, size_t mem_size, int in_arg_buf = NOPE);
 int tensDevRsc_free_mem(talsh_dev_rsc_t * drsc);
 int tensDevRsc_get_gmem_ptr(talsh_dev_rsc_t * drsc, void ** gmem_p);
 int tensDevRsc_device_id(talsh_dev_rsc_t * drsc);
 int tensDevRsc_release_all(talsh_dev_rsc_t * drsc);
 int tensDevRsc_destroy(talsh_dev_rsc_t * drsc);
//  C tensor block API:
//   Tensor signature:
 int tensSignature_create(talsh_tens_signature_t ** tsigna);
 int tensSignature_clean(talsh_tens_signature_t * tsigna);
 int tensSignature_construct(talsh_tens_signature_t * tsigna,
                             int rank,
                             const size_t * offsets = NULL);
 int tensSignature_destruct(talsh_tens_signature_t * tsigna);
 int tensSignature_destroy(talsh_tens_signature_t * tsigna);
//   Tensor shape:
 int tensShape_create(talsh_tens_shape_t ** tshape);
 int tensShape_clean(talsh_tens_shape_t * tshape);
 int tensShape_construct(talsh_tens_shape_t * tshape, int pinned,
                         int rank, const int * dims = NULL, const int * divs = NULL, const int * grps = NULL);
 int tensShape_destruct(talsh_tens_shape_t * tshape);
 int tensShape_destroy(talsh_tens_shape_t * tshape);
 size_t tensShape_volume(const talsh_tens_shape_t * tshape);
 int tensShape_rank(const talsh_tens_shape_t * tshape);
 int tensShape_reshape(talsh_tens_shape_t * tshape,
                       int rank, const int * dims = NULL, const int * divs = NULL, const int * grps = NULL);
 void tensShape_print(const talsh_tens_shape_t * tshape);
#ifdef __cplusplus
}
template <typename T> int gpu_matrix_multiply_tn(size_t ll, size_t lr, size_t lc, const T * lmat, const T * rmat, T * dmat);
#endif

#endif /*TENSOR_ALGEBRA_H_*/
