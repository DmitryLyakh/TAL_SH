/** ExaTensor::TAL-SH: Device-unified user-level C API header.
REVISION: 2019/04/17

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
**/

#ifndef TALSH_H_
#define TALSH_H_

#include <math.h>
#include "timer.h"
#include "tensor_algebra.h"

//TAL-SH PARAMETERS:
#define TALSH_MAX_DEV_PRESENT 16 //max number of on-node devices the tensor block can be simultaneously present on
#define TALSH_MEM_ALLOC_POLICY_HOST MEM_ALLOC_TMP_BUF //default Host memory allocation policy for CP-TAL (see tensor_algebra.h)
#define TALSH_MEM_ALLOC_FALLBACK_HOST 1 //default memory allocation fallback to regular allocate() for CP-TAL: {0|1}
#define TALSH_CPTAL_MIN_BUF_SIZE 1073741824 //minimun Host argument buffer size that can be used effectively by CP-TAL
#define TALSH_NO_HOST_BUFFER 16777216 //nominal Host argument buffer size when it is not needed by the application

//TAL-SH ERROR CODES (keep consistent with "talshf.F90"):
#define TALSH_SUCCESS 0
#define TALSH_FAILURE -666
#define TALSH_NOT_AVAILABLE -888
#define TALSH_NOT_IMPLEMENTED -999
#define TALSH_NOT_INITIALIZED 1000000
#define TALSH_ALREADY_INITIALIZED 1000001
#define TALSH_INVALID_ARGS 1000002
#define TALSH_INTEGER_OVERFLOW 1000003
#define TALSH_OBJECT_NOT_EMPTY 1000004
#define TALSH_OBJECT_IS_EMPTY 1000005
#define TALSH_IN_PROGRESS 1000006
#define TALSH_NOT_ALLOWED 1000007
#define TALSH_LIMIT_EXCEEDED 1000008
#define TALSH_NOT_FOUND 1000009
#define TALSH_OBJECT_BROKEN 1000010
#define TALSH_INVALID_REQUEST 1000011

//TAL-SH TASK STATUS:
#define TALSH_TASK_ERROR 1999999
#define TALSH_TASK_EMPTY 2000000
#define TALSH_TASK_SCHEDULED 2000001
#define TALSH_TASK_STARTED 2000002
#define TALSH_TASK_INPUT_READY 2000003
#define TALSH_TASK_OUTPUT_READY 2000004
#define TALSH_TASK_COMPLETED 2000005

//TAL-SH TENSOR OPERATION KINDS:
#define TALSH_TENSOR_NOOP -1
#define TALSH_TENSOR_INIT 68
#define TALSH_TENSOR_NORM1 69
#define TALSH_TENSOR_NORM2 70
#define TALSH_TENSOR_MIN 71
#define TALSH_TENSOR_MAX 72
#define TALSH_TENSOR_FOLD 73
#define TALSH_TENSOR_UNFOLD 74
#define TALSH_TENSOR_SLICE 75
#define TALSH_TENSOR_INSERT 76
#define TALSH_TENSOR_COPY 77
#define TALSH_TENSOR_PERMUTE 78
#define TALSH_TENSOR_SCALE 79
#define TALSH_TENSOR_ADD 80
#define TALSH_TENSOR_TRACE 81
#define TALSH_TENSOR_CONTRACT 82
#define TALSH_TENSOR_HADAMARD 83
#define TALSH_TENSOR_KHATRIRAO 84

//TAL-SH TENSOR OPERATION STAGES:
#define TALSH_OP_UNDEFINED -1
#define TALSH_OP_EMPTY 0
#define TALSH_OP_PARTIAL 1
#define TALSH_OP_DEFINED 2
#define TALSH_OP_RESOURCED 3
#define TALSH_OP_LOADED 4
#define TALSH_OP_SCHEDULED 5
#define TALSH_OP_COMPLETED 6
#define TALSH_OP_STORED 7
#define TALSH_OP_RETIRED 8


//TAL-SH DATA TYPES:
// Dense tensor with multiple images (interoperable):
typedef struct{
 talsh_tens_shape_t * shape_p; //shape of the tensor block
 talsh_dev_rsc_t * dev_rsc;    //list of device resources occupied by the tensor block body on each device
 int * data_kind;              //list of data kinds for each device location occupied by the tensor body {R4,R8,C4,C8}
 int * avail;                  //list of the data availability flags for each device location occupied by the tensor body
 int dev_rsc_len;              //capacity of .dev_rsc[], .data_kind[], .avail[]
 int ndev;                     //number of devices the tensor block body resides on: ndev <= dev_rsc_len
} talsh_tens_t;

// Dense tensor slice view (view of a dense tensor slice within an actual dense tensor):
typedef struct{
 talsh_tens_t * tensor;        //non-owning pointer to the host-tensor
 talsh_tens_signature_t bases; //tensor slice signature: base offsets of the tensor slice inside the host-tensor
 talsh_tens_shape_t shape;     //tensor slice shape: extents of tensor slice dimensions
} talsh_tens_slice_t;

// Tensor operation argument (auxiliary type):
typedef struct{
 talsh_tens_t * tens_p; //pointer to a tensor block
 int source_image;      //specific body image of that tensor block participating in the operation
} talshTensArg_t;

// TAL-SH task (interoperable):
typedef struct{
 void * task_p;    //pointer to the corresponding device-kind-specific task object
 int task_error;   //-1:undefined(task in progress or empty); 0:successfully completed; >0: error code
 int dev_kind;     //device kind (DEV_NULL: uninitalized)
 int data_kind;    //data kind {R4,R8,C4,C8}, NO_TYPE: uninitialized
 int coherence;    //coherence control (-1:undefined)
 int num_args;     //number of arguments participating in the tensor operation
 talshTensArg_t tens_args[MAX_TENSOR_OPERANDS]; //tensor arguments
 double data_vol;  //total data volume (information)
 double flops;     //number of floating point operations (information)
 double exec_time; //execution time in seconds (information)
} talsh_task_t;

// Basic tensor operation:
typedef struct{
 int opkind;                                         //operation kind
 int data_kind;                                      //operational data kind: {R4,R8,C4,C8}
 unsigned int num_args;                              //number of tensor operands: [0..MAX_TENSOR_OPERANDS]
 talsh_tens_slice_t tens_slice[MAX_TENSOR_OPERANDS]; //formal tensor operands (tensor slice views)
 const char * symb_pattern;                          //symbolic index pattern specification (non-owning pointer to a C-string)
 talshComplex8 alpha;                                //alpha prefactor (scalar factor)
 talsh_tens_t tens_arg[MAX_TENSOR_OPERANDS];         //actual tensor operands (actual TAL-SH tensors)
 talsh_task_t task_handle;                           //task handle
 int exec_dev_id;                                    //execution device id (flat device id)
 int stage;                                          //tensor operation stage
 double time_started;
 double time_scheduled;
 double time_completed;
 double time_finished;
} talsh_tens_op_t;


//EXPORTED FUNCTIONS:
#ifdef __cplusplus
extern "C"{
#endif
// TAL-SH helper functions:
//  Check the validity of a data kind and get its size:
 int talshValidDataKind(int datk, int * datk_size);
// TAL-SH control API:
//  Initialize TAL-SH:
 int talshInit(size_t * host_buf_size,
               int * host_arg_max,
               int ngpus,
               int gpu_list[],
               int nmics,
               int mic_list[],
               int namds,
               int amd_list[]);
//  Shutdown TAL-SH:
 int talshShutdown();
//  Set the memory allocation policy on Host:
 void talshSetMemAllocPolicyHost(int mem_policy,
                                 int fallback,
                                 int * ierr);
//  Get on-node device count:
 int talshDeviceCount(int dev_kind,
                      int * dev_count);
//  Get the flat device Id:
 int talshFlatDevId(int dev_kind,
                    int dev_num);
//  Get the kind-specific device Id:
 int talshKindDevId(int dev_id,
                    int * dev_kind);
//  Query the state of a device:
 int talshDeviceState(int dev_num,
                      int dev_kind = DEV_NULL);
 int talshDeviceState_(int dev_num, int dev_kind);
//  Find the least busy device:
 int talshDeviceBusyLeast(int dev_kind = DEV_NULL);
 int talshDeviceBusyLeast_(int dev_kind);
//  Query device memory size (bytes):
 size_t talshDeviceMemorySize(int dev_num,
                              int dev_kind = DEV_NULL);
 size_t talshDeviceMemorySize_(int dev_num, int dev_kind);
//  Query device argument buffer size (bytes):
 size_t talshDeviceBufferSize(int dev_num,
                              int dev_kind = DEV_NULL);
 size_t talshDeviceBufferSize_(int dev_num, int dev_kind);
//  Query device max tensor size (bytes):
 size_t talshDeviceTensorSize(int dev_num,
                              int dev_kind = DEV_NULL);
 size_t talshDeviceTensorSize_(int dev_num, int dev_kind);
//  Print TAL-SH statistics for specific devices:
 int talshStats(int dev_id = -1,
                int dev_kind = DEV_NULL);
 int talshStats_(int dev_id, int dev_kind);
// TAL-SH tensor block API:
//  Create an empty tensor block:
 int talshTensorCreate(talsh_tens_t ** tens_block);
//  Clean an undefined tensor block (default constructor):
 int talshTensorClean(talsh_tens_t * tens_block);
//  Check whether a tensor block is empty (clean):
 int talshTensorIsEmpty(const talsh_tens_t * tens_block);
//  Construct a tensor block:
 int talshTensorConstruct(talsh_tens_t * tens_block,
                          int data_kind,
                          int tens_rank,
                          const int tens_dims[],
                          int dev_id = 0,
                          void * ext_mem = NULL,
                          int in_hab = -1,
                          talsh_tens_init_i init_method = NULL,
                          double init_val_real = 0.0,
                          double init_val_imag = 0.0);
 int talshTensorConstruct_(talsh_tens_t * tens_block, int data_kind, int tens_rank, const int tens_dims[], int dev_id,
                           void * ext_mem, int in_hab, talsh_tens_init_i init_method, double init_val_real, double init_val_imag);
//  Import external data for the tensor body:
 int talshTensorImportData(talsh_tens_t * tens_block,
                           int data_kind,
                           const void * ext_data);
//  Destruct a tensor block:
 int talshTensorDestruct(talsh_tens_t * tens_block);
//  Destroy a tensor block:
 int talshTensorDestroy(talsh_tens_t * tens_block);
//  Get the tensor block rank (number of dimensions):
 int talshTensorRank(const talsh_tens_t * tens_block);
//  Get the tensor block volume (number of elements per image):
 size_t talshTensorVolume(const talsh_tens_t * tens_block);
//  Get the size of all tensor images in bytes:
 size_t talshTensorSizeAllImages(const talsh_tens_t * tens_block,
                                 int * num_images);
//  Get tensor dimension extents:
 const int * talshTensorDimExtents(const talsh_tens_t * tens_block, int * rank);
//  Get the shape of the tensor block:
 int talshTensorShape(const talsh_tens_t * tens_block,
                      talsh_tens_shape_t * tens_shape);
//  Get the data kind of each tensor image:
 int talshTensorDataKind(const talsh_tens_t * tens_block,
                         int * num_images,
                         int * data_kinds);
//  Reshape a tensor:
 int talshTensorReshape(talsh_tens_t * tens_block,
                        int tens_rank,
                        const int tens_dims[]);
//  Query whether the tensor block is currently in use:
 int talshTensorInUse(const talsh_tens_t * tens_block);
//  Query the presence of the tensor block on device(s):
 int talshTensorPresence(const talsh_tens_t * tens_block,
                         int * ncopies,
                         int copies[],
                         int data_kinds[],
                         int dev_kind = DEV_NULL,
                         int dev_id = -1);
 int talshTensorPresence_(const talsh_tens_t * tens_block, int * ncopies, int copies[], int data_kinds[], int dev_kind, int dev_id);
//  Get access to the tensor body image for a subsequent initialization:
 int talshTensorGetBodyAccess(talsh_tens_t * tens_block,
                              void ** body_p,
                              int data_kind,
                              int dev_id,
                              int dev_kind = DEV_NULL);
 int talshTensorGetBodyAccess_(talsh_tens_t * tens_block, void ** body_p, int data_kind, int dev_id, int dev_kind);
//  Get access to the tensor body image read-only:
 int talshTensorGetBodyAccessConst(const talsh_tens_t * tens_block,
                                   const void ** body_p,
                                   int data_kind,
                                   int dev_id,
                                   int dev_kind = DEV_NULL);
//  Get the scalar value of the rank-0 tensor:
 int talshTensorGetScalar(talsh_tens_t * tens_block,
                          talshComplex8 * scalar_complex);
 int talshTensorGetScalar_(talsh_tens_t * tens_block, double * scalar_real, double * scalar_imag);
//  Print the information on a tensor block:
 void talshTensorPrintInfo(const talsh_tens_t * tens_block);
//  Print tensor elements larger by absolute value than some threshold:
 void talshTensorPrintBody(const talsh_tens_t * tens_block, double thresh);
// TAL-SH tensor slice API:
//  Create an empty TAL-SH tensor slice:
 int talshTensorSliceCreate(talsh_tens_slice_t ** slice);
//  Clean an undefined TAL-SH tensor slice:
 int talshTensorSliceClean(talsh_tens_slice_t * slice);
//  Construct a TAL-SH tensor slice:
 int talshTensorSliceConstruct(talsh_tens_slice_t * slice,
                               const talsh_tens_t * tensor,
                               const size_t * offsets,
                               const int * dims,
                               const int * divs = NULL,
                               const int * grps = NULL);
//  Get the volume of the TAL-SH tensor slice:
 size_t talshTensorSliceVolume(const talsh_tens_slice_t * slice);
//  Destruct a TAL-SH tensor slice:
 int talshTensorSliceDestruct(talsh_tens_slice_t * slice);
//  Destroy a TAL-SH tensor slice:
 int talshTensorSliceDestroy(talsh_tens_slice_t * slice);
// TAL-SH task API:
//  Create a clean (defined-empty) TAL-SH task:
 int talshTaskCreate(talsh_task_t ** talsh_task);
//  Clean an undefined TAL-SH task:
 int talshTaskClean(talsh_task_t * talsh_task);
//  Query whether a TAL-SH task is empty:
 int talshTaskIsEmpty(const talsh_task_t * talsh_task);
//  Destruct a TAL-SH task:
 int talshTaskDestruct(talsh_task_t * talsh_task);
//  Destroy a TAL-SH task:
 int talshTaskDestroy(talsh_task_t * talsh_task);
//  Get the id of the device the TAL-SH task is scheduled on:
 int talshTaskDevId(talsh_task_t * talsh_task,
                    int * dev_kind = NULL);
 int talshTaskDevId_(talsh_task_t * talsh_task, int * dev_kind);
//  Get the TAL-SH task status:
 int talshTaskStatus(talsh_task_t * talsh_task);
//  Check whether a TAL-SH task has completed:
 int talshTaskComplete(talsh_task_t * talsh_task,
                       int * stats,
                       int * ierr);
//  Wait upon a completion of a TAL-SH task:
 int talshTaskWait(talsh_task_t * talsh_task,
                   int * stats);
//  Wait upon a completion of multiple TAL-SH tasks:
 int talshTasksWait(int ntasks,
                    talsh_task_t talsh_tasks[],
                    int stats[]);
//  Get the TAL-SH task timings:
 int talshTaskTime(talsh_task_t * talsh_task,
                   double * total,
                   double * comput = NULL,
                   double * input = NULL,
                   double * output = NULL,
                   double * mmul = NULL);
 int talshTaskTime_(talsh_task_t * talsh_task, double * total, double * comput, double * input, double * output, double * mmul);
//  Print TAL-SH task info:
 void talshTaskPrint(const talsh_task_t * talsh_task);
// TAL-SH tensor operations API:
//  Create an empty tensor operation:
 int talshTensorOpCreate(talsh_tens_op_t ** tens_op);
//  Clean an undefined tensor operation:
 int talshTensorOpClean(talsh_tens_op_t * tens_op);
//  Set a tensor operation argument (tensor slice):
 int talshTensorOpSetArgument(talsh_tens_op_t * tens_op,
                              const talsh_tens_t * tensor,
                              const size_t * offsets,
                              const int * dims);
//  Specify the kind of the tensor operation:
 int talshTensorOpSpecify(talsh_tens_op_t * tens_op,
                          int operation_kind,
                          int data_kind,
                          const char * symbolic_pattern = NULL,
                          double prefactor_real = 1.0,
                          double prefactor_imag = 0.0);
//  Preset execution device:
 int talshTensorOpSetExecDevice(talsh_tens_op_t * tens_op,
                                int dev_id,
                                int dev_kind = DEV_DEFAULT);
//  Activate tensor operation for subsequent processing (resources acquired):
 int talshTensorOpActivate(talsh_tens_op_t * tens_op);
//  Load input (extract input tensor slices):
 int talshTensorOpLoadInput(talsh_tens_op_t * tens_op);
//  Schedule tensor operation for execution of a given device:
 int talshTensorOpExecute(talsh_tens_op_t * tens_op,
                          int dev_id = DEV_DEFAULT,
                          int dev_kind = DEV_DEFAULT);
//  Test for tensor operation completion:
 int talshTensorOpTest(talsh_tens_op_t * tens_op,
                       int * completed,
                       int wait = NOPE);
//  Store output (insert/accumulate output tensor slice):
 int talshTensorOpStoreOutput(talsh_tens_op_t * tens_op);
//  Deactivate tensor operation (resources released):
 int talshTensorOpDeactivate(talsh_tens_op_t * tens_op);
//  Destruct tensor operation (back to an empty state):
 int talshTensorOpDestruct(talsh_tens_op_t * tens_op);
//  Destroy tensor operation:
 int talshTensorOpDestroy(talsh_tens_op_t * tens_op);
//  Progress tensor operation execution:
 int talshTensorOpProgress(talsh_tens_op_t * tens_op, int * done);
//  Get tensor argument volume:
 size_t talshTensorOpGetArgVolume(const talsh_tens_op_t * tens_op,
                                  unsigned int arg_num);
//  Get tensor argument size in bytes:
 size_t talshTensorOpGetArgSize(const talsh_tens_op_t * tens_op,
                                unsigned int arg_num);
//  Tensor operation byte count (memory requirements):
 double talshTensorOpGetByteCount(const talsh_tens_op_t * tens_op,
                                  unsigned int element_size = 1);
//  Tensor operation floating point count (compute requirements):
 double talshTensorOpGetFlopCount(const talsh_tens_op_t * tens_op);
//  Tensor operation arithmetic intensity:
 double talshTensorOpGetIntensity(const talsh_tens_op_t * tens_op);
//  Tensor operation decomposition into two sub-operations:
 int talshTensorOpDecompose2(const talsh_tens_op_t * tens_op, //in: parent tensor operation (defined on entrance)
                             talsh_tens_op_t * child_op1,     //inout: children tensor operation 1 (empty on entrance)
                             talsh_tens_op_t * child_op2);    //inout: children tensor operation 2 (empty on entrance)
//  Print tensor operation:
 void talshTensorOpPrint(const talsh_tens_op_t * tens_op);

//  Place a tensor block on a specific device:
 int talshTensorPlace(talsh_tens_t * tens,               //inout: tensor block
                      int dev_id,                        //in: device id (flat or kind-specific)
                      int dev_kind = DEV_NULL,           //in: device kind (if present, <dev_id> is kind-specific)
                      void * dev_mem = NULL,             //in: externally provided target device memory pointer
                      int copy_ctrl = COPY_M,            //in: copy control (COPY_X), defaults to COPY_M
                      talsh_task_t * talsh_task = NULL); //inout: TAL-SH task handle
 int talshTensorPlace_(talsh_tens_t * tens, int dev_id, int dev_kind, void * dev_mem, int copy_ctrl, talsh_task_t * talsh_task);
//  Discard a tensor block on a specific device:
 int talshTensorDiscard(talsh_tens_t * tens,      //inout: tensor block
                        int dev_id,               //in: device id (flat or kind-specific)
                        int dev_kind = DEV_NULL); //in: device kind (if present, <dev_id> is kind-specific)
 int talshTensorDiscard_(talsh_tens_t * tens, int dev_id, int dev_kind);
//  Discard a tensor block on all devices except a specific device:
 int talshTensorDiscardOther(talsh_tens_t * tens,      //inout: tensor block
                             int dev_id,               //in: device id (flat or kind-specific)
                             int dev_kind = DEV_NULL); //in: device kind (if present, <dev_id> is kind-specific)
 int talshTensorDiscardOther_(talsh_tens_t * tens, int dev_id, int dev_kind);
//  Tensor initialization:
 int talshTensorInit(talsh_tens_t * dtens,              //inout: tensor block
                     double val_real,                   //in: initialization value (real part)
                     double val_imag,                   //in: initialization value (imaginary part)
                     int dev_id = DEV_DEFAULT,          //in: device id (flat or kind-specific)
                     int dev_kind = DEV_DEFAULT,        //in: device kind (if present, <dev_id> is kind-specific)
                     int copy_ctrl = COPY_M,            //in: copy control (COPY_X), defaults to COPY_M
                     talsh_task_t * talsh_task = NULL); //inout: TAL-SH task handle
 int talshTensorInit_(talsh_tens_t * dtens, double val_real, double val_imag, int dev_id, int dev_kind, int copy_ctrl, talsh_task_t * talsh_task);
//  Tensor slicing:
 int talshTensorSlice(talsh_tens_t * dtens,                  //inout: destination tensor block (tensor slice)
                      talsh_tens_t * ltens,                  //inout: source tensor block
                      const int * offsets,                   //in: base offsets of the slice (0-based numeration)
                      int dev_id = DEV_DEFAULT,              //in: device id (flat or kind-specific)
                      int dev_kind = DEV_DEFAULT,            //in: device kind (if present, <dev_id> is kind-specific)
                      int copy_ctrl = COPY_MT,               //in: copy control (COPY_XX), defaults to COPY_MT
                      int accumulative = NOPE,               //in: accumulate in VS overwrite destination tensor: [YEP|NOPE]
                      talsh_task_t * talsh_task = NULL);     //inout: TAL-SH task handle
 int talshTensorSlice_(talsh_tens_t * dtens, talsh_tens_t * ltens, const int * offsets,
                       int dev_id, int dev_kind, int copy_ctrl, int accumulative, talsh_task_t * talsh_task);
//  Tensor insertion:
 int talshTensorInsert(talsh_tens_t * dtens,                  //inout: destination tensor block
                       talsh_tens_t * ltens,                  //inout: source tensor block (tensor slice)
                       const int * offsets,                   //in: base offsets of the slice (0-based numeration)
                       int dev_id = DEV_DEFAULT,              //in: device id (flat or kind-specific)
                       int dev_kind = DEV_DEFAULT,            //in: device kind (if present, <dev_id> is kind-specific)
                       int copy_ctrl = COPY_MT,               //in: copy control (COPY_XX), defaults to COPY_MT
                       int accumulative = NOPE,               //in: accumulate in VS overwrite destination tensor: [YEP|NOPE]
                       talsh_task_t * talsh_task = NULL);     //inout: TAL-SH task handle
 int talshTensorInsert_(talsh_tens_t * dtens, talsh_tens_t * ltens, const int * offsets,
                        int dev_id, int dev_kind, int copy_ctrl, int accumulative, talsh_task_t * talsh_task);
//  Tensor copy:
 int talshTensorCopy(talsh_tens_t * dtens,                  //inout: destination tensor block
                     talsh_tens_t * ltens,                  //inout: source tensor block
                     const int * permutation = NULL,        //in: sign-free O2N tensor dimension permutation (1-based numeration)
                     int dev_id = DEV_DEFAULT,              //in: device id (flat or kind-specific)
                     int dev_kind = DEV_DEFAULT,            //in: device kind (if present, <dev_id> is kind-specific)
                     int copy_ctrl = COPY_MT,               //in: copy control (COPY_XX), defaults to COPY_MT
                     talsh_task_t * talsh_task = NULL);     //inout: TAL-SH task handle
 int talshTensorCopy_(talsh_tens_t * dtens, talsh_tens_t * ltens, const int * permutation,
                      int dev_id, int dev_kind, int copy_ctrl, talsh_task_t * talsh_task);
//  Tensor addition:
 int talshTensorAdd(const char * cptrn,                    //in: C-string: symbolic addition pattern, e.g. "D(a,b,c,d)+=L(c,d,b,a)"
                    talsh_tens_t * dtens,                  //inout: destination tensor block
                    talsh_tens_t * ltens,                  //inout: source tensor block
                    double scale_real = 1.0,               //in: scaling value (real part), defaults to 1
                    double scale_imag = 0.0,               //in: scaling value (imaginary part), defaults to 0
                    int dev_id = DEV_DEFAULT,              //in: device id (flat or kind-specific)
                    int dev_kind = DEV_DEFAULT,            //in: device kind (if present, <dev_id> is kind-specific)
                    int copy_ctrl = COPY_MT,               //in: copy control (COPY_XX), defaults to COPY_MT
                    talsh_task_t * talsh_task = NULL);     //inout: TAL-SH task handle
 int talshTensorAdd_(const char * cptrn, talsh_tens_t * dtens, talsh_tens_t * ltens, double scale_real, double scale_imag,
                     int dev_id, int dev_kind, int copy_ctrl, talsh_task_t * talsh_task);
//  Tensor contraction:
 int talshTensorContract(const char * cptrn,                //in: C-string: symbolic contraction pattern, e.g. "D(a,b,c,d)+=L(c,i,j,a)*R(b,j,d,i)"
                         talsh_tens_t * dtens,              //inout: destination tensor block
                         talsh_tens_t * ltens,              //inout: left source tensor block
                         talsh_tens_t * rtens,              //inout: right source tensor block
                         double scale_real = 1.0,           //in: scaling value (real part), defaults to 1
                         double scale_imag = 0.0,           //in: scaling value (imaginary part), defaults to 0
                         int dev_id = DEV_DEFAULT,          //in: device id (flat or kind-specific)
                         int dev_kind = DEV_DEFAULT,        //in: device kind (if present, <dev_id> is kind-specific)
                         int copy_ctrl = COPY_MTT,          //in: copy control (COPY_XXX), defaults to COPY_MTT
                         int accumulative = YEP,            //in: accumulate in (default) VS overwrite destination tensor: [YEP|NOPE]
                         talsh_task_t * talsh_task = NULL); //inout: TAL-SH task (must be clean)
 int talshTensorContract_(const char * cptrn, talsh_tens_t * dtens, talsh_tens_t * ltens, talsh_tens_t * rtens,
                          double scale_real, double scale_imag, int dev_id, int dev_kind,
                          int copy_ctrl, int accumulative, talsh_task_t * talsh_task);
//  Tensor contraction (extra large):
 int talshTensorContractXL(const char * cptrn,          //in: C-string: symbolic contraction pattern, e.g. "D(a,b,c,d)+=L(c,i,j,a)*R(b,j,d,i)"
                           talsh_tens_t * dtens,        //inout: destination tensor block
                           talsh_tens_t * ltens,        //inout: left source tensor block
                           talsh_tens_t * rtens,        //inout: right source tensor block
                           double scale_real = 1.0,     //in: scaling value (real part), defaults to 1
                           double scale_imag = 0.0,     //in: scaling value (imaginary part), defaults to 0
                           int dev_id = DEV_DEFAULT,    //in: device id (flat or kind-specific)
                           int dev_kind = DEV_DEFAULT,  //in: device kind (if present, <dev_id> is kind-specific)
                           int accumulative = YEP);     //in: accumulate in (default) VS overwrite destination tensor: [YEP|NOPE]
 int talshTensorContractXL_(const char * cptrn, talsh_tens_t * dtens, talsh_tens_t * ltens, talsh_tens_t * rtens,
                            double scale_real, double scale_imag, int dev_id, int dev_kind, int accumulative);
// TAL-SH debugging:
//  1-norm of the tensor body image on Host:
 double talshTensorImageNorm1_cpu(const talsh_tens_t * talsh_tens);
#ifdef __cplusplus
}
#endif

#endif //TALSH_H_
