/** ExaTensor::TAL-SH: Device-unified user-level API.
REVISION: 2017/05/18

Copyright (C) 2014-2017 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2017 Oak Ridge National Laboratory (UT-Battelle)

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

FOR DEVELOPER(s):
 # TAL-SH runtime provides a device-kind unified API for performing basic
   tensor algebra operations on multicore CPU Host, Nvidia GPU, Intel MIC, etc.
   Tensor algebra tasks scheduled on Host are blocking (the scheduling call
   returns only after completion of the task). Tensor algebra tasks scheduled
   on an accelerator are non-blocking/asynchronous. Each TAL-SH tensor may
   be present on multiple devices at a time, where the data transfers and
   data consistency are taken care of by the TAL-SH runtime. Underneath,
   the TAL-SH runtime dispatches tasks to device-kind specific (lower-level)
   runtimes called:
   CP-TAL(multicore CPU, synchronous),
   NV-TAL (Nvidia GPU, asynchronous),
   XP-TAL (Intel MIC, asynchronous),
   AM-TAL (AMD GPU, asynchronous),
   etc.
   Contrary to accelerators, the Host does not have a dedicated runtime layer
   for managing resource acquisition, data transfers and consistency, and
   asynchronous execution. Except the latter, these functions are delegated
   to the TAL-SH layer (resource acquisition, data transfers/consistency).
 # Outstanding problems:
   1. Tensor body images participating in tensor operations must be marked
      as "IN_USE" even if they are not to be discarded because other tensor
      operations may mark them "TO_BE_DISCARDED" and then discard them before
      the former tensor operation finishes (inter-task data synchronization).
      So far it is the user responsibility to avoid race conditions.
   2. .data_kind[] array in <talsh_tens_t> is redundant because all images
      have the same data kind (because new tensor body images can only be
      created in tensor operations). Thus, it can be reduced to a scalar.
      Alternatively, by enabling coexistence of images of different data
      kinds, the data kind runtime check needs to be implemented in tensor
      operations and image selecting functions, with a possible data kind
      conversion.
**/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "talsh.h"

//GLOBALS:
// General:
static int talsh_on=0;           //TAL-SH initialization flag (1:initalized; 0:not)
static clock_t talsh_begin_time; //TAL-SH begin time (zero time reference)
// Accelerator configuration (`Needs modification for non-contiguous subranges):
static int talsh_gpu_beg;        //first Nvidia GPU in the range `Obsolete
static int talsh_gpu_end;        //last Nvidia GPU in the range `Obsolete
// Device status:
static int talsh_cpu=DEV_OFF;
static int talsh_gpu[MAX_GPUS_PER_NODE]={DEV_OFF}; //current GPU status: {DEV_OFF,DEV_ON,DEV_ON_BLAS}
static int talsh_mic[MAX_MICS_PER_NODE]={DEV_OFF}; //current MIC status: {DEV_OFF,DEV_ON,DEV_ON_BLAS}
static int talsh_amd[MAX_AMDS_PER_NODE]={DEV_OFF}; //current AMD status: {DEV_OFF,DEV_ON,DEV_ON_BLAS}
// Failure statistics:
static unsigned long long int not_clean_count=0LL; //number of times a NOT_CLEAN status was returned (possible indication of a memory leak)

//INTERNAL TYPES:
// Host task:
typedef struct{
 int task_error; //task error code (-1:empty or in progress; 0:success; >0:error code)
 int host_id;    //-1:uninitialized (empty task); 0:initialized (non-empty)
 unsigned int coherence; //coherence control value
} host_task_t;

//PROTOTYPES OF IMPORTED FUNCTIONS:
#ifdef __cplusplus
extern "C"{
#endif
// CP-TAL tensor operations:
int cpu_tensor_block_init(void * dftr, double val_real, double val_imag, int arg_conj);
int cpu_tensor_block_add(const int * contr_ptrn, void * lftr, void * dftr, double scale_real, double scale_imag, int arg_conj);
int cpu_tensor_block_contract(const int * contr_ptrn, void * lftr, void * rftr, void * dftr, double scale_real, double scale_imag, int arg_conj);
// Contraction pattern conversion:
int talsh_get_contr_ptrn_str2dig(const char * c_str, int * dig_ptrn, int * dig_len, int * conj_bits);
// Fortran tensor block aliasing:
int talsh_tensor_f_assoc(const talsh_tens_t * talsh_tens, int image_id, void ** tensF);
int talsh_tensor_f_dissoc(void * tensF);
int talsh_update_f_scalar(void * tensF, int data_kind, void * gmem_p);
// Host memory allocation policy in CP-TAL:
void talsh_set_mem_alloc_policy_host(int mem_policy, int fallback, int * ierr);
#ifdef __cplusplus
}
#endif

//PROTOTYPES OF INTERNAL FUNCTIONS:
#ifdef __cplusplus
extern "C"{
#endif
// Error counters:
static void talsh_raise_not_clean();
// Tensor body image info (exported to talshf.F90):
int talsh_tensor_image_info(const talsh_tens_t * talsh_tens, //in: TAL-SH tensor block
                            int image_id,                    //in: tensor body image id
                            int * dev_id,                    //out: flat device id where the image resides
                            int * data_kind,                 //out: data kind of the image
                            void ** gmem_p,                  //out: global memory pointer to the image location
                            int * buf_entry);                //out: argument buffer entry holding the image (-1:none)
// Discard tensor body images:
static int talsh_tensor_image_discard(talsh_tens_t * talsh_tens, int image_id);
static int talsh_tensor_image_discard_other(talsh_tens_t * talsh_tens, int image_id);
// Choose an appropriate tensor body image to use in a tensor operation:
static int talsh_choose_image_for_device(talsh_tens_t * tens, unsigned int coh_ctrl, int * copied, int dvk, int dvn = DEV_NULL);
// Host task API:
static int host_task_create(host_task_t ** host_task);
static int host_task_clean(host_task_t * host_task);
static int host_task_is_empty(const host_task_t * host_task);
static int host_task_record(host_task_t * host_task, unsigned int coh_ctrl, unsigned int error_code);
static int host_task_status(host_task_t * host_task);
static int host_task_error_code(const host_task_t * host_task);
static int host_task_destroy(host_task_t * host_task);
static void host_task_print(const host_task_t * host_task);
// C tensor block aliasing:
static int talsh_tensor_c_assoc(const talsh_tens_t * talsh_tens, int image_id, tensBlck_t ** tensC);
static int talsh_tensor_c_dissoc(tensBlck_t * tensC);
// Find an optimal device for a 1-, 2-, or 3-ary tensor operation:
static int talsh_find_optimal_device(const talsh_tens_t * tens0,
                                     const talsh_tens_t * tens1 = NULL,
                                     const talsh_tens_t * tens2 = NULL);
// Additional TAL-SH tensor API:
static int talshTensorIsHealthy(const talsh_tens_t * talsh_tens);
// Additional TAL-SH task API:
static int talshTaskConstruct(talsh_task_t * talsh_task, int dev_kind, int coh_ctrl, int data_kind = NO_TYPE);
static int talshTaskSetArg(talsh_task_t * talsh_task, talsh_tens_t * talsh_tens_p, int image_id);
static int talshTaskFinalize(talsh_task_t * talsh_task, int task_status);
#ifdef __cplusplus
}
#endif

//INTERNAL FUNCTIONS:
// Error counters:
static void talsh_raise_not_clean(){++not_clean_count;}

// Host task API:
static int host_task_create(host_task_t ** host_task)
/** Creates an empty (clean) Host task. **/
{
 *host_task=(host_task_t*)malloc(sizeof(host_task_t));
 if(*host_task == NULL) return TRY_LATER;
 return host_task_clean(*host_task);
}

static int host_task_clean(host_task_t * host_task)
{
 if(host_task == NULL) return TALSH_INVALID_ARGS;
 host_task->task_error=-1;
 host_task->host_id=-1;
 return TALSH_SUCCESS;
}

static int host_task_is_empty(const host_task_t * host_task)
/** Returns YEP if the Host task is empty, NOPE otherwise, unless an error occurs. **/
{
 if(host_task == NULL) return TALSH_INVALID_ARGS;
 if(host_task->host_id < 0){
  if(host_task->task_error >= 0) return TALSH_FAILURE;
  return YEP;
 }
 return NOPE;
}

static int host_task_record(host_task_t * host_task, unsigned int coh_ctrl, unsigned int error_code)
/** Records a Host task. A Host task does not require argument finalization
    because it is done by (blocking) CP-TAL and by higher-level TAL-SH runtimes. **/
{
 if(host_task == NULL) return TALSH_INVALID_ARGS;
 if(host_task_is_empty(host_task) == YEP){
  host_task->task_error=(int)error_code; if(host_task->task_error < 0) return TALSH_INTEGER_OVERFLOW;
  host_task->host_id=0; //Host device kind comprises only one device (multicore CPU Host #0)
  host_task->coherence=coh_ctrl;
 }else{
  return TALSH_OBJECT_NOT_EMPTY;
 }
 return TALSH_SUCCESS;
}

static int host_task_status(host_task_t * host_task)
{
 int errc;

 if(host_task == NULL) return TALSH_INVALID_ARGS;
 errc=host_task_is_empty(host_task);
 if(errc == NOPE){
  if(host_task->task_error == 0){
   return TALSH_TASK_COMPLETED;
  }else if(host_task->task_error > 0){
   return TALSH_TASK_ERROR;
  }
 }else if(errc == YEP){
  return TALSH_TASK_EMPTY;
 }else{
  return TALSH_FAILURE;
 }
 return TALSH_TASK_SCHEDULED;
}

static int host_task_error_code(const host_task_t * host_task)
{return host_task->task_error;}

static int host_task_destroy(host_task_t * host_task)
{
 if(host_task == NULL) return TALSH_INVALID_ARGS;
 free(host_task);
 return TALSH_SUCCESS;
}

static void host_task_print(const host_task_t * host_task)
/** Prints Host task info. **/
{
 if(host_task != NULL){
  printf("\n#MESSAGE: Printing Host task info:\n");
  printf(" Host task status       : %d\n",host_task->task_error);
  printf(" Host task device id    : %d\n",host_task->host_id);
  printf(" Host task coherence_var: %u\n",host_task->coherence);
  printf("#END OF MESSAGE\n");
 }
 return;
}

// Tensor image API:
int talsh_tensor_image_info(const talsh_tens_t * talsh_tens, int image_id,
                            int * dev_id, int * data_kind, void ** gmem_p, int * buf_entry)
/** Returns the information on a specific tensor body image. A return status
    TALSH_NOT_ALLOWED indicates that the image is no longer available (discarded). **/
{
 talsh_dev_rsc_t *drsc;

 if(talsh_tens == NULL) return TALSH_INVALID_ARGS;
 if(talshTensorIsEmpty(talsh_tens) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 if(talshTensorIsHealthy(talsh_tens) != YEP) return TALSH_FAILURE;
 if(image_id < 0 || image_id >= talsh_tens->ndev) return TALSH_INVALID_ARGS;
 drsc=&(talsh_tens->dev_rsc[image_id]);
 if(tensDevRsc_is_empty(drsc) != NOPE) return TALSH_FAILURE;
 if(talsh_tens->avail[image_id] == YEP){
  *data_kind=talsh_tens->data_kind[image_id];
  *dev_id=drsc->dev_id; //flat device id
  *gmem_p=drsc->gmem_p;
  *buf_entry=drsc->buf_entry;
 }else{
  return TALSH_NOT_ALLOWED; //image is no longer available (to be discarded)
 }
 return TALSH_SUCCESS;
}

static int talsh_tensor_image_discard(talsh_tens_t * talsh_tens, int image_id)
/** Discards a specific tensor body image. A return status TALSH_NOT_ALLOWED
    indicates that this is the last available image and it cannot be released. **/
{
 int i,n,errc;

 if(talsh_tens == NULL) return TALSH_INVALID_ARGS;
 if(talshTensorIsEmpty(talsh_tens) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 if(talshTensorIsHealthy(talsh_tens) != YEP) return TALSH_FAILURE;
 if(image_id < 0 || image_id >= talsh_tens->ndev) return TALSH_INVALID_ARGS;
 n=0; for(i=0;i<talsh_tens->ndev;++i){if(i != image_id && talsh_tens->avail[i] == YEP) ++n;}
 if(n == 0) return TALSH_NOT_ALLOWED; //at least one tensor body image must exist, otherwise just destroy the tensor
 errc=tensDevRsc_release_all(&(talsh_tens->dev_rsc[image_id]));
 if(errc != 0 && errc != NOT_CLEAN) errc=TALSH_FAILURE;
 if(image_id < talsh_tens->ndev-1){
  talsh_tens->dev_rsc[image_id]=talsh_tens->dev_rsc[talsh_tens->ndev-1];
  talsh_tens->data_kind[image_id]=talsh_tens->data_kind[talsh_tens->ndev-1];
  talsh_tens->avail[image_id]=talsh_tens->avail[talsh_tens->ndev-1];
 }
 --(talsh_tens->ndev);
 return errc;
}

static int talsh_tensor_image_discard_other(talsh_tens_t * talsh_tens, int image_id)
/** Discards all other tensor body images except the one specified. **/
{
 int i,j,errc;

 if(talsh_tens == NULL) return TALSH_INVALID_ARGS;
 if(talshTensorIsEmpty(talsh_tens) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 if(talshTensorIsHealthy(talsh_tens) != YEP) return TALSH_FAILURE;
 if(image_id < 0 || image_id >= talsh_tens->ndev) return TALSH_INVALID_ARGS;
 if(talsh_tens->avail[image_id] != YEP) return TALSH_NOT_ALLOWED; //at least one tensor body image must exist, otherwise just destroy the tensor
 errc=TALSH_SUCCESS;
 for(i=0;i<talsh_tens->ndev;++i){
  if(i != image_id){
   j=tensDevRsc_release_all(&(talsh_tens->dev_rsc[i]));
   if(j != 0){if(j == NOT_CLEAN){if(errc == TALSH_SUCCESS) errc=j;}else{errc=TALSH_FAILURE;}}
  }else{
   if(image_id > 0){
    talsh_tens->dev_rsc[0]=talsh_tens->dev_rsc[image_id];
    talsh_tens->data_kind[0]=talsh_tens->data_kind[image_id];
    talsh_tens->avail[0]=talsh_tens->avail[image_id];
   }
  }
 }
 talsh_tens->ndev=1;
 return errc;
}

static int talsh_tensor_c_assoc(const talsh_tens_t * talsh_tens, //in: TAL-SH tensor
                                int image_id,                    //in: id of the tensor body image to be used
                                tensBlck_t ** tensC)             //out: newly created <tensBlck_t> object
/** Creates a <tensBlck_t> object for a specific image of <talsh_tens>.
    A return status TRY_LATER indicates temporary shortage in available resources.
    A return status TALSH_NOT_ALLOWED indicates that the requested image is no longer available
    (marked to be discarded). **/
{
 int i,errc;
 tensBlck_t *ctens;
 talsh_dev_rsc_t *src_rsc_p;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(talsh_tens == NULL) return TALSH_INVALID_ARGS;
 if(talshTensorIsEmpty(talsh_tens) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 if(talshTensorIsHealthy(talsh_tens) != YEP) return TALSH_FAILURE;
 if(image_id < 0 || image_id >= talsh_tens->ndev) return TALSH_INVALID_ARGS;
 if(tens_valid_data_kind(talsh_tens->data_kind[image_id]) != YEP) return TALSH_FAILURE;
 if(talsh_tens->avail[image_id] == YEP){
  src_rsc_p=&(talsh_tens->dev_rsc[image_id]);
  errc=tensBlck_create(&ctens); if(errc){if(errc != TRY_LATER) errc=TALSH_FAILURE; return errc;}
  errc=tensBlck_construct(ctens,YEP,talsh_tens->shape_p->num_dim,talsh_tens->shape_p->dims, //YEP: shape in pinned memory
                                    talsh_tens->shape_p->divs,talsh_tens->shape_p->grps);
  if(errc){if(errc != TRY_LATER) errc=TALSH_FAILURE; i=tensBlck_destroy(ctens); return errc;}
  errc=tensBlck_attach_body(ctens,talsh_tens->data_kind[image_id],src_rsc_p->dev_id,src_rsc_p->gmem_p,src_rsc_p->buf_entry);
  if(errc){if(errc != TRY_LATER) errc=TALSH_FAILURE; i=tensBlck_destroy(ctens); return errc;}
  *tensC=ctens; //tensC has the right shape, data_kind, and source data
 }else{
  return TALSH_NOT_ALLOWED; //image is no longer available (to be discarded)
 }
 return TALSH_SUCCESS;
}

static int talsh_tensor_c_dissoc(tensBlck_t * tensC) //inout: <tensBlck_t> created by <talsh_tensor_c_assoc()>
/** Destroys the <tensBlck_t> object created by <talsh_tensor_c_assoc()>. **/
{
 int errc;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(tensC == NULL) return TALSH_INVALID_ARGS;
 errc=TALSH_SUCCESS;
 if(tensBlck_volume(tensC) > 0){
  errc=tensBlck_destroy(tensC); if(errc){if(errc != NOT_CLEAN) errc=TALSH_FAILURE;}
 }
 return errc;
}

static int talsh_find_optimal_device(const talsh_tens_t * tens0, const talsh_tens_t * tens1, const talsh_tens_t * tens2)
/** Given tensor arguments, returns a flat id of the most appropriate device
    based on the data residence, tensor sizes, and current device occupation.
    A negative return status indicates an error. **/
{
 int i,j,devid,al,am,as,ov[3][TALSH_MAX_DEV_PRESENT],ovl[3];
 const int idx[3][3]={-1,0,1, 0,-1,2, 1,2,-1};
 size_t s[3];

 s[0]=0; if(tens0 != NULL){s[0]=talshTensorVolume(tens0);}else{return DEV_NULL;}
 s[1]=0; if(tens1 != NULL) s[1]=talshTensorVolume(tens1);
 s[2]=0; if(tens2 != NULL) s[2]=talshTensorVolume(tens2);
 devid=DEV_NULL; for(i=0;i<3;++i) ovl[i]=0;
 //Overlap tens0 and tens1:
 if(s[0] > 0 && s[1] > 0){
  for(i=0;i<tens0->ndev;++i){
   if(tens0->avail[i] == YEP){
    for(j=0;j<tens1->ndev;++j){
     if(tens1->avail[j] == YEP){
      if(tens1->dev_rsc[j].dev_id == tens0->dev_rsc[i].dev_id){
       ov[0][(ovl[0])++]=tens1->dev_rsc[j].dev_id;
      }
     }
    }
   }
  }
  if(s[2] > 0){
   for(j=0;j<tens2->ndev;++j){
    if(tens2->avail[j] == YEP){
     for(i=0;i<ovl[0];++i){
      if(tens2->dev_rsc[j].dev_id == ov[0][i]) return ov[0][i]; //triple match
     }
    }
   }
  }
 }
 //Overlap tens0 and tens2:
 if(s[0] > 0 && s[2] > 0){
  for(i=0;i<tens0->ndev;++i){
   if(tens0->avail[i] == YEP){
    for(j=0;j<tens2->ndev;++j){
     if(tens2->avail[j] == YEP){
      if(tens2->dev_rsc[j].dev_id == tens0->dev_rsc[i].dev_id){
       ov[1][(ovl[1])++]=tens2->dev_rsc[j].dev_id;
      }
     }
    }
   }
  }
  if(s[1] > 0){
   for(j=0;j<tens1->ndev;++j){
    if(tens1->avail[j] == YEP){
     for(i=0;i<ovl[1];++i){
      if(tens1->dev_rsc[j].dev_id == ov[1][i]) return ov[1][i]; //triple match
     }
    }
   }
  }
 }
 //Overlap tens1 and tens2:
 if(s[1] > 0 && s[2] > 0){
  for(i=0;i<tens1->ndev;++i){
   if(tens1->avail[i] == YEP){
    for(j=0;j<tens2->ndev;++j){
     if(tens2->avail[j] == YEP){
      if(tens2->dev_rsc[j].dev_id == tens1->dev_rsc[i].dev_id){
       ov[2][(ovl[2])++]=tens2->dev_rsc[j].dev_id;
      }
     }
    }
   }
  }
  if(s[0] > 0){
   for(j=0;j<tens0->ndev;++j){
    if(tens0->avail[j] == YEP){
     for(i=0;i<ovl[2];++i){
      if(tens0->dev_rsc[j].dev_id == ov[2][i]) return ov[2][i]; //triple match
     }
    }
   }
  }
 }
 //No triple match happened => communication is necessary.
 //Order the arguments by size (#al >= #am >= #as):
 if(s[1] >= s[2]){al=1; am=2;}else{al=2; am=1;}
 if(s[0] >= al){
  as=am; am=al; al=0;
 }else{
  if(s[0] >= am){as=am; am=0;}else{as=0;}
 }
 //Find the optimal device to minimize the communication:
 if(s[al] > s[am] + s[as]){
  i=idx[al][am]; if(ovl[i] > 0) return ov[i][0]; //Large/medium overlap
  i=idx[al][as]; if(ovl[i] > 0) return ov[i][0]; //Large/small overlap
 }else{
  i=idx[al][am]; if(ovl[i] > 0) return ov[i][0]; //Large/medium overlap
  i=idx[al][as]; if(ovl[i] > 0) return ov[i][0]; //Large/small overlap
  i=idx[am][as]; if(ovl[i] > 0) return ov[i][0]; //Medium/small overlap
 }
 switch(al){ //Large does not overlap with other arguments
  case 0: devid=tens0->dev_rsc[0].dev_id; break;
  case 1: devid=tens1->dev_rsc[0].dev_id; break;
  case 2: devid=tens2->dev_rsc[0].dev_id; break;
 }
 return devid;
}

static int talsh_choose_image_for_device(talsh_tens_t * tens, unsigned int coh_ctrl, int * copied, int dvk, int dvn)
/** For a given execution device <[dvk,dvn]>, chooses the most appropriate
    tensor body image to be used on that device. Priority is given to the
    same device, then to the same device kind, then to the Host. If no image
    is found in that sequence, a blocking copy will be posted to the Host
    with a copy control value equal to COPY_K, thus creating an additional
    image of the tensor body (on Host)! If <dvn> = DEV_NULL, only the search
    within the given device kind will be performed. A negative return code
    indicates an error. **/
{
 int i,image_id,host_image,dn,dk,coh;

 *copied=0; image_id=-1; host_image=-1;
 if(tens == NULL) return -1;
 if(talshTensorIsEmpty(tens) != NOPE) return -1;
 if(talshTensorIsHealthy(tens) != YEP) return -1;
 for(i=0;i<tens->ndev;++i){
  if(tens->avail[i] == YEP){
   dn=talshKindDevId(tens->dev_rsc[i].dev_id,&dk); if(dn < 0) return -1;
   if(dk == dvk){image_id=i; if(dn == dvn) return image_id;}
   if(dk == DEV_HOST) host_image=i;
  }
 }
 if(image_id < 0){
  if(host_image < 0){ //create an image on Host
   switch(coh_ctrl){
    case COPY_D: coh=COPY_M; break;
    case COPY_M: coh=COPY_M; break;
    case COPY_T: coh=COPY_K; break;
    case COPY_K: coh=COPY_K; break;
    default: return -1;
   }
   i=talshTensorPlace(tens,0,DEV_HOST,NULL,coh); if(i) return -1;
   *copied=1; image_id=tens->ndev-1;
   if(tens->dev_rsc[image_id].dev_id != talshFlatDevId(DEV_HOST,0)) return -1;
  }else{
   image_id=host_image;
  }
 }
 return image_id;
}

//EXPORTED FUNCTIONS:
// TAL-SH helper functions:
int talshValidDataKind(int datk, int * datk_size)
/** Returns YEP if <datk> is a valid data kind (also returns its size in bytes in <datk_size>). **/
{
 return tens_valid_data_kind(datk,datk_size);
}
// TAL-SH control API:
int talshInit(size_t * host_buf_size,    //inout: Host Argument Buffer size in bytes (in: suggested; out: actual)
              int * host_arg_max,        //out: Max number of arguments that can fit into the Host Argument Buffer
              int ngpus, int gpu_list[], //in: number of Nvidia GPU(s) to use and the list of Nvidia GPU(s) to use
              int nmics, int mic_list[], //in: number of Intel Xeon Phi(s) to use and the list of Intel Xeon Phi(s) to use
              int namds, int amd_list[]) //in: number of AMD GPU(s) to use and the list of AMD GPU(s) to use
/** Initializes the TAL-SH runtime. **/
{
 int i,j,gpu_beg,gpu_end,errc;

 if(talsh_on) return TALSH_ALREADY_INITIALIZED;
//CPU Host:
#ifndef NO_BLAS
 talsh_cpu=DEV_ON_BLAS;
#else
 talsh_cpu=DEV_ON;
#endif
//NVidia GPU accelerators:
#ifndef NO_GPU
 if(ngpus > 0){
  if(ngpus > MAX_GPUS_PER_NODE) return TALSH_INVALID_ARGS;
  gpu_beg=gpu_list[0]; gpu_end=gpu_list[ngpus-1]; //`Allow for non-consecutive GPU ranges in arg_buf_allocate()
  if(gpu_beg < 0 || gpu_beg >= MAX_GPUS_PER_NODE) return TALSH_INVALID_ARGS;
  if(gpu_end < 0 || gpu_end >= MAX_GPUS_PER_NODE) return TALSH_INVALID_ARGS;
  for(i=1;i<ngpus;i++){
   if(gpu_list[i] != gpu_list[i-1]+1){
    printf("#FATAL(TALSH::talshInit): The current version only supports consecutive GPU ranges!");
    return TALSH_FAILURE;
   }
  }
 }else{
#endif
  gpu_beg=0; gpu_end=-1;
#ifndef NO_GPU
 }
#endif
//Intel Xeon Phi accelerators:
#ifndef NO_PHI
 if(nmics > 0){
  printf("#FATAL(TALSH::talshInit): Intel Xeon Phi is not fully supported yet!");
  return TALSH_NOT_IMPLEMENTED; //`Future
 }
#endif
//AMD GPU accelerators:
#ifndef NO_AMD
 if(namds > 0){
  printf("#FATAL(TALSH::talshInit): AMD GPU is not supported yet!");
  return TALSH_NOT_IMPLEMENTED; //`Future
 }
#endif
 errc=arg_buf_allocate(host_buf_size,host_arg_max,gpu_beg,gpu_end); if(errc) return TALSH_FAILURE;
 if(*host_buf_size >= TALSH_CPTAL_MIN_BUF_SIZE){ //Host argument buffer is big enough to be used in CP-TAL
  talsh_set_mem_alloc_policy_host(TALSH_MEM_ALLOC_POLICY_HOST,TALSH_MEM_ALLOC_FALLBACK_HOST,&errc);
  if(errc != 0){
   printf("#FATAL(TALSH::talshInit): Host memory allocation policy setting failed: Error %d",errc);
   return TALSH_FAILURE;
  }
 }
#ifndef NO_GPU
 for(i=0;i<ngpus;i++){
  j=gpu_list[i]; if(j < 0 || j >= MAX_GPUS_PER_NODE) return TALSH_INVALID_ARGS;
  talsh_gpu[j]=gpu_is_mine(j);
 }
#endif
 talsh_gpu_beg=gpu_beg; talsh_gpu_end=gpu_end;
 talsh_on=1; talsh_begin_time=clock();
 return TALSH_SUCCESS;
}

int talshShutdown()
/** Shuts down the TAL-SH runtime. **/
{
 int i,errc;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 errc=arg_buf_deallocate(talsh_gpu_beg,talsh_gpu_end);
 talsh_set_mem_alloc_policy_host(TALSH_MEM_ALLOC_POLICY_HOST,TALSH_MEM_ALLOC_FALLBACK_HOST,&i);
 talsh_gpu_beg=0; talsh_gpu_end=-1; talsh_on=0;
 talsh_cpu=DEV_OFF;
 for(i=0;i<MAX_GPUS_PER_NODE;i++) talsh_gpu[i]=DEV_OFF;
 for(i=0;i<MAX_MICS_PER_NODE;i++) talsh_mic[i]=DEV_OFF;
 for(i=0;i<MAX_AMDS_PER_NODE;i++) talsh_amd[i]=DEV_OFF;
 if(errc) return TALSH_FAILURE;
 return TALSH_SUCCESS;
}

int talshGetDeviceCount(int dev_kind, int * dev_count)
/** Returns the total number of devices of specific kind found on node. **/
{
 int errc;

 errc=TALSH_SUCCESS;
 switch(dev_kind){
  case DEV_HOST:
   *dev_count=1; //CPU Host is always assumed a single device (multicore)
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   errc=cuda_get_device_count(dev_count);
   if(errc != 0) errc=TALSH_FAILURE;
#else
   *dev_count=0;
#endif
   break;
  case DEV_INTEL_MIC:
   *dev_count=0;
   errc=TALSH_NOT_IMPLEMENTED;
   break;
  case DEV_AMD_GPU:
   *dev_count=0;
   errc=TALSH_NOT_IMPLEMENTED;
   break;
 }
 return errc;
}

int talshFlatDevId(int dev_kind, //in: device kind
                   int dev_num)  //in: device Id within its kind (0..MAX)
/** Converts a kind-specific device Id into the flat device Id.
    DEV_MAX return status indicates invalidity of the arguments. **/
{
 return encode_device_id(dev_kind,dev_num); //Success:[0..DEV_MAX-1]; Failure: DEV_MAX
}

int talshKindDevId(int dev_id,     //in: flat device Id: [0:DEV_MAX-1]
                   int * dev_kind) //out: device kind
/** Converts a flat device Id into the kind specific device Id.
    A negative return value indicates an invalid flat device Id. **/
{
 return decode_device_id(dev_id,dev_kind);
}

int talshDeviceState(int dev_num,  //in: either a flat or kind specific (when <dev_kind> is present) device id
                     int dev_kind) //in: device kind (note that it changes the meaning of the <dev_num> argument)
/** Returns device state (Success:[DEV_OFF,DEV_ON,DEV_ON_BLAS]) **/
{
 int devk,i,sts;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(dev_kind == DEV_NULL){
  i=talshKindDevId(dev_num,&devk);
  if(i < 0) return TALSH_INVALID_ARGS;
 }else{
  devk=dev_kind;
 }
 switch(devk){
  case DEV_HOST:
   sts=talsh_cpu;
   break;
  case DEV_NVIDIA_GPU:
   sts=talsh_gpu[i];
   break;
  case DEV_INTEL_MIC:
   sts=talsh_mic[i];
   break;
  case DEV_AMD_GPU:
   sts=talsh_amd[i];
   break;
  default:
   return TALSH_INVALID_ARGS;
 }
 return sts;
}

int talshDeviceState_(int dev_num, int dev_kind) //Fortran wrapper
{
 return talshDeviceState(dev_num,dev_kind);
}

int talshDeviceBusyLeast(int dev_kind) //in: device kind (defaults to any kind)
/** Returns the least busy device id. **/
{
 int i;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 switch(dev_kind){
  case DEV_NULL:
   return talshFlatDevId(DEV_HOST,0); //`if device kind not specified, return CPU Host for simplicity
  case DEV_HOST:
   return talshFlatDevId(DEV_HOST,0);
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   i=gpu_busy_least();
   if(i < 0 || i >= MAX_GPUS_PER_NODE) return TALSH_FAILURE;
   return i;
#else
   return TALSH_NOT_AVAILABLE;
#endif
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   return TALSH_NOT_IMPLEMENTED; //`Implement in future
#else
   return TALSH_NOT_AVAILABLE;
#endif
  case DEV_AMD_GPU:
#ifndef NO_AMD
   return TALSH_NOT_IMPLEMENTED; //`Implement in future
#else
   return TALSH_NOT_AVAILABLE;
#endif
 }
 return TALSH_INVALID_ARGS;
}

int talshDeviceBusyLeast_(int dev_kind) //Fortran wrapper
{
 return talshDeviceBusyLeast(dev_kind);
}

int talshStats(int dev_id,   //in: device id (either flat or kind specific device id, see below)
               int dev_kind) //in: device kind (if present, <dev_id> will be interpreted as kind specific)
/** Prints the run-time statistics for devices of interest. **/
{
 int rc=TALSH_SUCCESS,devk,devn;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 switch(dev_kind){
  case DEV_NULL:
   if(dev_id < 0){ //print stats for all active devices
    rc=talshStats(-1,DEV_HOST);
    rc=talshStats(-1,DEV_NVIDIA_GPU);
    rc=talshStats(-1,DEV_INTEL_MIC);
    rc=talshStats(-1,DEV_AMD_GPU);
    rc=TALSH_SUCCESS;
   }else{
    devn=talshKindDevId(dev_id,&devk);
    rc=talshStats(devn,devk);
   }
   break;
  case DEV_HOST:
   rc=TALSH_NOT_IMPLEMENTED; //`Implement
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   rc=gpu_print_stats(dev_id);
#else
   rc=TALSH_NOT_AVAILABLE;
#endif
   break;
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   rc=TALSH_NOT_IMPLEMENTED; //`Implement in future
#else
   rc=TALSH_NOT_AVAILABLE;
#endif
   break;
  case DEV_AMD_GPU:
#ifndef NO_AMD
   rc=TALSH_NOT_IMPLEMENTED; //`Implement in future
#else
   rc=TALSH_NOT_AVAILABLE;
#endif
   break;
  default:
   rc=TALSH_INVALID_ARGS;
 }
 return rc;
}

int talshStats_(int dev_id, int dev_kind) //Fortran wrapper
{
 return talshStats(dev_id,dev_kind);
}

// TAL-SH tensor block API:
int talshTensorCreate(talsh_tens_t ** tens_block) //out: pointer to a newly created empty tensor block
/** Returns a pointer to a newly created empty tensor block (0:Success; TRY_LATER:Short on memory). **/
{
 *tens_block=(talsh_tens_t*)malloc(sizeof(talsh_tens_t));
 if(*tens_block == NULL) return TRY_LATER;
 return talshTensorClean(*tens_block);
}

int talshTensorClean(talsh_tens_t * tens_block)
/** Cleans an undefined tensor block (default ctor) making it defined-empty. **/
{
 if(tens_block == NULL) return TALSH_INVALID_ARGS;
 tens_block->shape_p=NULL;
 tens_block->dev_rsc=NULL;   //`.tens_image.dev_rsc
 tens_block->data_kind=NULL; //`.tens_image.data_kind
 tens_block->avail=NULL;     //`.tens_image.avail
 tens_block->dev_rsc_len=0;  //`.tens_image.capacity
 tens_block->ndev=0;         //`.tens_image.ndev
 return TALSH_SUCCESS;
}

int talshTensorIsEmpty(const talsh_tens_t * tens_block)
/** Returns YEP if the tensor block is empty, NOPE otherwise, unless an error occurs. **/
{
 if(tens_block == NULL) return TALSH_INVALID_ARGS;
 if(tens_block->shape_p == NULL) return YEP;
 return NOPE;
}

int talshTensorConstruct(talsh_tens_t * tens_block,     //inout: empty tensor block on entrance, constructed tensor block on exit
                         int data_kind,                 //in: data kind: {R4,R8,C4,C8,NO_TYPE}
                         int tens_rank,                 //in: tensor block rank (number of dimensions)
                         const int tens_dims[],         //in: tensor block dimension extents
                         int dev_id,                    //in: flat device ID on which the tensor block will reside
                         void * ext_mem,                //in: pointer to externally provided memory for tensor elements
                         int in_hab,                    //in: if >=0, a non-NULL <ext_mem> points to the HAB entry #<in_hab>
                         talsh_tens_init_i init_method, //in: user-defined initialization method (function pointer)
                         double init_val_real,          //in: initialization value (real part), defaults to 0.0
                         double init_val_imag)          //in: initialization value (imaginary part), defaults to 0.0
/** Constructs a tensor block: {0: success; TRY_LATER: no enough free memory available; DEVICE_UNABLE: device unable}.
    If <data_kind> == NO_TYPE, the tensor body will not be allocated (only the tensor shape),
    unless an external storage is provided (<ext_mem>). In case the tensor body storage space
    is provided externally (<ext_mem> != NULL), the initialization step is skipped. In other cases,
    unless <data_kind>=NO_TYPE, the newly allocated tensor body will be initialized by a user-defined
    method, or, if no method is provided (NULL), by a user-defined value, which defaults to zero.
    If the tensor body initialization failed, a status NOT_CLEAN is returned but
    the tensor block is ready for use (except its body value is still undefined). **/
{
 int i,j,dev_num,dev_kind,dksize,errc,already_allocated,use_hab;
 size_t tvol,tsize;
 float fval;
 float *fp;
 double *dp;
 talshComplex4 *cfp,cfv;
 talshComplex8 *cdp,cdv;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 errc=TALSH_SUCCESS;
 //Check arguments:
 if(tens_block == NULL) return TALSH_INVALID_ARGS; //tensor block must have been preallocated
 if(talshTensorIsEmpty(tens_block) != YEP) return TALSH_OBJECT_NOT_EMPTY; //tensor block is not empty (destruct it first)
 if(tens_valid_data_kind(data_kind,&dksize) != YEP) return TALSH_INVALID_ARGS; //unknown data kind (NO_TYPE is a valid type here)
 dev_num=talshKindDevId(dev_id,&dev_kind); if(dev_num < 0) return TALSH_INVALID_ARGS; //invalid device id
 already_allocated=0; if(ext_mem != NULL) already_allocated=1; //check whether an external memory space is provided for the tensor body
 if(in_hab >= 0){use_hab=YEP;}else{in_hab=-1; use_hab=NOPE;}
 //Tensor shape:
 errc=tensShape_create(&(tens_block->shape_p)); if(errc == TRY_LATER || errc == DEVICE_UNABLE) return errc;
 if(errc != 0 || tens_block->shape_p == NULL) return TALSH_FAILURE;
 errc=tensShape_construct(tens_block->shape_p,NOPE,tens_rank,tens_dims); //NOPE = not pinned
 if(errc != 0 && errc != TRY_LATER && errc != DEVICE_UNABLE) errc=TALSH_FAILURE;
 if(errc != 0){i=talshTensorDestruct(tens_block); return errc;}
 //Device resource storage:
 if(tens_block->dev_rsc_len == 0 && tens_block->dev_rsc == NULL &&
    tens_block->data_kind == NULL && tens_block->avail == NULL){ //tensor block must be defined-empty
  tens_block->dev_rsc=(talsh_dev_rsc_t*)malloc(TALSH_MAX_DEV_PRESENT*sizeof(talsh_dev_rsc_t));
  if(tens_block->dev_rsc != NULL){
   tens_block->dev_rsc_len=TALSH_MAX_DEV_PRESENT; tens_block->ndev=0;
   for(j=0;j<TALSH_MAX_DEV_PRESENT;++j){i=tensDevRsc_clean(&(tens_block->dev_rsc[j]));}
   tens_block->data_kind=(int*)malloc(TALSH_MAX_DEV_PRESENT*sizeof(int));
   if(tens_block->data_kind != NULL){
    for(j=0;j<TALSH_MAX_DEV_PRESENT;++j){tens_block->data_kind[j]=NO_TYPE;}
    tens_block->avail=(int*)malloc(TALSH_MAX_DEV_PRESENT*sizeof(int));
    if(tens_block->avail != NULL){
     for(j=0;j<TALSH_MAX_DEV_PRESENT;++j){tens_block->avail[j]=NOPE;}
    }else{
     i=talshTensorDestruct(tens_block); return TRY_LATER;
    }
   }else{
    i=talshTensorDestruct(tens_block); return TRY_LATER;
   }
  }else{
   i=talshTensorDestruct(tens_block); return TRY_LATER;
  }
 }else{
  i=talshTensorDestruct(tens_block);
  return TALSH_INVALID_ARGS;
 }
 //Tensor body:
 if(already_allocated){ //tensor body storage has been allocated outside (no initialization will be performed)
  errc=tensDevRsc_attach_mem(&(tens_block->dev_rsc[0]),dev_id,ext_mem,in_hab);
  if(errc){i=talshTensorDestruct(tens_block); return TALSH_FAILURE;}
  tens_block->data_kind[0]=data_kind; tens_block->avail[0]=YEP; tens_block->ndev=1;
 }else{ //tensor body storage needs to be allocated here (will also be initialized), unless NO_TYPE
  if(data_kind != NO_TYPE){
   tvol=talshTensorVolume(tens_block);
   if(tvol > 0){
    tsize=tvol*dksize;
    if(tsize <= 0){i=talshTensorDestruct(tens_block); return TALSH_INTEGER_OVERFLOW;}
    errc=tensDevRsc_allocate_mem(&(tens_block->dev_rsc[0]),dev_id,tsize,use_hab);
    if(errc != 0 && errc != TRY_LATER && errc != DEVICE_UNABLE) errc=TALSH_FAILURE;
    if(errc != 0){i=talshTensorDestruct(tens_block); return errc;}
    tens_block->data_kind[0]=data_kind; tens_block->avail[0]=YEP; tens_block->ndev=1;
   }else{
    i=talshTensorDestruct(tens_block);
    return TALSH_FAILURE;
   }
   //Initialization:
   if(tens_block->ndev > 0){
    if(dev_kind == DEV_HOST){ //`Currently supported only on Host
     if(init_method != NULL){
      init_method(tens_block->dev_rsc[0].gmem_p,data_kind,tens_rank,tens_dims,&errc);
      if(errc) errc=NOT_CLEAN; //initialization failed, tensor block value is undefined, but one may continue
     }else{
      switch(data_kind){
       case R4:
        fval = (float)init_val_real;
        fp = (float*)(tens_block->dev_rsc[0].gmem_p);
#pragma omp parallel for schedule(guided)
        for(size_t l=0; l < tvol; l++) fp[l]=fval;
        break;
       case R8:
        dp = (double*)(tens_block->dev_rsc[0].gmem_p);
#pragma omp parallel for schedule(guided)
        for(size_t l=0; l < tvol; l++) dp[l]=init_val_real;
        break;
       case C4:
        cfv = talshComplex4Set(((float)init_val_real),((float)init_val_imag));
        cfp = (talshComplex4*)(tens_block->dev_rsc[0].gmem_p);
#pragma omp parallel for schedule(guided)
        for(size_t l=0; l < tvol; l++) cfp[l]=cfv;
        break;
       case C8:
        //printf("\n#DBG\n %llu",tvol); //debug
        cdv = talshComplex8Set(init_val_real,init_val_imag);
        cdp = (talshComplex8*)(tens_block->dev_rsc[0].gmem_p);
#pragma omp parallel for schedule(guided)
        for(size_t l=0; l < tvol; l++) cdp[l]=cdv;
        break;
       default:
        return TALSH_FAILURE;
      }
     }
    }else{
     errc=TALSH_NOT_IMPLEMENTED; //`Initialization on other device kinds should be enabled
    }
   }
  }
 }
 return errc;
}

int talshTensorConstruct_(talsh_tens_t * tens_block, int data_kind, int tens_rank, const int tens_dims[], int dev_id, //Fortran wrapper
                          void * ext_mem, int in_hab, talsh_tens_init_i init_method, double init_val_real, double init_val_imag)
{
 return talshTensorConstruct(tens_block, data_kind, tens_rank, tens_dims, dev_id,
                             ext_mem, in_hab, init_method, init_val_real, init_val_imag);
}

int talshTensorDestruct(talsh_tens_t * tens_block) //in: non-NULL pointer to a tensor block (empty tensor block on exit)
/** Destructs a tensor block and sets its status to empty. **/
{
 int i,j,errc;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 errc=TALSH_SUCCESS;
 if(tens_block == NULL) return TALSH_INVALID_ARGS;
 if(tens_block->shape_p != NULL){
  i=tensShape_destroy(tens_block->shape_p); tens_block->shape_p=NULL;
  if(i == 0 || i == NOT_CLEAN){if(errc == 0) errc=i;}else{errc=TALSH_FAILURE;}
 }
 if(tens_block->ndev > tens_block->dev_rsc_len){tens_block->ndev=tens_block->dev_rsc_len; errc=TALSH_FAILURE;}
 if(tens_block->dev_rsc != NULL){
  for(j=0;j<tens_block->ndev;++j){
   i=tensDevRsc_release_all(&(tens_block->dev_rsc[j]));
   if(i == 0 || i == NOT_CLEAN){if(errc == 0) errc=i;}else{errc=TALSH_FAILURE;}
  }
  free(tens_block->dev_rsc); tens_block->dev_rsc=NULL;
 }
 if(tens_block->data_kind != NULL){free(tens_block->data_kind); tens_block->data_kind=NULL;}
 if(tens_block->avail != NULL){free(tens_block->avail); tens_block->avail=NULL;}
 i=talshTensorClean(tens_block); //set to an empty status
 return errc;
}

int talshTensorDestroy(talsh_tens_t * tens_block) //in: non-NULL pointer to a tensor block
/** Completely destroys a talsh_tens_t object. **/
{
 int errc;
 if(tens_block == NULL) return TALSH_INVALID_ARGS;
 errc=talshTensorDestruct(tens_block);
 free(tens_block);
 return errc;
}

int talshTensorRank(const talsh_tens_t * tens_block)
/** Returns the tensor block rank (number of dimensions). **/
{
 return tensShape_rank(tens_block->shape_p);
}

size_t talshTensorVolume(const talsh_tens_t * tens_block) //in: tensor block
/** Returns the total number of elements in the tensor block.
    0 on return means the tensor block is empty. **/
{
 if(tens_block == NULL) return TALSH_INVALID_ARGS;
 if(talshTensorIsEmpty(tens_block) != NOPE) return 0;
 return tensShape_volume(tens_block->shape_p);
}

int talshTensorShape(const talsh_tens_t * tens_block, talsh_tens_shape_t * tens_shape)
/** Returns the shape of the tensor block. The tensor shape object <tens_shape>
    passed here must either be either empty defined or value defined. It is errorneous
    to pass an undefined <tens_shape> object. **/
{
 int errc;

 if(tens_block == NULL || tens_shape == NULL) return TALSH_INVALID_ARGS;
 if(talshTensorIsEmpty(tens_block) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 errc=tensShape_construct(tens_shape,NOPE,tens_block->shape_p->num_dim,tens_block->shape_p->dims, //NOPE: not pinned
                          tens_block->shape_p->divs,tens_block->shape_p->grps);
 if(errc) errc=TALSH_FAILURE;
 return errc;
}

int talshTensorDataKind(const talsh_tens_t * tens_block, int * num_images, int * data_kinds)
/** Returns the data kind of each tensor image. **/
{
 int i;

 if(tens_block == NULL || num_images == NULL || data_kinds == NULL) return TALSH_INVALID_ARGS;
 if(talshTensorIsEmpty(tens_block) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 *num_images=tens_block->ndev;
 for(i=0;i<(*num_images);++i) data_kinds[i]=tens_block->data_kind[i];
 return TALSH_SUCCESS;
}

int talshTensorInUse(const talsh_tens_t * tens_block)
/** Returns YEP is the tensor block is currently in use, NOPE otherwise.
    In case of an error, an error code is returned. **/
{
 int i;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(tens_block == NULL) return TALSH_INVALID_ARGS;
 if(talshTensorIsEmpty(tens_block) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 if(talshTensorIsHealthy(tens_block) != YEP) return TALSH_FAILURE;
 for(i=0;i<tens_block->ndev;++i) if(tens_block->avail[i] != YEP) return YEP;
 return NOPE;
}

int talshTensorPresence(const talsh_tens_t * tens_block, int * ncopies, int copies[], int data_kinds[], int dev_kind, int dev_id)
/** Returns the list of devices on which a copy of the tensor block resides, together with the data kind.
    The presence of optional <dev_kind> and <dev_id> arguments further customizes the search,
    making it look only for copies on the specified device kind and/or device. **/
{
 int i,j,m,devnum,devk,specific_kind,specific_device;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 *ncopies=0; devk=DEV_NULL; devnum=-1;
 if(tens_block == NULL) return TALSH_INVALID_ARGS;
 if(talshTensorIsEmpty(tens_block) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 if(talshTensorIsHealthy(tens_block) != YEP) return TALSH_FAILURE;
 if(valid_device_kind(dev_kind) != YEP) return TALSH_INVALID_ARGS;
 if(dev_kind == DEV_NULL){
  if(dev_id >= 0){
   devnum=talshKindDevId(dev_id,&devk); if(devnum < 0) return TALSH_INVALID_ARGS;
   specific_kind=1; specific_device=1;
  }else{
   specific_kind=0; specific_device=0;
  }
 }else{
  specific_kind=1; devk=dev_kind;
  if(dev_id >= 0){
   devnum=talshFlatDevId(dev_kind,dev_id); if(devnum >= DEV_MAX) return TALSH_INVALID_ARGS;
   specific_device=1; devnum=dev_id;
  }else{
   specific_device=0;
  }
 }
 if(tens_block->ndev > 0){
  for(i=0;i<tens_block->ndev;++i){
   j=talshKindDevId(tens_block->dev_rsc[i].dev_id,&m); if(j < 0) return TALSH_FAILURE;
   if(tens_block->avail[i] == YEP){ //images that are no longer available are skipped
    if((m == devk || specific_kind == 0) && (j == devnum || specific_device == 0)){
     copies[*ncopies]=tens_block->dev_rsc[i].dev_id; data_kinds[*ncopies]=tens_block->data_kind[i];
     ++(*ncopies);
    }
   }
  }
 }
 return TALSH_SUCCESS;
}

int talshTensorPresence_(const talsh_tens_t * tens_block, int * ncopies, int copies[], int data_kinds[], int dev_kind, int dev_id) //Fortran wrapper
{
 return talshTensorPresence(tens_block, ncopies, copies, data_kinds, dev_kind, dev_id);
}

int talshTensorGetBodyAccess(talsh_tens_t * tens_block,
                             void ** body_p,
                             int data_kind,
                             int dev_id,
                             int dev_kind)
/** Based on the requested data kind and device, returns a pointer to the body
    of the matching tensor image (if any). If no match, TALSH_NOT_FOUND is returned.
    Upon success, all other tensor images will be discarded. **/
{
 int i,errc;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(tens_block == NULL || body_p == NULL) return TALSH_INVALID_ARGS;
 *body_p=NULL;
 if(talshTensorIsEmpty(tens_block) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 if(talshTensorIsHealthy(tens_block) != YEP) return TALSH_FAILURE;
 if(talshTensorInUse(tens_block) != NOPE) return TALSH_NOT_ALLOWED;
 if(dev_kind != DEV_NULL) dev_id=talshFlatDevId(dev_kind,dev_id);
 if(dev_id >= 0 && dev_id < DEV_MAX){
  for(i=0;i<tens_block->ndev;++i){
   if(tens_block->dev_rsc[i].dev_id == dev_id && tens_block->data_kind[i] == data_kind){
    *body_p=tens_block->dev_rsc[i].gmem_p;
    errc=talsh_tensor_image_discard_other(tens_block,i);
    if(errc != TALSH_SUCCESS) errc=TALSH_FAILURE;
    return errc;
   }
  }
 }else{
  return TALSH_INVALID_ARGS;
 }
 return TALSH_NOT_FOUND;
}

int talshTensorGetBodyAccess_(talsh_tens_t * tens_block, void ** body_p, int data_kind, int dev_id, int dev_kind)
{
 return talshTensorGetBodyAccess(tens_block,body_p,data_kind,dev_id,dev_kind);
}

int talshTensorGetBodyAccessConst(const talsh_tens_t * tens_block,
                                  const void ** body_p,
                                  int data_kind,
                                  int dev_id,
                                  int dev_kind)
/** Based on the requested data kind and device, returns a constant pointer to the body
    of the matching tensor image (if any). If no match, TALSH_NOT_FOUND is returned. **/
{
 int i,errc;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(tens_block == NULL || body_p == NULL) return TALSH_INVALID_ARGS;
 errc=TALSH_SUCCESS; *body_p=NULL;
 if(talshTensorIsEmpty(tens_block) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 if(talshTensorIsHealthy(tens_block) != YEP) return TALSH_FAILURE;
 if(talshTensorInUse(tens_block) != NOPE) return TALSH_NOT_ALLOWED;
 if(dev_kind != DEV_NULL) dev_id=talshFlatDevId(dev_kind,dev_id);
 if(dev_id >= 0 && dev_id < DEV_MAX){
  for(i=0;i<tens_block->ndev;++i){
   if(tens_block->dev_rsc[i].dev_id == dev_id && tens_block->data_kind[i] == data_kind){
    *body_p=tens_block->dev_rsc[i].gmem_p;
    return errc;
   }
  }
 }else{
  return TALSH_INVALID_ARGS;
 }
 return TALSH_NOT_FOUND;
}

int talshTensorGetScalar(talsh_tens_t * tens_block, talshComplex8 * scalar_complex)
{
 int errc;
 double sreal,simag;

 errc=talshTensorGetScalar_(tens_block,&sreal,&simag);
 if(errc == TALSH_SUCCESS) *scalar_complex = talshComplex8Set(sreal,simag);
 return errc;
}

int talshTensorGetScalar_(talsh_tens_t * tens_block, double * scalar_real, double * scalar_imag)
{
 int errc,i,j,n,dh,dev[TALSH_MAX_DEV_PRESENT],dtk[TALSH_MAX_DEV_PRESENT];
 void * body_p;
 talshComplex4 cx4;
 talshComplex8 cx8;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(tens_block == NULL || scalar_real == NULL || scalar_imag == NULL) return TALSH_INVALID_ARGS;
 if(talshTensorIsEmpty(tens_block) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 if(talshTensorIsHealthy(tens_block) != YEP) return TALSH_FAILURE;
 if(talshTensorRank(tens_block) != 0) return TALSH_INVALID_ARGS;
 dh=talshFlatDevId(DEV_HOST,0); j=-1;
 errc=talshTensorPresence(tens_block,&n,dev,dtk);
 if(errc == TALSH_SUCCESS && n > 0){
  for(i=0; i<n; ++i){if(dev[i] == dh){j=i; break;}}
  if(j < 0){
   errc=talshTensorPlace(tens_block,0,DEV_HOST);
   if(errc == TALSH_SUCCESS){
    errc=talshTensorPresence(tens_block,&n,dev,dtk);
    if(errc == TALSH_SUCCESS && n > 0){
     for(i=0; i<n; ++i){if(dev[i] == dh){j=i; break;}}
     if(j < 0) errc=TALSH_FAILURE;
    }else{
     errc=TALSH_FAILURE;
    }
   }
  }
  if(errc == TALSH_SUCCESS){
   errc=talshTensorGetBodyAccess(tens_block,&body_p,dtk[j],0,DEV_HOST);
   if(errc == TALSH_SUCCESS){
    switch(dtk[j]){
     case R4: *scalar_real = (double)(*((float*)body_p)); *scalar_imag = 0.0; break;
     case R8: *scalar_real = *((double*)body_p); *scalar_imag = 0.0; break;
     case C4: cx4 = *((talshComplex4*)body_p); *scalar_real = (double)talshComplex4Real(cx4); *scalar_imag = (double)talshComplex4Imag(cx4); break;
     case C8: cx8 = *((talshComplex8*)body_p); *scalar_real = talshComplex8Real(cx8); *scalar_imag = talshComplex8Imag(cx8); break;
    }
   }
  }
 }else{
  errc=TALSH_FAILURE;
 }
 return errc;
}

static int talshTensorIsHealthy(const talsh_tens_t * talsh_tens)
/** Returns YEP is the TAL-SH tensor is fine, NOPE otherwise. A return
    status TALSH_OBJECT_IS_EMPTY indicates that the tensor is empty.
    Note that this function assumes at least one tensor body image
    to be present for healthy tensors. **/
{
 int errc;

 if(talsh_tens == NULL) return TALSH_INVALID_ARGS;
 errc=talshTensorIsEmpty(talsh_tens);
 if(errc == NOPE){
  if(talsh_tens->dev_rsc == NULL || talsh_tens->data_kind == NULL || talsh_tens->avail == NULL ||
     talsh_tens->ndev <= 0 || talsh_tens->ndev > talsh_tens->dev_rsc_len) return NOPE;
 }else if(errc == YEP){
  return TALSH_OBJECT_IS_EMPTY;
 }else{
  return NOPE;
 }
 return YEP;
}

void talshTensorPrintInfo(const talsh_tens_t * tens_block)
/** Prints information about a TAL-SH tensor. **/
{
 int i,dvn,dvk;

 if(tens_block != NULL){
  printf("\n#MESSAGE: Printing TAL-SH tensor info:\n");
  printf(" Tensor block address: %p\n",tens_block);
  printf(" Tensor block shape:\n");
  printf("  Tensor block rank: %d\n",tens_block->shape_p->num_dim);
  if(tens_block->shape_p->num_dim > 0){
   printf("  Tensor block dimension extents:");
   for(i=0;i<tens_block->shape_p->num_dim;++i) printf(" %d",tens_block->shape_p->dims[i]);
  }
  printf("\n Tensor block presence ([dev_kind,dev_id|data_kind|avail]):");
  for(i=0; i < tens_block->ndev; ++i){
   dvn=talshKindDevId(tens_block->dev_rsc[i].dev_id,&dvk);
   printf(" [%d,%d|%d|%d]",dvk,dvn,tens_block->data_kind[i],tens_block->avail[i]);
  }
  printf("\n#END OF MESSAGE\n");
 }else{
  printf("\n#WARNING(talshc:talshTensorPrintInfo): NULL pointer!\n");
 }
 return;
}

void talshTensorPrintBody(const talsh_tens_t * tens_block, double thresh)
/** Prints tensor elements larger by absolute value than some threshold.
    Only CPU resident tensor images can be printed.  **/
{
 int errc,n,devs[TALSH_MAX_DEV_PRESENT],dtks[TALSH_MAX_DEV_PRESENT];
 unsigned int mlndx[MAX_TENSOR_RANK],i,nd;
 unsigned int * tdims;
 const void * body_p;
 size_t l,vol;
 talsh_tens_shape_t tshape;
 const float * bpr4;
 const double * bpr8;
 const talshComplex4 * bpc4;
 const talshComplex8 * bpc8;

 printf("\n#MSG: Printing tensor body:");
 if(tens_block != NULL){
  if(talshTensorIsEmpty(tens_block) == NOPE){
   errc=talshTensorPresence(tens_block,&n,devs,dtks,DEV_HOST);
   if(errc == TALSH_SUCCESS && n > 0){
    errc=talshTensorGetBodyAccessConst(tens_block,&body_p,dtks[0],0,DEV_HOST);
    printf("ERROR CODE = %d",errc); //debug
    if(errc == TALSH_SUCCESS){
     vol=talshTensorVolume(tens_block);
     errc=tensShape_clean(&tshape);
     errc=talshTensorShape(tens_block,&tshape);
     if(errc == TALSH_SUCCESS){
      nd=(unsigned int)(tshape.num_dim);
      tdims=(unsigned int *)(tshape.dims);
      switch(dtks[0]){
       case R4:
        bpr4=(const float *)body_p;
        if(nd > 0){
         for(l=0;l<vol;++l){
          if((double)(ABS(bpr4[l])) >= thresh){
           tens_elem_mlndx_f(l,nd,tdims,mlndx);
           printf("\n%E",bpr4[l]); for(i=0;i<nd;++i) printf(" %u",mlndx[i]);
          }
         }
        }else{ //scalar
         if((double)(ABS(bpr4[0])) >= thresh) printf("\n%E",bpr4[0]);
        }
        break;
       case R8:
        bpr8=(const double *)body_p;
        if(nd > 0){
         for(l=0;l<vol;++l){
          if(ABS(bpr8[l]) >= thresh){
           tens_elem_mlndx_f(l,nd,tdims,mlndx);
           printf("\n%E",bpr8[l]); for(i=0;i<nd;++i) printf(" %u",mlndx[i]);
          }
         }
        }else{ //scalar
         if(ABS(bpr8[0]) >= thresh) printf("\n%E",bpr8[0]);
        }
        break;
       case C4:
        bpc4=(const talshComplex4 *)body_p;
        if(nd > 0){
         for(l=0;l<vol;++l){
          if((double)(talshComplex4Abs(bpc4[l])) >= thresh){
           tens_elem_mlndx_f(l,nd,tdims,mlndx);
           printf("\n(%E,%E)",talshComplex4Real(bpc4[l]),talshComplex4Imag(bpc4[l])); for(i=0;i<nd;++i) printf(" %u",mlndx[i]);
          }
         }
        }else{ //scalar
         if(talshComplex4Abs(bpc4[0]) >= thresh) printf("\n(%E,%E)",talshComplex4Real(bpc4[0]),talshComplex4Imag(bpc4[0]));
        }
        break;
       case C8:
        bpc8=(const talshComplex8 *)body_p;
        if(nd > 0){
         for(l=0;l<vol;++l){
          if(talshComplex8Abs(bpc8[l]) >= thresh){
           tens_elem_mlndx_f(l,nd,tdims,mlndx);
           printf("\n(%E,%E)",talshComplex8Real(bpc8[l]),talshComplex8Imag(bpc8[l])); for(i=0;i<nd;++i) printf(" %u",mlndx[i]);
          }
         }
        }else{ //scalar
         if((double)(talshComplex8Abs(bpc8[0])) >= thresh) printf("\n(%E,%E)",talshComplex8Real(bpc8[0]),talshComplex8Imag(bpc8[0]));
        }
        break;
      }
      printf("\n#END MSG\n");
     }else{
      printf("\n#WARNING(talshc:talshTensorPrintBody): Failed to obtain tensor shape!\n");
     }
    }else{
     printf("\n#WARNING(talshc:talshTensorPrintBody): No tensor body access!\n");
    }
   }else{
    printf("\n#WARNING(talshc:talshTensorPrintBody): No tensor image found on Host!\n");
   }
  }else{
   printf("\n#WARNING(talshc:talshTensorPrintBody): Empty tensor!\n");
  }
 }else{
  printf("\n#WARNING(talshc:talshTensorPrintBody): NULL pointer!\n");
 }
 return;
}

// TAL-SH task API:
int talshTaskCreate(talsh_task_t ** talsh_task)
/** Creates a clean <talsh_task_t> object on heap. **/
{
 *talsh_task=(talsh_task_t*)malloc(sizeof(talsh_task_t));
 if(*talsh_task == NULL) return TRY_LATER;
 return talshTaskClean(*talsh_task);
}

int talshTaskClean(talsh_task_t * talsh_task)
/** Cleans an undefined (statically allocated) <talsh_task_t> object making it defined-empty.
    Never call this function on value-defined <talsh_task_t> objects. **/
{
 talsh_task->task_p=NULL;
 talsh_task->task_error=-1;
 talsh_task->dev_kind=DEV_NULL;
 talsh_task->data_kind=NO_TYPE;
 talsh_task->coherence=-1;
 talsh_task->num_args=0;
 for(int i=0; i<MAX_TENSOR_OPERANDS; ++i){
  talsh_task->tens_args[i].tens_p=NULL;
  talsh_task->tens_args[i].source_image=-1;
 }
 talsh_task->data_vol=0.0;
 talsh_task->flops=0.0;
 talsh_task->exec_time=0.0;
 return TALSH_SUCCESS;
}

int talshTaskIsEmpty(const talsh_task_t * talsh_task)
/** Returns YEP if the task is empty, NOPE otherwise (for internal use). **/
{
 if(talsh_task != NULL){
  if(talsh_task->dev_kind >= 0){
   //talsh_task->task_p == NULL) return TALSH_FAILURE;
   return NOPE;
  }else{
   if(talsh_task->task_p != NULL) return TALSH_FAILURE;
  }
 }else{
  return YEP;
 }
 return YEP;
}

static int talshTaskConstruct(talsh_task_t * talsh_task, int dev_kind, int coh_ctrl, int data_kind)
/** Constructs a TAL-SH task. It is errorneous to pass undefined <talsh_task_t> objects here!
    At the same time, it is fine to pass a value-defined <talsh_task_t> object here because
    it will be destructed before the new construction. Note that this function does not set
    tensor arguments for the task, which is done by the talshTaskSetArg() function. **/
{
 int i,errc;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 errc=TALSH_SUCCESS;
 if(talsh_task == NULL) return TALSH_INVALID_ARGS;
 if(valid_device_kind(dev_kind) != YEP) return TALSH_INVALID_ARGS;
 if(tens_valid_data_kind(data_kind) != YEP) return TALSH_INVALID_ARGS;
 if(talshTaskIsEmpty(talsh_task) != YEP) errc=talshTaskDestruct(talsh_task); //destruct value-defined tasks first
 if(errc != TALSH_SUCCESS && errc != NOT_CLEAN) return TALSH_FAILURE; if(errc == NOT_CLEAN) talsh_raise_not_clean();
 switch(dev_kind){
  case DEV_HOST:
   i=host_task_create((host_task_t**)(&(talsh_task->task_p)));
   if(i != 0){
    errc=talshTaskClean(talsh_task);
    if(i == TRY_LATER || i == DEVICE_UNABLE){return i;}else{return TALSH_FAILURE;} //overwrites previous NOT_CLEAN status
   }
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   i=cuda_task_create((cudaTask_t**)(&(talsh_task->task_p)));
   if(i != 0){
    errc=talshTaskClean(talsh_task);
    if(i == TRY_LATER || i == DEVICE_UNABLE){return i;}else{return TALSH_FAILURE;} //overwrites previous NOT_CLEAN status
   }
#else
   return TALSH_NOT_AVAILABLE;
#endif
   break;
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  case DEV_AMD_GPU:
#ifndef NO_AMD
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  default:
   return TALSH_INVALID_ARGS;
 }
 talsh_task->task_error=-1;
 talsh_task->dev_kind=dev_kind;
 talsh_task->data_kind=data_kind;
 talsh_task->coherence=coh_ctrl;
 talsh_task->num_args=0;
 return errc;
}

static int talshTaskSetArg(talsh_task_t * talsh_task, talsh_tens_t * talsh_tens_p, int image_id)
/** Sets up a tensor argument for a given TAL-SH task. The tensor arguments must be set up in order,
    starting from the destination tensor and proceeding to the right. Each time a new tensor argument
    is passed here to be associated with the task, it will be appended as the next argument on the right.
    The TAL-SH task passed here must have already been constructed by talshTaskConstruct()! **/
{
 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(talsh_task == NULL) return TALSH_INVALID_ARGS;
 if(talshTaskIsEmpty(talsh_task) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 if(talsh_tens_p == NULL) return TALSH_INVALID_ARGS;
 if(image_id < 0 || image_id >= talsh_tens_p->ndev) return TALSH_INVALID_ARGS;
 if(talsh_task->num_args < 0 || talsh_task->num_args >= MAX_TENSOR_OPERANDS) return TALSH_FAILURE;
 talsh_task->tens_args[talsh_task->num_args].tens_p=talsh_tens_p;
 talsh_task->tens_args[talsh_task->num_args].source_image=image_id;
 ++(talsh_task->num_args);
 return TALSH_SUCCESS;
}

static int talshTaskFinalize(talsh_task_t * talsh_task, int task_status)
/** Applies top-level coherence control to a recorded TAL-SH task (discards/unmarks images):
    TALSH_TASK_COMPLETED activates "Discard", TALSH_TASK_ERROR activates "Unmark".
    Note that the device-specific task object, if present, is not destroyed here.
    If present, the device-specific task object is finalized. **/
{
 const unsigned int TWO_BITS_SET=3;
 int i,j,image_id,src_dev_id,errc,discard_device_aliases,coherence_control;
 unsigned int coh,cc;
 talsh_tens_t * talsh_tens;
 host_task_t * host_task;
#ifndef NO_GPU
 cudaTask_t * cuda_task;
#endif

 if(talsh_task == NULL) return TALSH_INVALID_ARGS;
 if(talshTaskIsEmpty(talsh_task) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 if(valid_device_kind(talsh_task->dev_kind) != YEP) return TALSH_FAILURE;
 if(talsh_task->num_args < 0 || talsh_task->num_args > MAX_TENSOR_OPERANDS) return TALSH_FAILURE;
 if(talsh_task->task_error >= 0) return TALSH_SUCCESS; //already finalized
 errc=TALSH_SUCCESS;
 //Determine the necessary finalization actions:
 discard_device_aliases=0; coherence_control=0;
 if(task_status == TALSH_TASK_COMPLETED){ //successfully completed task
  discard_device_aliases=1; coherence_control=1;
 }else if(task_status == TALSH_TASK_ERROR){ //failed task
  if(talsh_task->task_p != NULL){ //execution failure (device task is complete)
   discard_device_aliases=1;
  }
 }else{
  return TALSH_INVALID_ARGS;
 }
 //Coherence control:
 coh=talsh_task->coherence;
 for(i=talsh_task->num_args-1;i>=0;--i){ //loop over the tensor arguments
  cc=(coh)&(TWO_BITS_SET); coh=coh>>2;
  talsh_tens=talsh_task->tens_args[i].tens_p; image_id=talsh_task->tens_args[i].source_image;
  if(talsh_tens != NULL){
   if(talshTensorIsEmpty(talsh_tens) == NOPE && talshTensorIsHealthy(talsh_tens) == YEP){
    if(image_id >= 0 && image_id < talsh_tens->ndev){
     src_dev_id=tensDevRsc_device_id(&(talsh_tens->dev_rsc[image_id]));
     if(coherence_control != 0 || discard_device_aliases != 0){
      switch(talsh_task->dev_kind){
       case DEV_HOST: //Host: Destination images are created explicitly in the TAL-SH operation and tensor aliases are destroyed there as well
        host_task=(host_task_t*)(talsh_task->task_p);
        talsh_task->task_error=host_task_error_code(host_task);
        break;
       case DEV_NVIDIA_GPU:
#ifndef NO_GPU
        cuda_task=(cudaTask_t*)(talsh_task->task_p);
        //Append the newly formed destination image if needed:
        if(cc == COPY_M || cc == COPY_K){
         ++(talsh_tens->ndev); if(talsh_tens->ndev > talsh_tens->dev_rsc_len) return TALSH_LIMIT_EXCEEDED;
         j=cuda_task_dev_rsc_move(cuda_task,(unsigned int)i,'d',&(talsh_tens->dev_rsc[talsh_tens->ndev-1]));
         if(j == 0){
          if(tensDevRsc_device_id(&(talsh_tens->dev_rsc[talsh_tens->ndev-1])) != src_dev_id){
           talsh_tens->data_kind[talsh_tens->ndev-1]=talsh_task->data_kind;
           talsh_tens->avail[talsh_tens->ndev-1]=YEP;
          }else{
           if(tensDevRsc_same(&(talsh_tens->dev_rsc[image_id]),&(talsh_tens->dev_rsc[talsh_tens->ndev-1])) != YEP) return TALSH_FAILURE;
           --(talsh_tens->ndev);
          }
         }else{
          --(talsh_tens->ndev); errc=TALSH_FAILURE;
         }
        }
        //Discard device-specific tensor alias:
        if(discard_device_aliases){
         j=cuda_task_arg_destroy(cuda_task,i); if(j == NOT_CLEAN){errc=j;}else{if(j != 0) errc=TALSH_FAILURE;}
        }
        talsh_task->task_error=cuda_task_error_code(cuda_task);
#else
        return TALSH_NOT_AVAILABLE;
#endif
        break;
       case DEV_INTEL_MIC:
#ifndef NO_PHI
        return TALSH_NOT_IMPLEMENTED; //`Future
#else
        return TALSH_NOT_AVAILABLE;
#endif
        //break;
       case DEV_AMD_GPU:
#ifndef NO_AMD
        return TALSH_NOT_IMPLEMENTED; //`Future
#else
        return TALSH_NOT_AVAILABLE;
#endif
        //break;
       default:
        errc=TALSH_FAILURE;
      }
     }
     //Discard/unmark the source image if needed:
     if(cc == COPY_D || cc == COPY_M){ //source to be discarded
      if(talsh_tens->avail[image_id] == NOPE){ //discard the source tensor body image only if it is marked for discarding
       if(coherence_control){
        j=talsh_tensor_image_discard(talsh_tens,image_id); if(j != 0 && errc == TALSH_SUCCESS) errc=NOT_CLEAN; //discard image
        src_dev_id=DEV_NULL;
       }else{
        talsh_tens->avail[image_id]=YEP; //unmark image (make it available again)
       }
      }
     }
    }else{
     errc=TALSH_FAILURE;
    }
   }else{
    errc=TALSH_FAILURE;
   }
  }else{
   errc=TALSH_FAILURE;
  }
 }
 //Set the TAL-SH task error code to a non-negative value, if not set:
 if(talsh_task->task_error < 0){
  if(task_status == TALSH_TASK_COMPLETED && errc == TALSH_SUCCESS){
   talsh_task->task_error=0;
  }else{
   talsh_task->task_error=13;
  }
 }
 return errc;
}

int talshTaskDestruct(talsh_task_t * talsh_task)
/** Destructs a TAL-SH task, putting it back into the defined-empty (clean) state. **/
{
 int i,errc;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 errc=TALSH_SUCCESS;
 if(talsh_task == NULL) return TALSH_INVALID_ARGS;
 i=talshTaskStatus(talsh_task);
 if(i == TALSH_TASK_EMPTY) return TALSH_SUCCESS;
 if(i != TALSH_TASK_COMPLETED && i != TALSH_TASK_ERROR) return TALSH_IN_PROGRESS;
 if(i == TALSH_TASK_COMPLETED && talsh_task->task_p == NULL) return TALSH_INVALID_ARGS;
 switch(talsh_task->dev_kind){
  case DEV_HOST:
   if(talsh_task->task_p != NULL){
    errc=host_task_destroy((host_task_t*)(talsh_task->task_p));
    if(errc != 0 && errc != TRY_LATER && errc != NOT_CLEAN) errc=TALSH_FAILURE;
   }
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   if(talsh_task->task_p != NULL){
    errc=cuda_task_destroy((cudaTask_t*)(talsh_task->task_p));
    if(errc != 0 && errc != TRY_LATER && errc != NOT_CLEAN) errc=TALSH_FAILURE;
   }
#else
   return TALSH_NOT_AVAILABLE;
#endif
   break;
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  case DEV_AMD_GPU:
#ifndef NO_AMD
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  case DEV_NULL: //defined-empty task
   break;
  default:
   return TALSH_INVALID_ARGS;
 }
 i=talshTaskClean(talsh_task);
 return errc;
}

int talshTaskDestroy(talsh_task_t * talsh_task)
/** Completely destroys a <talsh_task_t> object. **/
{
 int errc;

 if(talsh_task == NULL) return TALSH_INVALID_ARGS;
 errc=talshTaskDestruct(talsh_task);
 free(talsh_task);
 return errc;
}

int talshTaskDevId(talsh_task_t * talsh_task, int * dev_kind)
/** Returns either a flat (<dev_kind> is absent) or kind-specific (<dev_kind> is present)
    device id on which the TAL-SH task is scheduled. DEV_NULL on return means an error. **/
{
 int devid,errc;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(talsh_task == NULL) return DEV_NULL;
 errc=talshTaskStatus(talsh_task);
 if(errc == TALSH_FAILURE || errc == TALSH_TASK_EMPTY) return DEV_NULL;
 switch(talsh_task->dev_kind){
  case DEV_HOST:
   devid=0; //Host is always single
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   devid=cuda_task_gpu_id((cudaTask_t*)(talsh_task->task_p));
   if(devid < 0) return DEV_NULL;
#else
   return DEV_NULL;
#endif
   break;
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   return DEV_NULL; //`Future
#else
   return DEV_NULL;
#endif
   //break;
  case DEV_AMD_GPU:
#ifndef NO_AMD
   return DEV_NULL; //`Future
#else
   return DEV_NULL;
#endif
   //break;
  default:
   return DEV_NULL;
 }
 if(devid < 0) return DEV_NULL;
 if(dev_kind != NULL){
  *dev_kind=talsh_task->dev_kind;
 }else{
  devid=talshFlatDevId(talsh_task->dev_kind,devid); //convert to flat device id
  if(devid < 0 || devid >= DEV_MAX) devid=DEV_NULL;
 }
 return devid;
}

int talshTaskDevId_(talsh_task_t * talsh_task, int * dev_kind) //Fortran wrapper
{
 return talshTaskDevId(talsh_task,dev_kind);
}

int talshTaskStatus(talsh_task_t * talsh_task)
/** Returns the current status of the TAL-SH task or an error status. **/
{
 int i,errc;
 host_task_t *host_task_p;
#ifndef NO_GPU
 cudaTask_t *cuda_task_p;
#endif

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(talsh_task == NULL) return TALSH_INVALID_ARGS;
 if(talsh_task->dev_kind == DEV_NULL) return TALSH_TASK_EMPTY;
 if(talsh_task->task_error >= 0){ //already finalized
  if(talsh_task->task_error > 0) return TALSH_TASK_ERROR;
  return TALSH_TASK_COMPLETED;
 }
 if(talsh_task->task_p == NULL) return TALSH_INVALID_ARGS;
 switch(talsh_task->dev_kind){
  case DEV_HOST:
   host_task_p=((host_task_t*)(talsh_task->task_p));
   if(host_task_is_empty(host_task_p) != NOPE) return TALSH_OBJECT_IS_EMPTY;
   errc=host_task_status(host_task_p);
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   cuda_task_p=(cudaTask_t*)(talsh_task->task_p);
   i=cuda_task_status(cuda_task_p);
   switch(i){
    case CUDA_TASK_ERROR: errc=TALSH_TASK_ERROR; break;
    case CUDA_TASK_EMPTY: errc=TALSH_TASK_EMPTY; break;
    case CUDA_TASK_SCHEDULED: errc=TALSH_TASK_SCHEDULED; break;
    case CUDA_TASK_STARTED: errc=TALSH_TASK_STARTED; break;
    case CUDA_TASK_INPUT_THERE: errc=TALSH_TASK_INPUT_READY; break;
    case CUDA_TASK_OUTPUT_THERE: errc=TALSH_TASK_OUTPUT_READY; break;
    case CUDA_TASK_COMPLETED: errc=TALSH_TASK_COMPLETED; break;
    default:
     errc=TALSH_FAILURE;
   }
#else
   return TALSH_NOT_AVAILABLE;
#endif
   break;
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  case DEV_AMD_GPU:
#ifndef NO_AMD
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  default:
   return TALSH_INVALID_ARGS;
 }
 if(errc == TALSH_TASK_COMPLETED || errc == TALSH_TASK_ERROR){
  i=talshTaskFinalize(talsh_task,errc); if(i) errc=TALSH_TASK_ERROR;
 }
 return errc;
}

int talshTaskComplete(talsh_task_t * talsh_task, int * stats, int * ierr)
/** Returns YEP if the TAL-SH has completed, NOPE otherwise.
    The TAL-SH task status will be returned in <stats>. **/
{
 int i,errc;
 host_task_t *host_task_p;
#ifndef NO_GPU
 cudaTask_t *cuda_task_p;
#endif

 errc=NOPE;
 if(ierr == NULL) return TALSH_INVALID_ARGS;
 if(talsh_on == 0){*ierr=TALSH_NOT_INITIALIZED; return errc;}
 if(talsh_task == NULL || stats == NULL){*ierr=TALSH_INVALID_ARGS; return errc;}
 *ierr=TALSH_SUCCESS;
 if(talsh_task->task_error >= 0){ //already finalized
  if(talsh_task->task_error == 0){*stats=TALSH_TASK_COMPLETED;}else{*stats=TALSH_TASK_ERROR;}
  return YEP;
 }
 if(talsh_task->task_p == NULL){*ierr=TALSH_OBJECT_IS_EMPTY; return errc;}
 switch(talsh_task->dev_kind){
  case DEV_HOST:
   host_task_p=((host_task_t*)(talsh_task->task_p));
   if(host_task_is_empty(host_task_p) != NOPE) return TALSH_OBJECT_IS_EMPTY;
   *stats=host_task_status(host_task_p);
   if(*stats == TALSH_TASK_COMPLETED || *stats == TALSH_TASK_ERROR){errc=YEP;}else{errc=NOPE;}
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   cuda_task_p=(cudaTask_t*)(talsh_task->task_p);
   *stats=cuda_task_completed(cuda_task_p);
   switch(*stats){
    case CUDA_TASK_ERROR: *stats=TALSH_TASK_ERROR; errc=YEP; break;
    case CUDA_TASK_EMPTY: *stats=TALSH_TASK_EMPTY; break;
    case CUDA_TASK_SCHEDULED: *stats=TALSH_TASK_SCHEDULED; break;
    case CUDA_TASK_STARTED: *stats=TALSH_TASK_STARTED; break;
    case CUDA_TASK_INPUT_THERE: *stats=TALSH_TASK_INPUT_READY; break;
    case CUDA_TASK_OUTPUT_THERE: *stats=TALSH_TASK_OUTPUT_READY; break;
    case CUDA_TASK_COMPLETED: *stats=TALSH_TASK_COMPLETED; errc=YEP; break;
    default:
     *stats=TALSH_FAILURE; *ierr=TALSH_FAILURE;
   }
   //if(errc == YEP){printf("\n#DEBUG(TALSH::talshTaskComplete): CUDA task completed:"); cuda_task_print(cuda_task_p);} //debug
#else
   *ierr=TALSH_NOT_AVAILABLE;
#endif
   break;
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   *ierr=TALSH_NOT_IMPLEMENTED; //`Future
#else
   *ierr=TALSH_NOT_AVAILABLE;
#endif
   break;
  case DEV_AMD_GPU:
#ifndef NO_AMD
   *ierr=TALSH_NOT_IMPLEMENTED; //`Future
#else
   *ierr=TALSH_NOT_AVAILABLE;
#endif
   break;
  default:
   *ierr=TALSH_INVALID_ARGS;
 }
 if(errc == YEP){i=talshTaskFinalize(talsh_task,*stats); if(i) *ierr=NOT_CLEAN;}
 return errc;
}

int talshTaskWait(talsh_task_t * talsh_task, int * stats)
/** Returns upon completion of a TAL-SH task. **/
{
 int errc;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(talsh_task == NULL || stats == NULL) return TALSH_INVALID_ARGS;
 errc=TALSH_SUCCESS;
 while(talshTaskComplete(talsh_task,stats,&errc) == NOPE){if(errc != TALSH_SUCCESS) break;};
 return errc;
}

int talshTasksWait(int ntasks, talsh_task_t talsh_tasks[], int stats[])
/** Returns upon completion of a number of TAL-SH tasks. **/
{
 int i,tc,sts,errc;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(ntasks <= 0 || talsh_tasks == NULL || stats == NULL) return TALSH_INVALID_ARGS;
 for(i=0;i<ntasks;++i) stats[i]=TALSH_TASK_EMPTY;
 tc=ntasks; errc=TALSH_SUCCESS;
 while(tc > 0){
  for(i=0;i<ntasks;++i){
   if(talsh_tasks[i].task_p == NULL || talsh_tasks[i].dev_kind == DEV_NULL) return TALSH_OBJECT_IS_EMPTY;
   if(stats[i] == TALSH_TASK_EMPTY){
    if(talshTaskComplete(&(talsh_tasks[i]),&sts,&errc) == YEP){stats[i]=sts; --tc;}
    if(errc != TALSH_SUCCESS) return TALSH_FAILURE;
   }
  }
 }
 return TALSH_SUCCESS;
}

int talshTaskTime(talsh_task_t * talsh_task, double * total, double * comput, double * input, double * output, double * mmul)
/** Returns the timing information for a given TAL-SH task. **/
{
 int sts,errc;
 float tot_tm,in_tm,out_tm,comp_tm,mmul_tm;
#ifndef NO_GPU
 cudaTask_t *cuda_task_p;
#endif

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(talsh_task == NULL || total == NULL) return TALSH_INVALID_ARGS;
 if(talsh_task->task_p == NULL) return TALSH_OBJECT_IS_EMPTY;
 if(talshTaskComplete(talsh_task,&sts,&errc) == NOPE){
  if(errc != TALSH_SUCCESS) return TALSH_FAILURE;
  return TALSH_IN_PROGRESS;
 }
 switch(talsh_task->dev_kind){
  case DEV_HOST:
   tot_tm=(float)(talsh_task->exec_time); in_tm=-1.0f; out_tm=-1.0f; comp_tm=-1.0f; mmul_tm=-1.0f;
   if(tot_tm < 0.0f) errc=TALSH_FAILURE;
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   cuda_task_p=(cudaTask_t*)(talsh_task->task_p);
   tot_tm=cuda_task_time(cuda_task_p,&in_tm,&out_tm,&comp_tm,&mmul_tm);
   if(tot_tm < 0.0f) errc=TALSH_FAILURE;
#else
   return TALSH_NOT_AVAILABLE;
#endif
   break;
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  case DEV_AMD_GPU:
#ifndef NO_AMD
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  default:
   return TALSH_INVALID_ARGS;
 }
 *total=(double)tot_tm;
 if(comput != NULL) *comput=(double)comp_tm;
 if(input != NULL) *input=(double)in_tm;
 if(output != NULL) *output=(double)out_tm;
 if(mmul != NULL) *mmul=(double)mmul_tm;
 return errc;
}

int talshTaskTime_(talsh_task_t * talsh_task, double * total, double * comput, double * input, double * output, double * mmul) //Fortran wrapper
{
 return talshTaskTime(talsh_task,total,comput,input,output,mmul);
}

void talshTaskPrint(const talsh_task_t * talsh_task)
/** Prints TAL-SH task info. **/
{
 if(talsh_task != NULL){
  switch(talsh_task->dev_kind){
   case DEV_HOST:
    host_task_print((host_task_t*)(talsh_task->task_p));
    break;
   case DEV_NVIDIA_GPU:
#ifndef NO_GPU
    cuda_task_print((cudaTask_t*)(talsh_task->task_p));
#endif
    break;
   case DEV_INTEL_MIC:
#ifndef NO_PHI
#endif
    break;
   case DEV_AMD_GPU:
#ifndef NO_AMD
#endif
    break;
  }
 }
 return;
}

// TAL-SH tensor operations API:
int talshTensorPlace(talsh_tens_t * tens, int dev_id, int dev_kind, void * dev_mem, int copy_ctrl, talsh_task_t * talsh_task)
/** Places a tensor block body image on a specific device. **/
{
 int i,j,dn,dk,errc,devid,dvk,dvn,image_id,host_image,runtime;
 talsh_task_t * tsk;
 host_task_t * host_task;
#ifndef NO_GPU
 cudaTask_t * cuda_task;
 tensBlck_t * ctens;
#endif

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 //Create a TAL-SH task:
 if(talsh_task == NULL){
  errc=talshTaskCreate(&tsk); if(errc) return errc; if(tsk == NULL) return TALSH_FAILURE;
 }else{
  tsk=talsh_task;
 }
 //Check function arguments:
 if(tens == NULL){tsk->task_error=100; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;}
 if(talshTensorIsEmpty(tens) != NOPE){tsk->task_error=101; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_OBJECT_IS_EMPTY;}
 if(talshTensorIsHealthy(tens) != YEP){tsk->task_error=102; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
 if(dev_kind == DEV_DEFAULT){devid=dev_id;}else{devid=talshFlatDevId(dev_kind,dev_id);}
 dvn=talshKindDevId(devid,&dvk); if(dvn < 0){tsk->task_error=103; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;} //[dvk,dvn]: destination device
 if(copy_ctrl < 0 || copy_ctrl == COPY_D || copy_ctrl == COPY_T){ //'Discard' and 'Temporary' do not make sense here
  tsk->task_error=104; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;
 }
 errc=TALSH_SUCCESS;
 //Find the source tensor body image:
 image_id=-1; host_image=-1;
 for(i=0;i<tens->ndev;++i){
  if(tens->avail[i] == YEP){ //no longer available images are not considered
   dn=talshKindDevId(tens->dev_rsc[i].dev_id,&dk);
   if(dn < 0){tsk->task_error=105; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
   if(dk == dvk){image_id=i; if(dn == dvn) break;} //`Unless exact match, the last device of the given kind will always be selected
   if(dk == DEV_HOST) host_image=i;
  }
 }
 if(dvk != DEV_HOST){ //the destination device is an accelerator
  if(image_id < 0){ //no device of requested kind holds an image
   if(host_image < 0){ //image is absent on Host as well => a blocking copy will be required
    errc=talshTensorPlace(tens,0,DEV_HOST,dev_mem,copy_ctrl); //clone/move the image to Host (blocking call!)
    if(errc != TALSH_SUCCESS){tsk->task_error=106; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return errc;}
    image_id=tens->ndev-1; //the last image is now residing on Host
    if(tens->dev_rsc[image_id].dev_id != talshFlatDevId(DEV_HOST,0)){
     tsk->task_error=107; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
    }
    if(copy_ctrl == COPY_K) copy_ctrl=COPY_M; //the intermediate image needs to be discarded at the end
   }else{
    image_id=host_image; //an existing image on Host will be used
   }
  }
 }else{ //the destination device is Host: Any device image can be used (if not available on Host)
  if(host_image < 0){image_id=0;}else{image_id=host_image;}
 }
 dn=talshKindDevId(tens->dev_rsc[image_id].dev_id,&dk); //[dk,dn]: source device; [dvk,dvn]: destination device
 //Choose the data transferring runtime:
 if(dvk == DEV_HOST){ //destination is Host
  runtime=dk; //source device will provide the runtime for data transfer
 }else{ //destination is an accelerator
  runtime=dvk; //that accelerator's runtime will be used
 }
 //Construct the TAL-SH task:
 if(talshTaskStatus(tsk) == TALSH_TASK_EMPTY){
  errc=talshTaskConstruct(tsk,runtime,copy_ctrl,tens->data_kind[image_id]);
  if(errc){tsk->task_error=108; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return errc;}
  errc=talshTaskSetArg(tsk,tens,image_id);
  if(errc){tsk->task_error=109; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return errc;}
 }else{
  tsk->task_error=110; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_OBJECT_NOT_EMPTY;
 }
 //Call the device-kind specific data transfer runtime function:
 switch(runtime){
  case DEV_HOST: //destination = Host, Source = Host: Nothing to do (image already there)
   host_task=(host_task_t*)(tsk->task_p);
   errc=host_task_record(host_task,(unsigned int)copy_ctrl,0); //record task success (no coherence control on Host)
   if(errc){tsk->task_error=111; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   errc=talsh_tensor_c_assoc(tens,image_id,&ctens);
   if(errc || ctens == NULL){tsk->task_error=112; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
   cuda_task=(cudaTask_t*)(tsk->task_p);
   if(dvk == DEV_HOST && dvn == 0){ //destination is Host
    j=-1; //Host
   }else if(dvk == DEV_NVIDIA_GPU){ //destination is Nvidia GPU
    j=dvn; //GPU device number
   }else{
    j=talsh_tensor_c_dissoc(ctens); ctens=NULL;
    tsk->task_error=113; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   errc=gpu_tensor_block_place(ctens,j,(unsigned int)copy_ctrl,cuda_task,dev_mem); //if source == destination, no transfer will be initiated (ok)
   if(errc){ //in case of error, CUDA task has already been finalized (with error) without coherence control
    if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=TALSH_FAILURE;
    j=talsh_tensor_c_dissoc(ctens); if(j) errc=TALSH_FAILURE;
    j=cuda_task_destroy(cuda_task); if(j) errc=TALSH_FAILURE;
    tsk->task_error=114; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
    return errc;
   }else{ //coherence control
    if(copy_ctrl == COPY_M){
     if(dn != dvn || dk != dvk) tens->avail[image_id]=NOPE; //mark the image not available since it will be discarded in case of successful execution
    }
   }
   if(talsh_task == NULL){ //blocking call
    errc=talshTaskWait(tsk,&j); if(errc == TALSH_SUCCESS && j != TALSH_TASK_COMPLETED) errc=TALSH_TASK_ERROR;
    j=talsh_tensor_c_dissoc(ctens); if(j) errc=TALSH_FAILURE;
    j=talshTaskDestroy(tsk); if(j != TALSH_SUCCESS && errc == TALSH_SUCCESS) errc=j;
   }
#else
   tsk->task_error=115; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_AVAILABLE;
#endif
   break;
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   tsk->task_error=116; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   tsk->task_error=117; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  case DEV_AMD_GPU:
#ifndef NO_AMD
   tsk->task_error=118; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   tsk->task_error=119; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  default:
   tsk->task_error=120; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_INVALID_ARGS;
 }
 return errc;
}

int talshTensorPlace_(talsh_tens_t * tens, int dev_id, int dev_kind, void * dev_mem, int copy_ctrl, talsh_task_t * talsh_task) //Fortran wrapper
{
 return talshTensorPlace(tens,dev_id,dev_kind,dev_mem,copy_ctrl,talsh_task);
}

int talshTensorDiscard(talsh_tens_t * tens, int dev_id, int dev_kind)
/** Discards a tensor block body image on a specific device. **/
{
 int i,j,k,errc,devid;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 if(tens == NULL) return TALSH_INVALID_ARGS;
 if(talshTensorIsEmpty(tens) != NOPE) return TALSH_OBJECT_IS_EMPTY;
 if(talshTensorIsHealthy(tens) != YEP) return TALSH_FAILURE;
 if(dev_kind == DEV_NULL){devid=dev_id;}else{devid=talshFlatDevId(dev_kind,dev_id);}
 if(devid < 0 || devid >= DEV_MAX) return TALSH_INVALID_ARGS;
 errc=TALSH_SUCCESS;
 k=0;
 for(i=0;i<tens->ndev;++i){
  if(tens->avail[i] == YEP){ //images to be discarded cannot be discarded again
   if(tens->dev_rsc[i].dev_id == devid){
    j=tensDevRsc_release_all(&(tens->dev_rsc[i]));
    if(j != 0 && errc != TALSH_FAILURE){if(j == NOT_CLEAN){errc=NOT_CLEAN;}else{errc=TALSH_FAILURE;}}
   }else{
    if(i > k){
     tens->dev_rsc[k]=tens->dev_rsc[i]; tens->data_kind[k]=tens->data_kind[i]; tens->avail[k]=tens->avail[i];
    }
    ++k;
   }
  }else{
   ++k;
  }
 }
 tens->ndev=k;
 return errc;
}

int talshTensorDiscard_(talsh_tens_t * tens, int dev_id, int dev_kind) //Fortran wrapper
{
 return talshTensorDiscard(tens,dev_id,dev_kind);
}

int talshTensorInit(talsh_tens_t * dtens, double val_real, double val_imag, int dev_id, int dev_kind,
                    int copy_ctrl, talsh_task_t * talsh_task)
/** Tensor initialization dispatcher **/
{
 int j,devid,dvk,dvn,dimg,dcp,errc;
 unsigned int coh_ctrl,coh,cohd;
 talsh_task_t * tsk;
 host_task_t * host_task;
 void *dftr;
 clock_t ctm;
#ifndef NO_GPU
 cudaTask_t * cuda_task;
 tensBlck_t *dctr;
#endif

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 //Create a TAL-SH task:
 if(talsh_task == NULL){
  errc=talshTaskCreate(&tsk); if(errc) return errc; if(tsk == NULL) return TALSH_FAILURE;
 }else{
  tsk=talsh_task;
 }
 coh_ctrl=copy_ctrl;
 //Check function arguments:
 if(dtens == NULL){
  tsk->task_error=100; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;
 }
 if(talshTensorIsEmpty(dtens) != NOPE){
  tsk->task_error=101; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_OBJECT_IS_EMPTY;
 }
 if(talshTensorIsHealthy(dtens) != YEP){
  tsk->task_error=102; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
 }
 //Determine the execution device (devid:[dvk,dvn]):
 if(dev_kind == DEV_DEFAULT){ //device kind is not specified explicitly
  if(dev_id == DEV_DEFAULT){ //neither specific device nor device kind are specified: Find one
   devid=talsh_find_optimal_device(dtens);
   if(devid < 0 || devid >= DEV_MAX){
    tsk->task_error=104; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
  }else{ //<dev_id> is a flat device id
   devid=dev_id;
  }
  dvn=talshKindDevId(devid,&dvk);
  if(dvn < 0){tsk->task_error=105; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;}
 }else{ //device kind is specified explicitly
  if(valid_device_kind(dev_kind) != YEP){
   tsk->task_error=106; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;
  }
  dvk=dev_kind;
  if(dev_id == DEV_DEFAULT){ //kind-specific device id is not specified: Implicit
   dvn=-1; //kind-specific device id will be chosen by the corresponding runtime
  }else{ //kind-specific device id is specified
   dvn=dev_id;
   if(talshFlatDevId(dvk,dvn) >= DEV_MAX){
    tsk->task_error=107; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;
   }
  }
 }
 //The tensor operation will be executed on device of kind <dvk>.
 errc=TALSH_SUCCESS;
 //Choose the tensor body image for each tensor argument and adjust the coherence control:
 cohd=argument_coherence_get_value(coh_ctrl,1,0);
 dimg=talsh_choose_image_for_device(dtens,cohd,&dcp,dvk,dvn);
 if(dcp != 0){ //an intermediate copy was introduced on Host
  if(cohd == COPY_K){ //adjust coherence control
   cohd=COPY_M; j=argument_coherence_set_value(&coh_ctrl,1,0,cohd);
  }else if(cohd == COPY_T){
   cohd=COPY_D; j=argument_coherence_set_value(&coh_ctrl,1,0,cohd);
  }
 }
 if(dimg < 0){
  tsk->task_error=108; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
 }
 //Construct the TAL-SH task:
 if(talshTaskStatus(tsk) == TALSH_TASK_EMPTY){
  errc=talshTaskConstruct(tsk,dvk,coh_ctrl,dtens->data_kind[dimg]);
  if(errc){tsk->task_error=110; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return errc;}
  errc=talshTaskSetArg(tsk,dtens,dimg);
  if(errc){tsk->task_error=111; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return errc;}
 }else{
  tsk->task_error=114; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_OBJECT_NOT_EMPTY;
 }
 //Schedule the tensor operation via the device-kind specific runtime:
 switch(dvk){
  case DEV_HOST:
   //Associate TAL-SH tensor images with <tensor_block_t> objects:
   errc=talsh_tensor_f_assoc(dtens,dimg,&dftr);
   if(errc || dftr == NULL){
    tsk->task_error=115; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   //Get the Host task alias:
   host_task=(host_task_t*)(tsk->task_p);
   //Mark images to be discarded as unavailable:
   devid=talshFlatDevId(DEV_HOST,0); //execution device
   if(cohd == COPY_D || (cohd == COPY_M && dtens->dev_rsc[dimg].dev_id != devid)) dtens->avail[dimg] = NOPE;
   //Schedule the tensor operation via the device-kind specific runtime:
   ctm=clock();
   errc=cpu_tensor_block_init(dftr,val_real,val_imag,0); //blocking call (`no conjugation bits)
   if(talshTensorRank(dtens) == 0){ //an explicit update is needed for scalar destinations
    j=talsh_update_f_scalar(dftr,dtens->data_kind[dimg],dtens->dev_rsc[dimg].gmem_p);
    if(j) errc=TALSH_FAILURE;
   }
   tsk->exec_time=((double)(clock()-ctm))/CLOCKS_PER_SEC;
   //Dissociate <tensor_block_t> objects:
   j=talsh_tensor_f_dissoc(dftr); if(j) errc=TALSH_FAILURE;
   if(errc){
    if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=TALSH_FAILURE;
    j=host_task_record(host_task,coh_ctrl,13);
    j=host_task_destroy(host_task); if(j) errc=TALSH_FAILURE;
    tsk->task_p=NULL;
    tsk->task_error=117; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
    return errc;
   }else{ //coherence control
    errc=host_task_record(host_task,coh_ctrl,0); //record task success (no coherence control on Host)
    if(errc){tsk->task_error=118; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
    errc=talsh_tensor_image_discard_other(dtens,dimg); //discard other images after the update operation
    if(errc){tsk->task_error=121; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
    if(dtens->avail[0] == NOPE){ //only one image left
     errc=talsh_tensor_image_discard(dtens,0);
     if(errc){tsk->task_error=122; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
    }
   }
   //If blocking call, complete it here:
   if(talsh_task == NULL){
    errc=talshTaskWait(tsk,&j); if(errc == TALSH_SUCCESS && j != TALSH_TASK_COMPLETED) errc=TALSH_TASK_ERROR;
    j=talshTaskDestroy(tsk); if(j != TALSH_SUCCESS && errc == TALSH_SUCCESS) errc=j;
   }
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   //Associate TAL-SH tensor images with <tensBlck_t> objects:
   errc=talsh_tensor_c_assoc(dtens,dimg,&dctr);
   if(errc || dctr == NULL){
    tsk->task_error=123; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   //Get the CUDA task alias:
   cuda_task=(cudaTask_t*)(tsk->task_p);
   //Schedule the operation via the device-kind specific runtime:
   errc=gpu_tensor_block_init(dctr,val_real,coh_ctrl,cuda_task,dvn); //non-blocking call
   dvn=cuda_task_gpu_id(cuda_task);
   if(errc || dvn < 0){ //in case of error, CUDA task has already been finalized (with error) without coherence control
    if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=TALSH_FAILURE;
    j=talsh_tensor_c_dissoc(dctr); if(j) errc=TALSH_FAILURE;
    j=cuda_task_destroy(cuda_task); if(j) errc=TALSH_FAILURE;
    tsk->task_p=NULL;
    tsk->task_error=126; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   }else{ //coherence control (mark images to be discarded as unavailable)
    devid=talshFlatDevId(dvk,dvn);
    coh=argument_coherence_get_value(coh_ctrl,1,0);
    if(coh == COPY_D || (coh == COPY_M && dtens->dev_rsc[dimg].dev_id != devid)) dtens->avail[dimg] = NOPE;
   }
   //If blocking call, complete it here:
   if(talsh_task == NULL){
    errc=talshTaskWait(tsk,&j); if(errc == TALSH_SUCCESS && j != TALSH_TASK_COMPLETED) errc=TALSH_TASK_ERROR;
    j=talsh_tensor_c_dissoc(dctr); if(j) errc=TALSH_FAILURE;
    j=talshTaskDestroy(tsk); if(j != TALSH_SUCCESS && errc == TALSH_SUCCESS) errc=j;
   }
#else
   tsk->task_error=127; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_AVAILABLE;
#endif
   break;
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   tsk->task_error=128; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   tsk->task_error=129; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  case DEV_AMD_GPU:
#ifndef NO_AMD
   tsk->task_error=130; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   tsk->task_error=131; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  default:
  tsk->task_error=132; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
 }
 return errc;
}

int talshTensorInit_(talsh_tens_t * dtens, double val_real, double val_imag, int dev_id, int dev_kind,
                     int copy_ctrl, talsh_task_t * talsh_task) //Fortran wrapper
{
 return talshTensorInit(dtens,val_real,val_imag,dev_id,dev_kind,copy_ctrl,talsh_task);
}

int talshTensorAdd(const char * cptrn, talsh_tens_t * dtens, talsh_tens_t * ltens, double scale_real, double scale_imag,
                   int dev_id, int dev_kind, int copy_ctrl, talsh_task_t * talsh_task)
/** Tensor addition dispatcher **/
{
 int j,devid,dvk,dvn,dimg,limg,dcp,lcp,errc;
 int contr_ptrn[MAX_TENSOR_RANK],cpl,conj_bits;
 unsigned int coh_ctrl,coh,cohd,cohl;
 talsh_task_t * tsk;
 host_task_t * host_task;
 void *dftr,*lftr;
 clock_t ctm;
#ifndef NO_GPU
 cudaTask_t * cuda_task;
 tensBlck_t *dctr,*lctr;
#endif

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 //Create a TAL-SH task:
 if(talsh_task == NULL){
  errc=talshTaskCreate(&tsk); if(errc) return errc; if(tsk == NULL) return TALSH_FAILURE;
 }else{
  tsk=talsh_task;
 }
 coh_ctrl=copy_ctrl;
 //Check function arguments:
 if(dtens == NULL || ltens == NULL){
  tsk->task_error=100; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;
 }
 if(talshTensorIsEmpty(dtens) != NOPE || talshTensorIsEmpty(ltens) != NOPE){
  tsk->task_error=101; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_OBJECT_IS_EMPTY;
 }
 if(talshTensorIsHealthy(dtens) != YEP || talshTensorIsHealthy(ltens) != YEP){
  tsk->task_error=102; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
 }
 //Check and parse the index correspondence pattern:
 errc=talsh_get_contr_ptrn_str2dig(cptrn,contr_ptrn,&cpl,&conj_bits);
 if(errc){tsk->task_error=103; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;}
 //Determine the execution device (devid:[dvk,dvn]):
 if(dev_kind == DEV_DEFAULT){ //device kind is not specified explicitly
  if(dev_id == DEV_DEFAULT){ //neither specific device nor device kind are specified: Find one
   devid=talsh_find_optimal_device(dtens,ltens);
   if(devid < 0 || devid >= DEV_MAX){
    tsk->task_error=104; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
  }else{ //<dev_id> is a flat device id
   devid=dev_id;
  }
  dvn=talshKindDevId(devid,&dvk);
  if(dvn < 0){tsk->task_error=105; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;}
 }else{ //device kind is specified explicitly
  if(valid_device_kind(dev_kind) != YEP){
   tsk->task_error=106; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;
  }
  dvk=dev_kind;
  if(dev_id == DEV_DEFAULT){ //kind-specific device id is not specified: Implicit
   dvn=-1; //kind-specific device id will be chosen by the corresponding runtime
  }else{ //kind-specific device id is specified
   dvn=dev_id;
   if(talshFlatDevId(dvk,dvn) >= DEV_MAX){
    tsk->task_error=107; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;
   }
  }
 }
 //The tensor operation will be executed on device of kind <dvk>.
 errc=TALSH_SUCCESS;
 //Choose the tensor body image for each tensor argument and adjust the coherence control:
 cohd=argument_coherence_get_value(coh_ctrl,2,0);
 dimg=talsh_choose_image_for_device(dtens,cohd,&dcp,dvk,dvn);
 if(dcp != 0){ //an intermediate copy was introduced on Host
  if(cohd == COPY_K){ //adjust coherence control
   cohd=COPY_M; j=argument_coherence_set_value(&coh_ctrl,2,0,cohd);
  }else if(cohd == COPY_T){
   cohd=COPY_D; j=argument_coherence_set_value(&coh_ctrl,2,0,cohd);
  }
 }
 cohl=argument_coherence_get_value(coh_ctrl,2,1);
 limg=talsh_choose_image_for_device(ltens,cohl,&lcp,dvk,dvn);
 if(lcp != 0){ //an intermediate copy was introduced on Host
  if(cohl == COPY_K){ //adjust coherence control
   cohl=COPY_M; j=argument_coherence_set_value(&coh_ctrl,2,1,cohl);
  }else if(cohl == COPY_T){
   cohl=COPY_D; j=argument_coherence_set_value(&coh_ctrl,2,1,cohl);
  }
 }
 if(dimg < 0 || limg < 0){
  tsk->task_error=108; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
 }
 //Check data kind of each image (must match):
 if(dtens->data_kind[dimg] != ltens->data_kind[limg]){
  tsk->task_error=109; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;
 }
 //Construct the TAL-SH task:
 if(talshTaskStatus(tsk) == TALSH_TASK_EMPTY){
  errc=talshTaskConstruct(tsk,dvk,coh_ctrl,dtens->data_kind[dimg]);
  if(errc){tsk->task_error=110; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return errc;}
  errc=talshTaskSetArg(tsk,dtens,dimg);
  if(errc){tsk->task_error=111; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return errc;}
  errc=talshTaskSetArg(tsk,ltens,limg);
  if(errc){tsk->task_error=112; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return errc;}
 }else{
  tsk->task_error=114; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_OBJECT_NOT_EMPTY;
 }
 //Schedule the tensor operation via the device-kind specific runtime:
 switch(dvk){
  case DEV_HOST:
   //Associate TAL-SH tensor images with <tensor_block_t> objects:
   errc=talsh_tensor_f_assoc(dtens,dimg,&dftr);
   if(errc || dftr == NULL){
    tsk->task_error=115; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   errc=talsh_tensor_f_assoc(ltens,limg,&lftr);
   if(errc || lftr == NULL){
    errc=talsh_tensor_f_dissoc(dftr);
    tsk->task_error=116; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   //Get the Host task alias:
   host_task=(host_task_t*)(tsk->task_p);
   //Mark images to be discarded as unavailable:
   devid=talshFlatDevId(DEV_HOST,0); //execution device
   if(cohl == COPY_D || (cohl == COPY_M && ltens->dev_rsc[limg].dev_id != devid)) ltens->avail[limg] = NOPE;
   if(cohd == COPY_D || (cohd == COPY_M && dtens->dev_rsc[dimg].dev_id != devid)) dtens->avail[dimg] = NOPE;
   //Schedule the tensor operation via the device-kind specific runtime:
   ctm=clock();
   errc=cpu_tensor_block_add(contr_ptrn,lftr,dftr,scale_real,scale_imag,conj_bits); //blocking call
   if(talshTensorRank(dtens) == 0){ //an explicit update is needed for scalar destinations
    j=talsh_update_f_scalar(dftr,dtens->data_kind[dimg],dtens->dev_rsc[dimg].gmem_p);
    if(j) errc=TALSH_FAILURE;
   }
   tsk->exec_time=((double)(clock()-ctm))/CLOCKS_PER_SEC;
   //Dissociate <tensor_block_t> objects:
   j=talsh_tensor_f_dissoc(lftr); if(j) errc=TALSH_FAILURE;
   j=talsh_tensor_f_dissoc(dftr); if(j) errc=TALSH_FAILURE;
   if(errc){
    if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=TALSH_FAILURE;
    j=host_task_record(host_task,coh_ctrl,13);
    j=host_task_destroy(host_task); if(j) errc=TALSH_FAILURE;
    tsk->task_p=NULL;
    tsk->task_error=117; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
    return errc;
   }else{ //coherence control
    errc=host_task_record(host_task,coh_ctrl,0); //record task success (no coherence control on Host)
    if(errc){tsk->task_error=118; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
    if(ltens->avail[limg] == NOPE){
     errc=talsh_tensor_image_discard(ltens,limg);
     if(errc){tsk->task_error=120; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
    }
    errc=talsh_tensor_image_discard_other(dtens,dimg); //discard other images after the update operation
    if(errc){tsk->task_error=121; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
    if(dtens->avail[0] == NOPE){ //only one image left
     errc=talsh_tensor_image_discard(dtens,0);
     if(errc){tsk->task_error=122; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
    }
   }
   //If blocking call, complete it here:
   if(talsh_task == NULL){
    errc=talshTaskWait(tsk,&j); if(errc == TALSH_SUCCESS && j != TALSH_TASK_COMPLETED) errc=TALSH_TASK_ERROR;
    j=talshTaskDestroy(tsk); if(j != TALSH_SUCCESS && errc == TALSH_SUCCESS) errc=j;
   }
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   //Associate TAL-SH tensor images with <tensBlck_t> objects:
   errc=talsh_tensor_c_assoc(dtens,dimg,&dctr);
   if(errc || dctr == NULL){
    tsk->task_error=123; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   errc=talsh_tensor_c_assoc(ltens,limg,&lctr);
   if(errc || lctr == NULL){
    errc=talsh_tensor_c_dissoc(dctr);
    tsk->task_error=124; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   //Get the CUDA task alias:
   cuda_task=(cudaTask_t*)(tsk->task_p);
   //Schedule the operation via the device-kind specific runtime:
   if(conj_bits != 0){ //`Add complex conjugation feature to NV-TAL
    printf("#FATAL(talshc:talshTensorAdd): Complex conjugation feature is not implemented for GPU target!");
    tsk->task_error=200; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   errc=gpu_tensor_block_add(contr_ptrn,lctr,dctr,coh_ctrl,cuda_task,dvn,scale_real,scale_imag); //non-blocking call
   dvn=cuda_task_gpu_id(cuda_task);
   if(errc || dvn < 0){ //in case of error, CUDA task has already been finalized (with error) without coherence control
    if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=TALSH_FAILURE;
    j=talsh_tensor_c_dissoc(lctr); if(j) errc=TALSH_FAILURE;
    j=talsh_tensor_c_dissoc(dctr); if(j) errc=TALSH_FAILURE;
    j=cuda_task_destroy(cuda_task); if(j) errc=TALSH_FAILURE;
    tsk->task_p=NULL;
    tsk->task_error=126; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   }else{ //coherence control (mark images to be discarded as unavailable)
    devid=talshFlatDevId(dvk,dvn);
    coh=argument_coherence_get_value(coh_ctrl,2,1);
    if(coh == COPY_D || (coh == COPY_M && ltens->dev_rsc[limg].dev_id != devid)) ltens->avail[limg] = NOPE;
    coh=argument_coherence_get_value(coh_ctrl,2,0);
    if(coh == COPY_D || (coh == COPY_M && dtens->dev_rsc[dimg].dev_id != devid)) dtens->avail[dimg] = NOPE;
   }
   //If blocking call, complete it here:
   if(talsh_task == NULL){
    errc=talshTaskWait(tsk,&j); if(errc == TALSH_SUCCESS && j != TALSH_TASK_COMPLETED) errc=TALSH_TASK_ERROR;
    j=talsh_tensor_c_dissoc(lctr); if(j) errc=TALSH_FAILURE;
    j=talsh_tensor_c_dissoc(dctr); if(j) errc=TALSH_FAILURE;
    j=talshTaskDestroy(tsk); if(j != TALSH_SUCCESS && errc == TALSH_SUCCESS) errc=j;
   }
#else
   tsk->task_error=127; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_AVAILABLE;
#endif
   break;
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   tsk->task_error=128; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   tsk->task_error=129; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  case DEV_AMD_GPU:
#ifndef NO_AMD
   tsk->task_error=130; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   tsk->task_error=131; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  default:
  tsk->task_error=132; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
 }
 return errc;
}

int talshTensorAdd_(const char * cptrn, talsh_tens_t * dtens, talsh_tens_t * ltens, double scale_real, double scale_imag,
                    int dev_id, int dev_kind, int copy_ctrl, talsh_task_t * talsh_task) //Fortran wrapper
{
 return talshTensorAdd(cptrn,dtens,ltens,scale_real,scale_imag,dev_id,dev_kind,copy_ctrl,talsh_task);
}

int talshTensorContract(const char * cptrn,        //in: C-string: symbolic contraction pattern, e.g. "D(a,b,c,d)+=L(c,i,j,a)*R(b,j,d,i)"
                        talsh_tens_t * dtens,      //inout: destination tensor block
                        talsh_tens_t * ltens,      //inout: left source tensor block
                        talsh_tens_t * rtens,      //inout: right source tensor block
                        double scale_real,         //in: scaling value (real part), defaults to 1
                        double scale_imag,         //in: scaling value (imaginary part), defaults to 0
                        int dev_id,                //in: device id (flat or kind-specific)
                        int dev_kind,              //in: device kind (if present, <dev_id> is kind-specific)
                        int copy_ctrl,             //in: copy control (COPY_XXX), defaults to COPY_MTT
                        talsh_task_t * talsh_task) //inout: TAL-SH task (must be clean on entrance)
/** Tensor contraction dispatcher. **/
{
 int j,devid,dvk,dvn,dimg,limg,rimg,dcp,lcp,rcp,errc;
 int contr_ptrn[MAX_TENSOR_RANK*2],cpl,conj_bits;
 unsigned int coh_ctrl,coh,cohd,cohl,cohr;
 talsh_task_t * tsk;
 host_task_t * host_task;
 void *dftr,*lftr,*rftr;
 clock_t ctm;
#ifndef NO_GPU
 cudaTask_t * cuda_task;
 tensBlck_t *dctr,*lctr,*rctr;
#endif

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 //Create a TAL-SH task:
 if(talsh_task == NULL){
  errc=talshTaskCreate(&tsk); if(errc) return errc; if(tsk == NULL) return TALSH_FAILURE;
 }else{
  tsk=talsh_task;
 }
 coh_ctrl=copy_ctrl;
 //Check function arguments:
 if(dtens == NULL || ltens == NULL || rtens == NULL){
  tsk->task_error=100; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;
 }
 if(talshTensorIsEmpty(dtens) != NOPE || talshTensorIsEmpty(ltens) != NOPE || talshTensorIsEmpty(rtens) != NOPE){
  tsk->task_error=101; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_OBJECT_IS_EMPTY;
 }
 if(talshTensorIsHealthy(dtens) != YEP || talshTensorIsHealthy(ltens) != YEP || talshTensorIsHealthy(rtens) != YEP){
  tsk->task_error=102; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
 }
 //Check and parse the index contraction pattern:
 errc=talsh_get_contr_ptrn_str2dig(cptrn,contr_ptrn,&cpl,&conj_bits);
 if(errc){tsk->task_error=103; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;}
 //Determine the execution device (devid:[dvk,dvn]):
 if(dev_kind == DEV_DEFAULT){ //device kind is not specified explicitly
  if(dev_id == DEV_DEFAULT){ //neither specific device nor device kind are specified: Find one
   devid=talsh_find_optimal_device(dtens,ltens,rtens);
   if(devid < 0 || devid >= DEV_MAX){
    tsk->task_error=104; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
  }else{ //<dev_id> is a flat device id
   devid=dev_id;
  }
  dvn=talshKindDevId(devid,&dvk);
  if(dvn < 0){tsk->task_error=105; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;}
 }else{ //device kind is specified explicitly
  if(valid_device_kind(dev_kind) != YEP){
   tsk->task_error=106; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;
  }
  dvk=dev_kind;
  if(dev_id == DEV_DEFAULT){ //kind-specific device id is not specified: Implicit
   dvn=-1; //kind-specific device id will be chosen by the corresponding runtime
  }else{ //kind-specific device id is specified
   dvn=dev_id;
   if(talshFlatDevId(dvk,dvn) >= DEV_MAX){
    tsk->task_error=107; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;
   }
  }
 }
 //The tensor operation will be executed on device of kind <dvk>.
 errc=TALSH_SUCCESS;
 //Choose the tensor body image for each tensor argument and adjust the coherence control:
 cohd=argument_coherence_get_value(coh_ctrl,3,0);
 dimg=talsh_choose_image_for_device(dtens,cohd,&dcp,dvk,dvn);
 if(dcp != 0){ //an intermediate copy was introduced on Host
  if(cohd == COPY_K){ //adjust coherence control
   cohd=COPY_M; j=argument_coherence_set_value(&coh_ctrl,3,0,cohd);
  }else if(cohd == COPY_T){
   cohd=COPY_D; j=argument_coherence_set_value(&coh_ctrl,3,0,cohd);
  }
 }
 cohl=argument_coherence_get_value(coh_ctrl,3,1);
 limg=talsh_choose_image_for_device(ltens,cohl,&lcp,dvk,dvn);
 if(lcp != 0){ //an intermediate copy was introduced on Host
  if(cohl == COPY_K){ //adjust coherence control
   cohl=COPY_M; j=argument_coherence_set_value(&coh_ctrl,3,1,cohl);
  }else if(cohl == COPY_T){
   cohl=COPY_D; j=argument_coherence_set_value(&coh_ctrl,3,1,cohl);
  }
 }
 cohr=argument_coherence_get_value(coh_ctrl,3,2);
 rimg=talsh_choose_image_for_device(rtens,cohr,&rcp,dvk,dvn);
 if(rcp != 0){ //an intermediate copy was introduced on Host
  if(cohr == COPY_K){ //adjust coherence control
   cohr=COPY_M; j=argument_coherence_set_value(&coh_ctrl,3,2,cohr);
  }else if(cohr == COPY_T){
   cohr=COPY_D; j=argument_coherence_set_value(&coh_ctrl,3,2,cohr);
  }
 }
 if(dimg < 0 || limg < 0 || rimg < 0){
  tsk->task_error=108; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
 }
 //Check data kind of each image (must match):
 if(dtens->data_kind[dimg] != ltens->data_kind[limg] ||
    dtens->data_kind[dimg] != rtens->data_kind[rimg] ||
    ltens->data_kind[limg] != rtens->data_kind[rimg]){
  tsk->task_error=109; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_INVALID_ARGS;
 }
 //Construct the TAL-SH task:
 if(talshTaskStatus(tsk) == TALSH_TASK_EMPTY){
  errc=talshTaskConstruct(tsk,dvk,coh_ctrl,dtens->data_kind[dimg]);
  if(errc){tsk->task_error=110; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return errc;}
  errc=talshTaskSetArg(tsk,dtens,dimg);
  if(errc){tsk->task_error=111; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return errc;}
  errc=talshTaskSetArg(tsk,ltens,limg);
  if(errc){tsk->task_error=112; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return errc;}
  errc=talshTaskSetArg(tsk,rtens,rimg);
  if(errc){tsk->task_error=113; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return errc;}
 }else{
  tsk->task_error=114; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_OBJECT_NOT_EMPTY;
 }
 //Schedule the tensor operation via the device-kind specific runtime:
 switch(dvk){
  case DEV_HOST:
   //Associate TAL-SH tensor images with <tensor_block_t> objects:
   errc=talsh_tensor_f_assoc(dtens,dimg,&dftr);
   if(errc || dftr == NULL){
    tsk->task_error=115; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   errc=talsh_tensor_f_assoc(ltens,limg,&lftr);
   if(errc || lftr == NULL){
    errc=talsh_tensor_f_dissoc(dftr);
    tsk->task_error=116; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   errc=talsh_tensor_f_assoc(rtens,rimg,&rftr);
   if(errc || rftr == NULL){
    errc=talsh_tensor_f_dissoc(lftr); errc=talsh_tensor_f_dissoc(dftr);
    tsk->task_error=116; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   //Get the Host task alias:
   host_task=(host_task_t*)(tsk->task_p);
   //Mark images to be discarded as unavailable:
   devid=talshFlatDevId(DEV_HOST,0); //execution device
   if(cohr == COPY_D || (cohr == COPY_M && rtens->dev_rsc[rimg].dev_id != devid)) rtens->avail[rimg] = NOPE;
   if(cohl == COPY_D || (cohl == COPY_M && ltens->dev_rsc[limg].dev_id != devid)) ltens->avail[limg] = NOPE;
   if(cohd == COPY_D || (cohd == COPY_M && dtens->dev_rsc[dimg].dev_id != devid)) dtens->avail[dimg] = NOPE;
   //Schedule the tensor operation via the device-kind specific runtime:
   ctm=clock();
   errc=cpu_tensor_block_contract(contr_ptrn,lftr,rftr,dftr,scale_real,scale_imag,conj_bits); //blocking call
   if(talshTensorRank(dtens) == 0){ //an explicit update is needed for scalar destinations
    j=talsh_update_f_scalar(dftr,dtens->data_kind[dimg],dtens->dev_rsc[dimg].gmem_p);
    if(j) errc=TALSH_FAILURE;
   }
   tsk->exec_time=((double)(clock()-ctm))/CLOCKS_PER_SEC;
   //Dissociate <tensor_block_t> objects:
   j=talsh_tensor_f_dissoc(rftr); if(j) errc=TALSH_FAILURE;
   j=talsh_tensor_f_dissoc(lftr); if(j) errc=TALSH_FAILURE;
   j=talsh_tensor_f_dissoc(dftr); if(j) errc=TALSH_FAILURE;
   if(errc){
    if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=TALSH_FAILURE;
    j=host_task_record(host_task,coh_ctrl,13);
    j=host_task_destroy(host_task); if(j) errc=TALSH_FAILURE;
    tsk->task_p=NULL;
    tsk->task_error=117; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
    return errc;
   }else{ //coherence control
    errc=host_task_record(host_task,coh_ctrl,0); //record task success (no coherence control on Host)
    if(errc){tsk->task_error=118; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
    if(rtens->avail[rimg] == NOPE){
     errc=talsh_tensor_image_discard(rtens,rimg);
     if(errc){tsk->task_error=119; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
    }
    if(ltens->avail[limg] == NOPE){
     errc=talsh_tensor_image_discard(ltens,limg);
     if(errc){tsk->task_error=120; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
    }
    errc=talsh_tensor_image_discard_other(dtens,dimg); //discard other images after the update operation
    if(errc){tsk->task_error=121; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
    if(dtens->avail[0] == NOPE){ //only one image left
     errc=talsh_tensor_image_discard(dtens,0);
     if(errc){tsk->task_error=122; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;}
    }
   }
   //If blocking call, complete it here:
   if(talsh_task == NULL){
    errc=talshTaskWait(tsk,&j); if(errc == TALSH_SUCCESS && j != TALSH_TASK_COMPLETED) errc=TALSH_TASK_ERROR;
    j=talshTaskDestroy(tsk); if(j != TALSH_SUCCESS && errc == TALSH_SUCCESS) errc=j;
   }
   break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
   //Associate TAL-SH tensor images with <tensBlck_t> objects:
   errc=talsh_tensor_c_assoc(dtens,dimg,&dctr);
   if(errc || dctr == NULL){
    tsk->task_error=123; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   errc=talsh_tensor_c_assoc(ltens,limg,&lctr);
   if(errc || lctr == NULL){
    errc=talsh_tensor_c_dissoc(dctr);
    tsk->task_error=124; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   errc=talsh_tensor_c_assoc(rtens,rimg,&rctr);
   if(errc || rctr == NULL){
    errc=talsh_tensor_c_dissoc(lctr); errc=talsh_tensor_c_dissoc(dctr);
    tsk->task_error=125; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   //Get the CUDA task alias:
   cuda_task=(cudaTask_t*)(tsk->task_p);
   //Schedule the operation via the device-kind specific runtime:
   if(conj_bits != 0){ //`Add complex conjugation feature to NV-TAL
    printf("#FATAL(talshc:talshTensorContract): Complex conjugation feature is not implemented for GPU target!");
    tsk->task_error=200; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
   }
   errc=gpu_tensor_block_contract_dlf(contr_ptrn,lctr,rctr,dctr,coh_ctrl,cuda_task,dvn,scale_real,scale_imag); //non-blocking call
   dvn=cuda_task_gpu_id(cuda_task);
   if(errc || dvn < 0){ //in case of error, CUDA task has already been finalized (with error) without coherence control
    if(errc != TRY_LATER && errc != DEVICE_UNABLE) errc=TALSH_FAILURE;
    j=talsh_tensor_c_dissoc(rctr); if(j) errc=TALSH_FAILURE;
    j=talsh_tensor_c_dissoc(lctr); if(j) errc=TALSH_FAILURE;
    j=talsh_tensor_c_dissoc(dctr); if(j) errc=TALSH_FAILURE;
    j=cuda_task_destroy(cuda_task); if(j) errc=TALSH_FAILURE;
    tsk->task_p=NULL;
    tsk->task_error=126; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   }else{ //coherence control (mark images to be discarded as unavailable)
    devid=talshFlatDevId(dvk,dvn);
    coh=argument_coherence_get_value(coh_ctrl,3,2);
    if(coh == COPY_D || (coh == COPY_M && rtens->dev_rsc[rimg].dev_id != devid)) rtens->avail[rimg] = NOPE;
    coh=argument_coherence_get_value(coh_ctrl,3,1);
    if(coh == COPY_D || (coh == COPY_M && ltens->dev_rsc[limg].dev_id != devid)) ltens->avail[limg] = NOPE;
    coh=argument_coherence_get_value(coh_ctrl,3,0);
    if(coh == COPY_D || (coh == COPY_M && dtens->dev_rsc[dimg].dev_id != devid)) dtens->avail[dimg] = NOPE;
   }
   //If blocking call, complete it here:
   if(talsh_task == NULL){
    errc=talshTaskWait(tsk,&j); if(errc == TALSH_SUCCESS && j != TALSH_TASK_COMPLETED) errc=TALSH_TASK_ERROR;
    j=talsh_tensor_c_dissoc(rctr); if(j) errc=TALSH_FAILURE;
    j=talsh_tensor_c_dissoc(lctr); if(j) errc=TALSH_FAILURE;
    j=talsh_tensor_c_dissoc(dctr); if(j) errc=TALSH_FAILURE;
    j=talshTaskDestroy(tsk); if(j != TALSH_SUCCESS && errc == TALSH_SUCCESS) errc=j;
   }
#else
   tsk->task_error=127; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_AVAILABLE;
#endif
   break;
  case DEV_INTEL_MIC:
#ifndef NO_PHI
   tsk->task_error=128; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   tsk->task_error=129; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  case DEV_AMD_GPU:
#ifndef NO_AMD
   tsk->task_error=130; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_IMPLEMENTED; //`Future
#else
   tsk->task_error=131; if(talsh_task == NULL) j=talshTaskDestroy(tsk);
   return TALSH_NOT_AVAILABLE;
#endif
   //break;
  default:
  tsk->task_error=132; if(talsh_task == NULL) j=talshTaskDestroy(tsk); return TALSH_FAILURE;
 }
 return errc;
}

int talshTensorContract_(const char * cptrn, talsh_tens_t * dtens, talsh_tens_t * ltens, talsh_tens_t * rtens,
                         double scale_real, double scale_imag, int dev_id, int dev_kind, int copy_ctrl, talsh_task_t * talsh_task) //Fortran wrapper
{
 return talshTensorContract(cptrn,dtens,ltens,rtens,scale_real,scale_imag,dev_id,dev_kind,copy_ctrl,talsh_task);
}

double talshTensorImageNorm1_cpu(const talsh_tens_t * talsh_tens)
/** Computes the 1-norm of the tensor body image residing on Host. **/
{
 int i,nimg;
 size_t j,n;
 int dtk[TALSH_MAX_DEV_PRESENT];
 double norm1;
 float *r4p;
 double *r8p;
 talshComplex4 *c4p;
 talshComplex8 *c8p;

 norm1=-1.0;
 if(talsh_tens != NULL){
  i=talshTensorDataKind(talsh_tens,&nimg,dtk);
  if(i == TALSH_SUCCESS){
   for(i=0;i<talsh_tens->ndev;++i){
    if(talsh_tens->dev_rsc[i].dev_id == talshFlatDevId(DEV_HOST,0)){
     n=talshTensorVolume(talsh_tens); norm1=0.0;
     switch(dtk[i]){
      case R4:
       r4p=(float*)(talsh_tens->dev_rsc[i].gmem_p);
       for(j=0;j<n;++j){norm1+=(double)(ABS(r4p[j]));}
       break;
      case R8:
       r8p=(double*)(talsh_tens->dev_rsc[i].gmem_p);
       for(j=0;j<n;++j){norm1+=ABS(r8p[j]);}
       break;
      case C4:
       c4p=(talshComplex4*)(talsh_tens->dev_rsc[i].gmem_p);
       for(j=0;j<n;++j){norm1+=(double)(talshComplex4Abs(c4p[j]));}
       break;
      case C8:
       c8p=(talshComplex8*)(talsh_tens->dev_rsc[i].gmem_p);
       for(j=0;j<n;++j){norm1+=talshComplex8Abs(c8p[j]);}
       break;
     }
     break;
    }
   }
  }
 }
 return norm1;
}
