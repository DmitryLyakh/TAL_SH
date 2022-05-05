/** Tensor Algebra Library core for GPU
AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com, liakhdi@ornl.gov
REVISION: 2020/09/23

Copyright (C) 2014-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2022 Oak Ridge National Laboratory (UT-Battelle)

LICENSE: BSD 3-Clause

-------------------------------------------------------------------
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

#include "tensor_algebra.h"
#include "mem_manager.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

static int VERBOSE=1; //verbosity for error messages
static int DEBUG=0; //debugging mode

size_t fortran_cptr_int(void * cptr)
{
 return ((size_t)(cptr));
}

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

int permutation_trivial(const int perm_len, const int * perm, const int base)
{
 int trivial = 1;
 for(int i = 0; i < perm_len; ++i){
  if(perm[i] != (base + i)){trivial = 0; break;}
 }
 return trivial;
}

#ifdef USE_CUTENSOR
int get_contr_pattern_cutensor(const int * dig_ptrn,
                               int drank, int32_t * ptrn_d,
                               int lrank, int32_t * ptrn_l,
                               int rrank, int32_t * ptrn_r)
/** Converts a digital tensor contraction pattern used by TAL-SH into the cuTensor digital format. **/
{
 int errc = 0;
 if(drank >= 0 && lrank >= 0 && rrank >= 0){
  if(lrank + rrank > 0){
   if(dig_ptrn != NULL){
    int ci = drank; //contracted indices will have ids: drank+1,drank+2,drank+3,...
    for(int i = 0; i < drank; ++i) ptrn_d[i] = (i+1); //dtens[1,2,3,4,...]
    for(int i = 0; i < lrank; ++i){
     int j = dig_ptrn[i];
     if(j > 0){ //uncontracted index
      ptrn_l[i] = j;
     }else if(j < 0){ //contracted index
      ptrn_l[i] = ++ci;
      ptrn_r[-j-1] = ci;
     }else{
      errc = -5;
      break;
     }
    }
    if(errc == 0){
     for(int i = 0; i < rrank; ++i){
      int j = dig_ptrn[lrank+i];
      if(j > 0){ //uncontracted index
       ptrn_r[i] = j;
      }else if(j < 0){ //contracted index
       if(ptrn_r[i] != ptrn_l[-j-1]){ //already set
        errc = -4;
        break;
       }
      }else{
       errc = -3;
       break;
      }
     }
    }
   }else{
    errc = -2;
   }
  }
 }else{
  errc = -1;
 }
 return errc;
}
#endif /*USE_CUTENSOR*/

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
#endif /*NO_GPU*/
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
#endif /*NO_GPU*/
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

//TENSOR BLOCK API:
int tensSignature_create(talsh_tens_signature_t ** tsigna)
{
 if(tsigna == NULL) return -1;
 *tsigna = (talsh_tens_signature_t*)malloc(sizeof(talsh_tens_signature_t));
 if(*tsigna == NULL) return TRY_LATER;
 return tensSignature_clean(*tsigna);
}

int tensSignature_clean(talsh_tens_signature_t * tsigna)
{
 if(tsigna != NULL){
  tsigna->num_dim = -1;   //tensor rank
  tsigna->offsets = NULL; //array of offsets
 }else{
  return -1;
 }
 return 0;
}

int tensSignature_construct(talsh_tens_signature_t * tsigna, int rank, const size_t * offsets)
{
 int errc = 0;
 if(tsigna != NULL){
  if(tsigna->num_dim >= 0) errc = tensSignature_destruct(tsigna);
  if(errc == 0){
   if(rank > 0){
    if(offsets != NULL){
     tsigna->offsets = (size_t*)malloc(sizeof(size_t)*rank);
     if(tsigna->offsets == NULL) return TRY_LATER;
     for(int i = 0; i < rank; ++i) tsigna->offsets[i] = offsets[i];
     tsigna->num_dim = rank;
    }else{
     errc = -3;
    }
   }else if(rank == 0){
    tsigna->num_dim = rank;
   }else{
    errc = -2;
   }
  }
 }else{
  errc = -1;
 }
 return errc;
}

int tensSignature_destruct(talsh_tens_signature_t * tsigna)
{
 if(tsigna == NULL) return -1;
 if(tsigna->offsets != NULL) free(tsigna->offsets);
 return tensSignature_clean(tsigna);
}

int tensSignature_destroy(talsh_tens_signature_t * tsigna)
{
 if(tsigna == NULL) return -1;
 int errc = tensSignature_destruct(tsigna);
 free(tsigna);
 return errc;
}

int tensShape_create(talsh_tens_shape_t ** tshape)
/** Creates a tensor shape and cleans it. **/
{
 if(tshape == NULL) return -1;
 *tshape=(talsh_tens_shape_t*)malloc(sizeof(talsh_tens_shape_t));
 if(*tshape == NULL) return TRY_LATER;
 return tensShape_clean(*tshape);
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
   mi_dims=(int*)malloc(3*rank*sizeof(int));
   if(mi_dims == NULL) return TRY_LATER;
   mi_divs=mi_dims+rank;
   mi_grps=mi_divs+rank;
  }else{
   if(rank > MAX_TENSOR_RANK) return -8;
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
 if(n != 0){
  if(VERBOSE) printf("#ERROR(tensShape_destruct): Resource release error %d\n",n);
  n=NOT_CLEAN;
 }
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

int tensShape_reshape(talsh_tens_shape_t * tshape, int rank, const int * dims, const int * divs, const int * grps)
{
 int tens_rank,pinned,errc;
 size_t vol;

 errc=0;
 if(tshape == NULL) return -1;
 tens_rank=tensShape_rank(tshape);
 if(tens_rank > 0){
  if(tshape->dims != NULL){
   vol=tensShape_volume(tshape);
   pinned=mi_entry_pinned(tshape->dims);
   errc=tensShape_destruct(tshape);
   if(errc == 0){
    errc=tensShape_construct(tshape,pinned,rank,dims,divs,grps);
    if(errc == 0 && tensShape_volume(tshape) != vol) errc=-2;
   }
  }else{
   errc=-3;
  }
 }else{
  if(dims != NULL || divs != NULL || grps != NULL) errc=-4;
 }
 return errc;
}

void tensShape_print(const talsh_tens_shape_t * tshape)
/** Prints the tensor shape (dimension extents). **/
{
 int i;

 printf("[");
 for(i=0;i<(tshape->num_dim);++i){
  if(i == (tshape->num_dim)-1){
   printf("%d",tshape->dims[i]);
  }else{
   printf("%d,",tshape->dims[i]);
  }
 }
 printf("]");
 return;
}
