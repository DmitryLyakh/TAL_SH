/** ExaTensor::TAL-SH: Device-unified user-level API.
REVISION: 2016/02/11

Copyright (C) 2014-2016 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2016 Oak Ridge National Laboratory (UT-Battelle)

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
-------------------------------------------------------------------------------
**/

#include <stdio.h>
#include <stdlib.h>

#include "talsh.h"

static int talsh_on=0;    //TAL-SH initialization flag (1:initalized; 0:not)
static int talsh_gpu_beg;
static int talsh_gpu_end;

static int talsh_gpus[MAX_GPUS_PER_NODE]; //current GPU status
static int talsh_mics[MAX_MICS_PER_NODE]; //current MIC status
static int talsh_amds[MAX_AMDS_PER_NODE]; //current AMD status

static talsh_task_t talsh_tasks[TALSH_MAX_ACTIVE_TASKS]; //reusable TAL-SH tasks

//Exported functions:
// TAL-SH device control:
int talshInit(size_t * host_buf_size,    //inout: Host Argument Buffer size in bytes (in: suggested; out: actual)
              int * host_arg_max,        //out: Max number of arguments that can fit into the Host Argument Buffer
              int ngpus, int gpu_list[], //in: number of Nvidia GPU(s) to use and the list of Nvidia GPU(s) to use
              int nmics, int mic_list[], //in: number of Intel Xeon Phi(s) to use and the list of Intel Xeon Phi(s) to use
              int namds, int amd_list[]) //in: number of AMD GPU(s) to use and the list of AMD GPU(s) to use
/** Initializes the TAL-SH runtime. **/
{
 int i,gpu_beg,gpu_end,errc;

 if(talsh_on) return TALSH_ALREADY_INITIALIZED;
//NVidia GPU accelerators:
 if(ngpus > 0){
  if(ngpus > MAX_GPUS_PER_NODE) return TALSH_INVALID_ARGS;
  gpu_beg=gpu_list[0]; gpu_end=gpu_list[ngpus-1];
  if(gpu_beg < 0 || gpu_beg >= MAX_GPUS_PER_NODE) return TALSH_INVALID_ARGS;
  if(gpu_end < 0 || gpu_end >= MAX_GPUS_PER_NODE) return TALSH_INVALID_ARGS;
  for(i=1;i<ngpus;i++){
   if(gpu_list[i] != gpu_list[i-1]+1){
    printf("#FATAL(TALSH::talshInit): The current version only supports consecutive GPU ranges!");
    return TALSH_FAILURE;
   }
  }
  errc=arg_buf_allocate(host_buf_size,host_arg_max,gpu_beg,gpu_end);
  if(errc) return TALSH_FAILURE;
  talsh_gpu_beg=gpu_beg; talsh_gpu_end=gpu_end;
 }else{
  talsh_gpu_beg=0; talsh_gpu_end=-1;
 }
//Intel Xeon Phi accelerators:
 if(nmics > 0){
  printf("#FATAL(TALSH::talshInit): Intel Xeon Phi is not supported yet!");
  return TALSH_FAILURE;
 }
//AMD GPU accelerators:
 if(namds > 0){
  printf("#FATAL(TALSH::talshInit): AMD GPU is not supported yet!");
  return TALSH_FAILURE;
 }
 talsh_on=1;
 return TALSH_SUCCESS;
}

int talshShutdown()
/** Shuts down the TAL-SH runtime. **/
{
 int errc;

 if(talsh_on == 0) return TALSH_NOT_INITIALIZED;
 errc=arg_buf_deallocate(talsh_gpu_beg,talsh_gpu_end);
 talsh_gpu_beg=0; talsh_gpu_end=-1; talsh_on=0;
 if(errc) return TALSH_FAILURE;
 return TALSH_SUCCESS;
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
