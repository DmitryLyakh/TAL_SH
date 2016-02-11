/** ExaTensor::TAL-SH: Device-unified user-level API.
REVISION: 2016/02/11
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
// TAL-SH control:
int talshInit(size_t * host_buf_size, int * host_arg_max, int ngpus, int gpu_list[],
                                                          int nmics, int mic_list[],
                                                          int namds, int amd_list[])
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
 errc=arg_buf_deallocate(talsh_gpu_beg,talsh_gpu_end); if(errc) return TALSH_FAILURE;
 talsh_gpu_beg=0; talsh_gpu_end=-1; talsh_on=0;
 return TALSH_SUCCESS;
}
