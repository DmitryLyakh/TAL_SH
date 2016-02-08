/** ExaTensor::TAL-SH: Device-unified user-level API. **/

#include <stdio.h>
#include <stdlib.h>

#include "talsh.h"

static int talsh_on=0; //TAL-SH initialization flag (1:initalized; 0:not)

static int talsh_gpus[MAX_GPUS_PER_NODE]; //current GPU status
static int talsh_mics[MAX_MICS_PER_NODE]; //current MIC status
static int talsh_amds[MAX_AMDS_PER_NODE]; //current AMD status

static talsh_task_t talsh_tasks[TALSH_MAX_ACTIVE_TASKS]; //reusable TAL-SH tasks

//Exported functions:
// TAL-SH control:
int talshInit(size_t * host_buf_size, int * host_arg_max, int ngpus, int gpu_list[],
                                                          int nmics, int mic_list[],
                                                          int namds, int amd_list[])
{
 
 talsh_on=1;
 return TALSH_SUCCESS;
}

int talshShutdown()
{
 
 talsh_on=0;
 return TALSH_SUCCESS;
}
