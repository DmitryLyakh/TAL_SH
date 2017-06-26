/** TALSH::C/C++ API testing.

!Copyright (C) 2014-2016 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2014-2016 Oak Ridge National Laboratory (UT-Battelle)

!This file is part of ExaTensor.

!ExaTensor is free software: you can redistribute it and/or modify
!it under the terms of the GNU Lesser General Public License as published
!by the Free Software Foundation, either version 3 of the License, or
!(at your option) any later version.

!ExaTensor is distributed in the hope that it will be useful,
!but WITHOUT ANY WARRANTY; without even the implied warranty of
!MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
!GNU Lesser General Public License for more details.

!You should have received a copy of the GNU Lesser General Public License
!along with ExaTensor. If not, see <http://www.gnu.org/licenses/>.
**/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "talsh.h"

#ifdef __cplusplus
extern "C"{
#endif
#ifndef NO_GPU
void test_nvtal_c(int * ierr);
#endif
void test_talsh_c(int * ierr);
#ifdef __cplusplus
}
#endif

#ifndef NO_GPU
void test_nvtal_c(int * ierr)
{
 int host_arg_max,errc;
 size_t host_buf_size;
 cudaTask_t *tsk0,*tsk1; //CUDA tasks (pointers)
 tensBlck_t *t0,*t1,*t2; //tensor blocks (pointers)
 int r0=4,r1=4,r2=4; //tensor block ranks
 int dims0[]={40,40,40,40}; //tensor block 0 dimensions
 int dims1[]={40,40,40,40}; //tensor block 1 dimensions
 int dims2[]={40,40,40,40}; //tensor block 2 dimensions
 float tm_tot,tm_in,tm_out,tm_comp;

 *ierr=0;
//Initialize Host/GPU argument buffers and NV-TAL:
 host_buf_size=1000000000;
 printf(" Initializing NV-TAL ...");
 errc=arg_buf_allocate(&host_buf_size,&host_arg_max,0,0);
 printf(" Status %d: Host argument buffer size = %lu; Max args in HAB = %d\n",errc,host_buf_size,host_arg_max);
 if(errc){*ierr=1; return;}

//Create tensor blocks:
 //Tensor block 0:
 printf(" Creating tensor block 0 ...");
 errc=tensBlck_create(&t0);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}
 printf(" Constructing shape of tensor block 0 ...");
 errc=tensBlck_construct(t0,YEP,r0,dims0);
 printf(" Status %d: Tensor block volume = %lu:",errc,tensBlck_volume(t0)); if(errc){*ierr=1; return;}
 printf(" Attaching body to tensor block 0 ...");
 errc=tensBlck_attach_body(t0,R8);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}
 //Tensor block 1:
 printf(" Creating tensor block 1 ...");
 errc=tensBlck_create(&t1);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}
 printf(" Constructing shape of tensor block 1 ...");
 errc=tensBlck_construct(t1,YEP,r1,dims1);
 printf(" Status %d: Tensor block volume = %lu:",errc,tensBlck_volume(t1)); if(errc){*ierr=1; return;}
 printf(" Attaching body to tensor block 1 ...");
 errc=tensBlck_attach_body(t1,R8);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}
 //Tensor block 2:
 printf(" Creating tensor block 2 ...");
 errc=tensBlck_create(&t2);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}
 printf(" Constructing shape of tensor block 2 ...");
 errc=tensBlck_construct(t2,YEP,r2,dims2);
 printf(" Status %d: Tensor block volume = %lu:",errc,tensBlck_volume(t2)); if(errc){*ierr=1; return;}
 printf(" Attaching body to tensor block 2 ...");
 errc=tensBlck_attach_body(t2,R8);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}

//Initialize tensor blocks to value (on Host):
 //Tensor block 0:
 printf(" Initializing tensor block 0 ...");
 errc=tensBlck_init_host(t0,0.0);
 printf(" Status %d:",errc); if(errc){*ierr=1; return;}
 printf(" Squared 2-norm = %e\n",tensBlck_norm2_host(t0));
 //Tensor block 1:
 printf(" Initializing tensor block 1 ...");
 errc=tensBlck_init_host(t1,0.01);
 printf(" Status %d:",errc); if(errc){*ierr=1; return;}
 printf(" Squared 2-norm = %e\n",tensBlck_norm2_host(t1));
 //Tensor block 2:
 printf(" Initializing tensor block 2 ...");
 errc=tensBlck_init_host(t2,0.001);
 printf(" Status %d:",errc); if(errc){*ierr=1; return;}
 printf(" Squared 2-norm = %e\n",tensBlck_norm2_host(t2));

//Create CUDA tasks:
 printf(" Creating a CUDA task ...");
 errc=cuda_task_create(&tsk0);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}
 printf(" Creating a CUDA task ...");
 errc=cuda_task_create(&tsk1);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}

//Tensor contraction on GPU:
 int cptrn0[]={4,3,-3,-4,2,1,-3,-4}; //tensor contraction pattern
 //Schedule a tensor contraction task on GPU:
 printf(" Scheduling a tensor contraction on GPU ...");
 errc=gpu_tensor_block_contract_dlf(cptrn0,t1,t2,t0,COPY_TTT,tsk0);
 cuda_task_print(tsk0);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}
 //Wait until task completion:
 printf(" Waiting upon completion ...");
 errc=cuda_task_wait(tsk0);
 printf(" Status %d",errc); if(errc != CUDA_TASK_COMPLETED){*ierr=1; return;}
 tm_tot=cuda_task_time(tsk0,&tm_in,&tm_out,&tm_comp); //task timing
 printf(": Timings (total,in,out,comp): %f %f %f %f\n",tm_tot,tm_in,tm_out,tm_comp);
 //Print the 2-norm of the destination tensor:
 printf(" Destination tensor squared 2-norm = %e\n",tensBlck_norm2_host(t0));

//Destroy CUDA tasks:
 printf(" Destroying a CUDA task ...");
 errc=cuda_task_destroy(tsk1);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}
 printf(" Destroying a CUDA task ...");
 errc=cuda_task_destroy(tsk0);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}

//Destroy tensor blocks:
 //Tensor block 2:
 printf(" Destroying tensor block 2 ...");
 errc=tensBlck_destroy(t2);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}
 //Tensor block 1:
 printf(" Destroying tensor block 1 ...");
 errc=tensBlck_destroy(t1);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}
 //Tensor block 0:
 printf(" Destroying tensor block 0 ...");
 errc=tensBlck_destroy(t0);
 printf(" Status %d\n",errc); if(errc){*ierr=1; return;}

//NV-TAL statistics:
 printf(" NV-TAL statistics:");
 gpu_print_stats();

//Free Host/GPU argument buffers and shutdown NV-TAL:
 printf(" Shutting down NV-TAL ...");
 errc=arg_buf_deallocate(0,0);
 printf(" Status: %d\n",errc); if(errc){*ierr=1; return;}
 return;
}
#endif //NO_GPU

void test_talsh_c(int * ierr)
{
 int errc;
 size_t small_buffer_size=TALSH_NO_HOST_BUFFER;
 int gpu_list[MAX_GPUS_PER_NODE];

 *ierr=0;

//Query the total number of NVIDIA GPU on node:
 int ngpu;
 errc=talshGetDeviceCount(DEV_NVIDIA_GPU,&ngpu); if(errc){*ierr=1; return;};
 printf(" Number of NVIDIA GPU found on node = %d\n",ngpu);

//Initialize TAL-SH (with a negligible Host buffer since we will use external memory):
 int host_arg_max;
 for(int i=0; i<ngpu; ++i) gpu_list[i]=i; //list of NVIDIA GPU devices to use in this process
 errc=talshInit(&small_buffer_size,&host_arg_max,ngpu,gpu_list,0,NULL,0,NULL);
 printf(" TAL-SH has been initialized: Status %d\n",errc); if(errc){*ierr=2; return;};

//Allocate three tensor blocks in Host memory outside of TAL-SH (external application):
 //Tensor block 0:
 int trank0 = 4; //tensor block rank
 const int dims0[] = {40,40,20,20}; //tensor block dimension extents
 size_t vol0 = 1; for(int i=0; i<trank0; ++i) vol0*=(size_t)dims0[i]; //tensor block volume (number of elements)
 double * tblock0 = (double*)malloc(vol0*sizeof(double)); //tensor block body (tensor elements(
 for(size_t l=0; l<vol0; ++l) tblock0[l]=0.0; //initialize it to zero
 //Tensor block 1:
 int trank1 = 4; //tensor block rank
 const int dims1[] = {40,40,40,40}; //tensor block dimension extents
 size_t vol1 = 1; for(int i=0; i<trank1; ++i) vol1*=(size_t)dims1[i]; //tensor block volume (number of elements)
 double * tblock1 = (double*)malloc(vol1*sizeof(double)); //tensor block body (tensor elements(
 for(size_t l=0; l<vol1; ++l) tblock1[l]=0.01; //initialize it to something
 //Tensor block 2:
 int trank2 = 4; //tensor block rank
 const int dims2[] = {40,20,40,20}; //tensor block dimension extents
 size_t vol2 = 1; for(int i=0; i<trank2; ++i) vol2*=(size_t)dims2[i]; //tensor block volume (number of elements)
 double * tblock2 = (double*)malloc(vol2*sizeof(double)); //tensor block body (tensor elements(
 for(size_t l=0; l<vol2; ++l) tblock2[l]=0.001; //initialize it to something
 printf(" Three external tensor blocks have been allocated by application\n");
 double gflops=(sqrt(((double)(vol0))*((double)(vol1))*((double)(vol2)))*2.0)/1e9; //total number of floating point operations

//Register external tensor blocks with TAL-SH (in Host memory):
 //Tensor block 0:
 talsh_tens_t tens0; //declare a TAL-SH tensor block
 errc=talshTensorClean(&tens0); if(errc){*ierr=3; return;}; //clean TAL-SH tensor block object (default ctor)
 errc=talshTensorConstruct(&tens0,R8,trank0,dims0,talshFlatDevId(DEV_HOST,0),(void*)tblock0); //register tensor block
 if(errc){*ierr=4; return;};
 //Tensor block 1:
 talsh_tens_t tens1; //declare a TAL-SH tensor block
 errc=talshTensorClean(&tens1); if(errc){*ierr=5; return;}; //clean TAL-SH tensor block object (default ctor)
 errc=talshTensorConstruct(&tens1,R8,trank1,dims1,talshFlatDevId(DEV_HOST,0),(void*)tblock1); //register tensor block
 if(errc){*ierr=6; return;};
 //Tensor block 2:
 talsh_tens_t tens2; //declare a TAL-SH tensor block
 errc=talshTensorClean(&tens2); if(errc){*ierr=7; return;}; //clean TAL-SH tensor block object (default ctor)
 errc=talshTensorConstruct(&tens2,R8,trank2,dims2,talshFlatDevId(DEV_HOST,0),(void*)tblock2); //register tensor block
 if(errc){*ierr=8; return;};
 printf(" Three external tensor blocks have been registered with TAL-SH\n");

//Declare a TAL-SH task handle:
 talsh_task_t task0; //declare a TAL-SH task handle
 errc=talshTaskClean(&task0); //clean TAL-SH task handle object to an empty state

//Execute a tensor contraction either on CPU (synchronously) or GPU (asynchronously):
#ifndef NO_GPU
 int dev_kind = DEV_NVIDIA_GPU; //NVIDIA GPU devices
 int dev_num = 0; //specific device number (any from gpu_list[])
#else
 int dev_kind = DEV_HOST; //CPU Host (multicore)
 int dev_num = 0; //CPU Host is always a single device (but multicore)
#endif
 //Schedule:
 errc=talshTensorContract("D(a,b,i,j)+=L(a,b,c,d)*R(c,i,d,j)",&tens0,&tens1,&tens2,0.5,0.0,dev_num,dev_kind,COPY_MTT,&task0);
 printf(" Tensor contraction has been scheduled for execution: Status %d\n",errc); if(errc){*ierr=9; return;};
 //Test for completion:
 int sts,done=NOPE;
 while(done != YEP && errc == TALSH_SUCCESS){done=talshTaskComplete(&task0,&sts,&errc);}
 if(errc == TALSH_SUCCESS){
  printf(" Tensor contraction has completed successfully: Status %d\n",sts);
 }else{
  printf(" Tensor contraction has failed: Status %d: Error %d\n",sts,errc);
  *ierr=10; return;
 }
 //Timing:
 double total_time;
 errc=talshTaskTime(&task0,&total_time); if(errc){*ierr=11; return;};
 printf(" Tensor contraction total time = %f: GFlop/s = %f\n",total_time,gflops/total_time);
 //Destruct the task handle:
 errc=talshTaskDestruct(&task0); if(errc){*ierr=12; return;};
#ifndef NO_GPU
 //If executed on GPU, COPY_MTT parameter in the tensor contraction call above means that the
 //destination tensor image was moved to GPU device (letter M means MOVE).
 //So, let's move it back to Host (to a user-specified memory location):
 errc=talshTensorPlace(&tens0,0,DEV_HOST,(void*)tblock0,COPY_M); //this will move the resulting tensor block back to Host (letter M means MOVE)
 if(errc){*ierr=13; return;};
 printf(" Tensor result was moved back to Host: Norm1 = %E\n",talshTensorImageNorm1_cpu(&tens0));
#endif

//Unregister tensor blocks with TAL-SH:
 errc=talshTensorDestruct(&tens2); if(errc){*ierr=14; return;};
 errc=talshTensorDestruct(&tens1); if(errc){*ierr=15; return;};
 errc=talshTensorDestruct(&tens0); if(errc){*ierr=16; return;};
 printf(" Three external tensor blocks have been unregistered with TAL-SH\n");

//Free external memory (local tensor blocks):
 free(tblock2); tblock2=NULL;
 free(tblock1); tblock1=NULL;
 free(tblock0); tblock0=NULL;
//Shutdown TAL-SH:
 errc=talshShutdown();
 printf(" TAL-SH has been shut down: Status %d\n",errc); if(errc){*ierr=17; return;};

 return;
}
