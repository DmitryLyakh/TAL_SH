/** TALSH::C/C++ API testing. **/

#include <stdio.h>
#include <stdlib.h>

#include "talsh.h"

#ifdef __cplusplus
extern "C"{
#endif
void test_talsh_c(int * ierr);
#ifdef __cplusplus
}
#endif

void test_talsh_c(int * ierr)
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
 printf(" Status %d: Host argument buffer size = %lu; Host arg max = %d\n",errc,host_buf_size,host_arg_max);
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
 errc=gpu_tensor_block_contract_dlf_(cptrn0,t1,t2,t0,COPY_TTT,tsk0);
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
