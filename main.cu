/** Examples how to use GPU tensor algebra library (NV-TAL).
AUTHOR: Dmitry I. Lyakh
LICENSE: GPL v2 **/

#include <stdio.h>
#include <time.h>
#include "tensor_algebra.h"

#define U_DIM_EXT 81 //uncontracted tensor dimension extent
#define C_DIM_EXT 81 //contracted tensor dimension extent

const int first_gpu=0, last_gpu=0; //range of GPUs used
size_t buf_size=1000000; //desired argument buffer size in bytes (Host pinned RAM)
const double init_val=0.01;

int main(int argc, char** args){
 tensBlck_t *tb0,*tb1,*tb2,*tb3,*tb4,*tb5; //tensor blocks
 cudaTask_t *tsk0,*tsk1,*tsk2,*tsk3,*tsks[4]; //cuda task handles
 float tm,incopy,outcopy,comput;
 int errc,max_args,tsk_stats[4];
 double flops,tmm,u_ext,c_ext;
 clock_t tms,tme;

//Allocate argument buffers (initializes both the Host and all GPUs):
 errc=arg_buf_allocate(&buf_size,&max_args,first_gpu,last_gpu);
 if(errc){printf("#ERROR: arg_buf_allocate failed!"); return 1;};
 printf("Allocated argument buffers: Host buffer size = %zu bytes: max args = %d \n",buf_size,max_args);

//Activate a specific GPU:
 errc=gpu_activate(first_gpu); if(errc){printf("#ERROR: gpu_activate failed!"); return 1;};
 errc=gpu_set_shmem_width(R8); //set Shared Memory width to 8-byte (double)
 printf("Focused on GPU #%d \n",first_gpu);

//Create and initialize few tensor blocks on Host:
 //Tensor block 0:
 errc=tensBlck_create(&tb0); //create an empty tensor block
 if(errc){printf("#ERROR: tensBlck_create [0] failed!"); return 1;};
 int trank0=4; int dims0[4]={U_DIM_EXT,U_DIM_EXT,U_DIM_EXT,U_DIM_EXT}; //tensor block shape
 errc=tensBlck_alloc(tb0,first_gpu,R8,trank0,dims0); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: tensBlck_alloc [0] failed!"); return 1;};
 printf("Allocated tensor block %d \n",0);
 //Tensor block 1:
 errc=tensBlck_create(&tb1); //create an empty tensor block
 if(errc){printf("#ERROR: tensBlck_create [1] failed!"); return 1;};
 int trank1=4; int dims1[4]={U_DIM_EXT,C_DIM_EXT,U_DIM_EXT,C_DIM_EXT}; //tensor block shape
 errc=tensBlck_alloc(tb1,first_gpu,R8,trank1,dims1); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: tensBlck_alloc [1] failed!"); return 1;};
 printf("Allocated tensor block %d \n",1);
 //Tensor block 2:
 errc=tensBlck_create(&tb2); //create an empty tensor block
 if(errc){printf("#ERROR: tensBlck_create [2] failed!"); return 1;};
 int trank2=4; int dims2[4]={C_DIM_EXT,U_DIM_EXT,C_DIM_EXT,U_DIM_EXT}; //tensor block shape
 errc=tensBlck_alloc(tb2,first_gpu,R8,trank2,dims2); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: tensBlck_alloc [2] failed!"); return 1;};
 printf("Allocated tensor block %d \n",2);
 //Tensor block 3:
 errc=tensBlck_create(&tb3); //create an empty tensor block
 if(errc){printf("#ERROR: tensBlck_create [3] failed!"); return 1;};
 int trank3=4; int dims3[4]={U_DIM_EXT,U_DIM_EXT,U_DIM_EXT,U_DIM_EXT}; //tensor block shape
 errc=tensBlck_alloc(tb3,first_gpu,R8,trank3,dims3); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: tensBlck_alloc [3] failed!"); return 1;};
 printf("Allocated tensor block %d \n",3);
 //Tensor block 4:
 errc=tensBlck_create(&tb4); //create an empty tensor block
 if(errc){printf("#ERROR: tensBlck_create [4] failed!"); return 1;};
 int trank4=4; int dims4[4]={U_DIM_EXT,C_DIM_EXT,U_DIM_EXT,C_DIM_EXT}; //tensor block shape
 errc=tensBlck_alloc(tb4,first_gpu,R8,trank4,dims4); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: tensBlck_alloc [4] failed!"); return 1;};
 printf("Allocated tensor block %d \n",4);
 //Tensor block 5:
 errc=tensBlck_create(&tb5); //create an empty tensor block
 if(errc){printf("#ERROR: tensBlck_create [5] failed!"); return 1;};
 int trank5=4; int dims5[4]={C_DIM_EXT,U_DIM_EXT,C_DIM_EXT,U_DIM_EXT}; //tensor block shape
 errc=tensBlck_alloc(tb5,first_gpu,R8,trank5,dims5); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: tensBlck_alloc [5] failed!"); return 1;};
 printf("Allocated tensor block %d \n",5);

//Create empty CUDA task handles:
 errc=cuda_task_create(&tsk0); if(errc){printf("#ERROR: cuda_task_create [0] failed!"); return 1;};
 errc=cuda_task_create(&tsk1); if(errc){printf("#ERROR: cuda_task_create [1] failed!"); return 1;};
 errc=cuda_task_create(&tsk2); if(errc){printf("#ERROR: cuda_task_create [2] failed!"); return 1;};
 errc=cuda_task_create(&tsk3); if(errc){printf("#ERROR: cuda_task_create [3] failed!"); return 1;};
//-------------------------------------------------------------------------------------------------
//CUDA tasks (examples):

//Tensor initializations (four concurrent tasks are executed on GPU asynchronously w.r.t. Host):
 errc=gpu_tensor_block_init_(tb1,init_val,COPY_BACK,tsk0); if(errc){printf("#ERROR: gpu_tensor_block_init_ [0] failed!"); return 1;};
 errc=gpu_tensor_block_init_(tb2,init_val,COPY_BACK,tsk1); if(errc){printf("#ERROR: gpu_tensor_block_init_ [1] failed!"); return 1;};
 errc=gpu_tensor_block_init_(tb4,0.0,COPY_BACK,tsk2); if(errc){printf("#ERROR: gpu_tensor_block_init_ [2] failed!"); return 1;};
 errc=gpu_tensor_block_init_(tb5,0.0,COPY_BACK,tsk3); if(errc){printf("#ERROR: gpu_tensor_block_init_ [3] failed!"); return 1;};
// Wait on completion (or do something on Host):
 tsks[0]=tsk0; tsks[1]=tsk1; tsks[2]=tsk2; tsks[3]=tsk3; errc=cuda_tasks_wait(4,tsks,tsk_stats);
 if(errc){printf("#ERROR: cuda_tasks_wait [0,1,2,3] failed!"); return 1;};
// Print timings (total time, H2D time, D2H time, GPU computing time):
 tm=cuda_task_time(tsk0,&incopy,&outcopy,&comput);
 printf("CUDA task 0 final status is %d, time %f %f %f %f \n",tsk_stats[0],tm,incopy,outcopy,comput);
 tm=cuda_task_time(tsk1,&incopy,&outcopy,&comput);
 printf("CUDA task 1 final status is %d, time %f %f %f %f \n",tsk_stats[1],tm,incopy,outcopy,comput);
 tm=cuda_task_time(tsk2,&incopy,&outcopy,&comput);
 printf("CUDA task 2 final status is %d, time %f %f %f %f \n",tsk_stats[2],tm,incopy,outcopy,comput);
 tm=cuda_task_time(tsk3,&incopy,&outcopy,&comput);
 printf("CUDA task 3 final status is %d, time %f %f %f %f \n",tsk_stats[3],tm,incopy,outcopy,comput);
// Mark tensor blocks as present on GPU:
 errc=tensBlck_set_presence(tb1);
 errc=tensBlck_set_presence(tb2);
 errc=tensBlck_set_presence(tb4);
 errc=tensBlck_set_presence(tb5);
// Clean CUDA tasks for reuse:
 errc=cuda_task_clean(tsk0); if(errc){printf("#ERROR: cuda_task_clean [0] failed!"); return 1;};
 errc=cuda_task_clean(tsk1); if(errc){printf("#ERROR: cuda_task_clean [1] failed!"); return 1;};
 errc=cuda_task_clean(tsk2); if(errc){printf("#ERROR: cuda_task_clean [2] failed!"); return 1;};
 errc=cuda_task_clean(tsk3); if(errc){printf("#ERROR: cuda_task_clean [3] failed!"); return 1;};

//Tensor copy (for bandwidth measure only):
 int prmn0[]={1, 1,2,3,4}; //trivial permutation
 int prmn1[]={1, 4,3,2,1}; //reverse permutation (scatter)
 gpu_set_transpose_algorithm(EFF_TRN_OFF);
 printf("GPU RAM copy addresses: %p --> %p\n",(void*)(tb2->elems_d),(void*)(tb5->elems_d));
 errc=gpu_tensor_block_copy_dlf(prmn1,tb1,tb5); if(errc){printf("#ERROR: gpu_tensor_block_copy [1] failed!"); return 1;};
 printf("GPU RAM copy addresses: %p --> %p\n",(void*)(tb1->elems_d),(void*)(tb4->elems_d));
 errc=gpu_tensor_block_copy_dlf(prmn0,tb1,tb4); if(errc){printf("#ERROR: gpu_tensor_block_copy [2] failed!"); return 1;};
 gpu_set_transpose_algorithm(EFF_TRN_ON); //turn on the efficient tensor transpose
 printf("GPU RAM copy addresses: %p --> %p\n",(void*)(tb2->elems_d),(void*)(tb5->elems_d));
 errc=gpu_tensor_block_copy_dlf(prmn1,tb1,tb5); if(errc){printf("#ERROR: gpu_tensor_block_copy [3] failed!"); return 1;};

// Mark tensor blocks as absent on GPU:
 errc=tensBlck_set_absence(tb0);
 errc=tensBlck_set_absence(tb1);
 errc=tensBlck_set_absence(tb2);
 errc=tensBlck_set_absence(tb3);
 errc=tensBlck_set_absence(tb4);
 errc=tensBlck_set_absence(tb5);

//A single tensor contraction:
 int cptrn[8]={4,-3,3,-1,-4,2,-2,1}; //D(a,b,c,d)+=L(d,k,c,l)*R(l,b,k,a)
 tms=clock();
 errc=gpu_tensor_block_contract_dlf_(cptrn,tb1,tb2,tb0,COPY_BACK,tsk0); if(errc){printf("#ERROR: gpu_tensor_block_contract_ [0] failed!"); return 1;};
 tsks[0]=tsk0; errc=cuda_tasks_wait(1,tsks,tsk_stats);
 tme=clock(); tmm=((double)(tme-tms))/CLOCKS_PER_SEC;
 if(errc){printf("#ERROR: cuda_tasks_wait [0] failed!"); return 1;};
 tm=cuda_task_time(tsk0,&incopy,&outcopy,&comput);
 printf("CUDA task 0 final status is %d, time %f %f %f %f \n",tsk_stats[0],tm,incopy,outcopy,comput);
 u_ext=(double)(U_DIM_EXT); c_ext=(double)(C_DIM_EXT);
 flops=u_ext*u_ext*u_ext*u_ext*c_ext*c_ext*2.0; //multiplications and additions
 printf("Effective GFlop/s  = %f \n",flops/(tmm*1073741824.0)); //single tensor contractions
 printf("On-GPU GFlop/s     = %f \n",flops/(comput*1073741824.0));
 printf("Destination element inspection [0]: %e \n",((double*)(tb0->elems_h))[13]);
 printf("Should be  %e \n",c_ext*c_ext*init_val*init_val);
 errc=cuda_task_clean(tsk0); if(errc){printf("#ERROR: cuda_task_clean [0] failed!"); return 1;};
 errc=tensBlck_set_absence(tb0);
 errc=tensBlck_set_absence(tb1);
 errc=tensBlck_set_absence(tb2);

//Tensor contractions (two concurrent tasks are executed on GPU asynchronously w.r.t. Host):
// int cptrn[8]={4,-3,3,-1,-4,2,-2,1}; //D(a,b,c,d)+=L(d,k,c,l)*R(l,b,k,a)
 tms=clock();
 errc=gpu_tensor_block_contract_dlf_(cptrn,tb1,tb2,tb0,COPY_BACK,tsk0); if(errc){printf("#ERROR: gpu_tensor_block_contract_ [0] failed!"); return 1;};
 errc=gpu_tensor_block_contract_dlf_(cptrn,tb4,tb5,tb3,COPY_BACK,tsk1); if(errc){printf("#ERROR: gpu_tensor_block_contract_ [1] failed!"); return 1;};
// Wait on completion:
// ...Do something on Host
 tsks[0]=tsk0; tsks[1]=tsk1; errc=cuda_tasks_wait(2,tsks,tsk_stats);
 tme=clock(); tmm=((double)(tme-tms))/CLOCKS_PER_SEC;
 if(errc){printf("#ERROR: cuda_tasks_wait [0,1] failed!"); return 1;};
// Print timings:
 tm=cuda_task_time(tsk0,&incopy,&outcopy,&comput);
 printf("CUDA task 0 final status is %d, time %f %f %f %f \n",tsk_stats[0],tm,incopy,outcopy,comput);
 tm=cuda_task_time(tsk1,&incopy,&outcopy,&comput);
 printf("CUDA task 1 final status is %d, time %f %f %f %f \n",tsk_stats[1],tm,incopy,outcopy,comput);
 u_ext=(double)(U_DIM_EXT); c_ext=(double)(C_DIM_EXT);
 flops=u_ext*u_ext*u_ext*u_ext*c_ext*c_ext*2.0; //multiplications and additions
 printf("Effective GFlop/s  = %f \n",2.0*flops/(tmm*1073741824.0)); //two tensor contractions
// Inspect results:
 printf("Destination element inspection [0]: %e \n",((double*)(tb0->elems_h))[13]);
 printf("Destination element inspection [3]: %e \n",((double*)(tb3->elems_h))[13]);
 printf("Should be  %e %e \n",c_ext*c_ext*init_val*init_val*2.0,c_ext*c_ext*init_val*init_val);
//---------------------------------------------------------------------------------

 printf("Cleaning ... ");
//Destroy CUDA task handles:
 errc=cuda_task_destroy(tsk0); if(errc){printf("#ERROR: cuda_task_destroy [0] failed!"); return 1;};
 errc=cuda_task_destroy(tsk1); if(errc){printf("#ERROR: cuda_task_destroy [1] failed!"); return 1;};
 errc=cuda_task_destroy(tsk2); if(errc){printf("#ERROR: cuda_task_destroy [2] failed!"); return 1;};
 errc=cuda_task_destroy(tsk3); if(errc){printf("#ERROR: cuda_task_destroy [3] failed!"); return 1;};
//Free tensor block memory:
 errc=tensBlck_free(tb0); if(errc){printf("#ERROR: tensBlck_free [0] failed!"); return 1;};
 errc=tensBlck_free(tb1); if(errc){printf("#ERROR: tensBlck_free [1] failed!"); return 1;};
 errc=tensBlck_free(tb2); if(errc){printf("#ERROR: tensBlck_free [2] failed!"); return 1;};
 errc=tensBlck_free(tb3); if(errc){printf("#ERROR: tensBlck_free [3] failed!"); return 1;};
 errc=tensBlck_free(tb4); if(errc){printf("#ERROR: tensBlck_free [4] failed!"); return 1;};
 errc=tensBlck_free(tb5); if(errc){printf("#ERROR: tensBlck_free [5] failed!"); return 1;};
//Destroy tensor blocks:
 errc=tensBlck_destroy(tb0); if(errc){printf("#ERROR: tensBlck_destroy [0] failed!"); return 1;};
 errc=tensBlck_destroy(tb1); if(errc){printf("#ERROR: tensBlck_destroy [1] failed!"); return 1;};
 errc=tensBlck_destroy(tb2); if(errc){printf("#ERROR: tensBlck_destroy [2] failed!"); return 1;};
 errc=tensBlck_destroy(tb3); if(errc){printf("#ERROR: tensBlck_destroy [3] failed!"); return 1;};
 errc=tensBlck_destroy(tb4); if(errc){printf("#ERROR: tensBlck_destroy [4] failed!"); return 1;};
 errc=tensBlck_destroy(tb5); if(errc){printf("#ERROR: tensBlck_destroy [5] failed!"); return 1;};
 printf("Ok\n");
//Deallocate argument buffers (finalizes both the Host and all GPUs):
 errc=arg_buf_deallocate(first_gpu,last_gpu); if(errc){printf("#ERROR: arg_buf_deallocate failed!"); return 1;};
 printf("Deallocated argument buffers\n");

//Exit:
 return 0;
}
