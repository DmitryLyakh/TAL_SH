/** Examples how to use GPU tensor algebra library (NV-TAL).
AUTHOR: Dmitry I. Lyakh: quant4me@gmail.com
LICENSE: GPL v2 **/

#include <stdio.h>
#include <time.h>
#include "tensor_algebra.h"

#define OCC 21 //occupied tensor dimension extent
#define VIR 81 //virtual tensor dimension extent
#define MAX_TASKS 32 //max number of simultaneously active tasks

const int first_gpu=0, last_gpu=0; //range of GPUs to use
size_t buf_size=1000000; //desired argument buffer size in bytes (Host pinned RAM): May be skipped
const double init_val=0.01; //just some initialization value for tensor blocks


void print_task_timings(int ntasks, const cudaTask_t * tsks, const int * tsk_stats, int first_task=0){
 int i;
 float tm,incopy,outcopy,comput;
//Print timings (total time, H2D time, D2H time, GPU computing time):
 for(i=first_task;i<first_task+ntasks;i++){
  tm=cuda_task_time(tsks[i],&incopy,&outcopy,&comput);
  printf("CUDA task %d final status is %d, time: %f %f %f %f \n",i,tsk_stats[i],tm,incopy,outcopy,comput);
 };
 return;
}


int main(int argc, char** args){
 tensBlck_t *Tvv,*Too,*Toovv,*Tvvoo,*Tvvvv,*Tvovo,*Tovov,*Tovvv,*Tooov,*Tvovv,*Tovvo,*Tvvvooo,*Tovvvoo,*Toovovv; //tensor blocks
 tensBlck_t *tb[14]; //aliases for tensor blocks
 cudaTask_t *tsks[MAX_TASKS]; //task handles 
 int i,errc,max_args,tsk_stats[MAX_TASKS];
 double flops,tmm;
 clock_t tms,tme;

//Allocate argument buffers (initializes both the Host and all GPUs):
 errc=arg_buf_allocate(&buf_size,&max_args,first_gpu,last_gpu); //if you don't want to use Host buffer, set buf_size to a small value
 if(errc){printf("#ERROR: arg_buf_allocate failed!"); return 1;};
 printf("Allocated argument buffers: Host buffer size = %zu bytes: max args = %d \n",buf_size,max_args);

//Activate a specific GPU:
 errc=gpu_activate(first_gpu); if(errc){printf("#ERROR: gpu_activate failed!"); return 1;};
 printf("Focused on GPU #%d \n",first_gpu);

//Create and initialize tensor blocks on Host:
 //Set up tensor ranks and dimension extents:
 int trank1=2; int dims1[2]={VIR,VIR};
 int trank2=2; int dims2[2]={OCC,OCC};
 int trank3=4; int dims3[4]={OCC,OCC,VIR,VIR};
 int trank4=4; int dims4[4]={VIR,VIR,OCC,OCC};
 int trank5=4; int dims5[4]={VIR,VIR,VIR,VIR};
 int trank6=4; int dims6[4]={VIR,OCC,VIR,OCC};
 int trank7=4; int dims7[4]={OCC,VIR,OCC,VIR};
 int trank8=4; int dims8[4]={OCC,VIR,VIR,VIR};
 int trank9=4; int dims9[4]={OCC,OCC,OCC,VIR};
 int trank10=4; int dims10[4]={VIR,OCC,VIR,VIR};
 int trank11=4; int dims11[4]={OCC,VIR,VIR,OCC};
 int trank12=6; int dims12[6]={VIR,VIR,VIR,OCC,OCC,OCC};
 int trank13=6; int dims13[6]={OCC,VIR,VIR,VIR,OCC,OCC};
 int trank14=6; int dims14[6]={OCC,OCC,VIR,OCC,VIR,VIR};
 //Toovovv:
 errc=tensBlck_create(&Toovovv); //create an empty tensor block
 if(errc){printf("#ERROR: Toovovv tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Toovovv,first_gpu,R8,trank14,dims14); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Toovovv tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Toovovv \n");
 //Tovvvoo:
 errc=tensBlck_create(&Tovvvoo); //create an empty tensor block
 if(errc){printf("#ERROR: Tovvvoo tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Tovvvoo,first_gpu,R8,trank13,dims13); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Tovvvoo tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Tovvvoo \n");
 //Tvvvooo:
 errc=tensBlck_create(&Tvvvooo); //create an empty tensor block
 if(errc){printf("#ERROR: Tvvvooo tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Tvvvooo,first_gpu,R8,trank12,dims12); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Tvvvooo tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Tvvvooo \n");
 //Tovvo:
 errc=tensBlck_create(&Tovvo); //create an empty tensor block
 if(errc){printf("#ERROR: Tovvo tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Tovvo,first_gpu,R8,trank11,dims11); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Tovvo tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Tovvo \n");
 //Tvovv:
 errc=tensBlck_create(&Tvovv); //create an empty tensor block
 if(errc){printf("#ERROR: Tvovv tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Tvovv,first_gpu,R8,trank10,dims10); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Tvovv tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Tvovv \n");
 //Tooov:
 errc=tensBlck_create(&Tooov); //create an empty tensor block
 if(errc){printf("#ERROR: Tooov tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Tooov,first_gpu,R8,trank9,dims9); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Tooov tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Tooov \n");
 //Tovvv:
 errc=tensBlck_create(&Tovvv); //create an empty tensor block
 if(errc){printf("#ERROR: Tovvv tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Tovvv,first_gpu,R8,trank8,dims8); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Tovvv tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Tovvv \n");
 //Tovov:
 errc=tensBlck_create(&Tovov); //create an empty tensor block
 if(errc){printf("#ERROR: Tovov tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Tovov,first_gpu,R8,trank7,dims7); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Tovov tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Tovov \n");
 //Tvovo:
 errc=tensBlck_create(&Tvovo); //create an empty tensor block
 if(errc){printf("#ERROR: Tvovo tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Tvovo,first_gpu,R8,trank6,dims6); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Tvovo tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Tvovo \n");
 //Tvvvv:
 errc=tensBlck_create(&Tvvvv); //create an empty tensor block
 if(errc){printf("#ERROR: Tvvvv tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Tvvvv,first_gpu,R8,trank5,dims5); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Tvvvv tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Tvvvv \n");
 //Tvvoo:
 errc=tensBlck_create(&Tvvoo); //create an empty tensor block
 if(errc){printf("#ERROR: Tvvoo tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Tvvoo,first_gpu,R8,trank4,dims4); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Tvvoo tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Tvvoo \n");
 //Toovv:
 errc=tensBlck_create(&Toovv); //create an empty tensor block
 if(errc){printf("#ERROR: Toovv tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Toovv,first_gpu,R8,trank3,dims3); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Toovv tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Toovv \n");
 //Too:
 errc=tensBlck_create(&Too); //create an empty tensor block
 if(errc){printf("#ERROR: Too tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Too,first_gpu,R8,trank2,dims2); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Too tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Too \n");
 //Tvv:
 errc=tensBlck_create(&Tvv); //create an empty tensor block
 if(errc){printf("#ERROR: Tvv tensBlck_create failed!"); return 1;};
 errc=tensBlck_alloc(Tvv,first_gpu,R8,trank1,dims1); //allocate memory on Host (pinned) and GPU, init to zero on Host
 if(errc){printf("#ERROR: Tvv tensBlck_alloc failed!"); return 1;};
 printf("Allocated tensor block Tvv \n");

//Create empty CUDA task handles:
 for(i=0;i<MAX_TASKS;i++){
  errc=cuda_task_create(&tsks[i]); if(errc){printf("#ERROR: cuda_task_create failed: %d \n",i); return 1;};
 };
//---------------------------------------------------------------------------------------------------------
//Tensor operations (examples):

//Tensor initialization (concurrent tasks are executed on GPU asynchronously w.r.t. Host):
 //Batch 1:
 errc=gpu_tensor_block_init_(Toovovv,init_val,COPY_BACK,tsks[0]); if(errc){printf("#ERROR: gpu_tensor_block_init_ 14 failed!\n"); return 1;};
 errc=gpu_tensor_block_init_(Tovvvoo,init_val,COPY_BACK,tsks[1]); if(errc){printf("#ERROR: gpu_tensor_block_init_ 13 failed!\n"); return 1;};
 errc=cuda_tasks_wait(2,tsks,tsk_stats); //wait on completion (or do something on Host)
 if(errc){printf("#ERROR: cuda_tasks_wait 1 failed!"); return 1;};
 print_task_timings(2,tsks,tsk_stats);
 
 
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
//---------------------------------------------------------------------------------------------

//Done. Release resources:
 printf("Cleaning ... ");
//Destroy CUDA task handles:
 for(i=0;i<MAX_TASKS;i++){
  errc=cuda_task_destroy(tsks[i]); if(errc){printf("#ERROR: cuda_task_destroy failed: %d \n",i); return 1;};
 };
//Free tensor block memory:
 errc=tensBlck_free(Tvv); if(errc){printf("#ERROR: tensBlck_free 1 failed!"); return 1;};
 errc=tensBlck_free(Too); if(errc){printf("#ERROR: tensBlck_free 2 failed!"); return 1;};
 errc=tensBlck_free(Toovv); if(errc){printf("#ERROR: tensBlck_free 3 failed!"); return 1;};
 errc=tensBlck_free(Tvvoo); if(errc){printf("#ERROR: tensBlck_free 4 failed!"); return 1;};
 errc=tensBlck_free(Tvvvv); if(errc){printf("#ERROR: tensBlck_free 5 failed!"); return 1;};
 errc=tensBlck_free(Tvovo); if(errc){printf("#ERROR: tensBlck_free 6 failed!"); return 1;};
 errc=tensBlck_free(Tovov); if(errc){printf("#ERROR: tensBlck_free 7 failed!"); return 1;};
 errc=tensBlck_free(Tovvv); if(errc){printf("#ERROR: tensBlck_free 8 failed!"); return 1;};
 errc=tensBlck_free(Tooov); if(errc){printf("#ERROR: tensBlck_free 9 failed!"); return 1;};
 errc=tensBlck_free(Tvovv); if(errc){printf("#ERROR: tensBlck_free 10 failed!"); return 1;};
 errc=tensBlck_free(Tovvo); if(errc){printf("#ERROR: tensBlck_free 11 failed!"); return 1;};
 errc=tensBlck_free(Tvvvooo); if(errc){printf("#ERROR: tensBlck_free 12 failed!"); return 1;};
 errc=tensBlck_free(Tovvvoo); if(errc){printf("#ERROR: tensBlck_free 13 failed!"); return 1;};
 errc=tensBlck_free(Toovovv); if(errc){printf("#ERROR: tensBlck_free 14 failed!"); return 1;};
//Destroy tensor blocks:
 errc=tensBlck_destroy(Tvv); if(errc){printf("#ERROR: tensBlck_destroy 1 failed!"); return 1;};
 errc=tensBlck_destroy(Too); if(errc){printf("#ERROR: tensBlck_destroy 2 failed!"); return 1;};
 errc=tensBlck_destroy(Toovv); if(errc){printf("#ERROR: tensBlck_destroy 3 failed!"); return 1;};
 errc=tensBlck_destroy(Tvvoo); if(errc){printf("#ERROR: tensBlck_destroy 4 failed!"); return 1;};
 errc=tensBlck_destroy(Tvvvv); if(errc){printf("#ERROR: tensBlck_destroy 5 failed!"); return 1;};
 errc=tensBlck_destroy(Tvovo); if(errc){printf("#ERROR: tensBlck_destroy 6 failed!"); return 1;};
 errc=tensBlck_destroy(Tovov); if(errc){printf("#ERROR: tensBlck_destroy 7 failed!"); return 1;};
 errc=tensBlck_destroy(Tovvv); if(errc){printf("#ERROR: tensBlck_destroy 8 failed!"); return 1;};
 errc=tensBlck_destroy(Tooov); if(errc){printf("#ERROR: tensBlck_destroy 9 failed!"); return 1;};
 errc=tensBlck_destroy(Tvovv); if(errc){printf("#ERROR: tensBlck_destroy 10 failed!"); return 1;};
 errc=tensBlck_destroy(Tovvo); if(errc){printf("#ERROR: tensBlck_destroy 11 failed!"); return 1;};
 errc=tensBlck_destroy(Tvvvooo); if(errc){printf("#ERROR: tensBlck_destroy 12 failed!"); return 1;};
 errc=tensBlck_destroy(Tovvvoo); if(errc){printf("#ERROR: tensBlck_destroy 13 failed!"); return 1;};
 errc=tensBlck_destroy(Toovovv); if(errc){printf("#ERROR: tensBlck_destroy 14 failed!"); return 1;};
 printf("Ok\n");
//Deallocate argument buffers (finalizes the Host and all GPUs):
 errc=arg_buf_deallocate(first_gpu,last_gpu); if(errc){printf("#ERROR: arg_buf_deallocate failed!\n"); return 1;};
 printf("Deallocated argument buffers\n");
 printf("Success!\n");

//Exit:
 return 0;
}
