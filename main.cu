/** Examples how to use GPU tensor algebra library (NV-TAL).
There are several drawbacks in the current implementation of NV-TAL.
All of them will be fixed very soon with a release of the unified
tensor algebra library TAL-SH that will combine CP-TAL (multicore CPU),
NV-TAL (NVidia GPU), and XP-TAL (Intel Xeon Phi) under a unified
API interface with a transparent data movement runtime.
Anticipated date is September 2015.
AUTHOR: Dmitry I. Lyakh: quant4me@gmail.com
LICENSE: GPL v2 **/

#include <stdio.h>
#include <time.h>
#include "tensor_algebra.h"

#define DM0 1  //the smallest possible dimension extent
#define DM1 11 //small dimension extent
#define DM2 31 //medium small dimension extent
#define DM3 51 //medium large dimension extent
#define DM4 71 //large dimension extent
#define DM5 81 //the largest dimension extent
#define MAX_TASKS 32 //max number of simultaneously active tasks

//User input:
const int first_gpu=0, last_gpu=3; //range of NVidia GPUs to use
size_t buf_size=1000000; //desired argument buffer size in bytes (Host pinned RAM): Currently ignored
const double init_val=0.01; //some initialization value for tensor blocks


void print_task_timings(int ntasks, cudaTask_t * tsks[], const int * tsk_stats, int first_task=0){
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
 double flops,tmm;
 clock_t tms,tme;
 tensBlck_t *tb[24]; //tensor blocks
 cudaTask_t *tsks[MAX_TASKS]; //task handles
 int i,j,k,my_gpu,errc,max_args,tsk_stats[MAX_TASKS];
 int rank[24]={4,4,4,4,4,4, 6,4,4,6,4,4, 2,4,4,2,4,4, 4,2,4,4,2,4}; //tensor block ranks
 int dims[24][6]={ //Tensor block dimension extents:
  DM4,DM4,DM4,DM4,0,0,     DM4,DM4,DM4,DM4,0,0,  DM4,DM4,DM4,DM4,0,0,  //Contraction 1: D(a,b,i,j)+=L(k,l,a,b)*R(k,l,i,j)
  DM4,DM4,DM4,DM4,0,0,     DM4,DM4,DM4,DM4,0,0,  DM4,DM4,DM4,DM4,0,0,  //Contraction 2: D(a,b,i,j)+=L(j,k,i,l)*R(l,b,k,a)
  DM1,DM2,DM2,DM2,DM2,DM1, DM1,DM1,DM2,DM2,0,0,  DM1,DM2,DM2,DM1,0,0,  //Contraction 3: D(a,b,c,i,j,k)+=L(d,a,b,c)*R(d,i,j,k)
  DM1,DM2,DM2,DM2,DM2,DM1, DM1,DM1,DM2,DM2,0,0,  DM2,DM2,DM1,DM1,0,0,  //Contraction 4: D(a,b,c,i,j,k)+=L(d,k,j,i)*R(c,b,a,d)
  DM5,DM5,0,0,0,0,         DM5,DM5,DM5,DM5,0,0,  DM5,DM5,DM5,DM5,0,0,  //Contraction 5: D(a,i)+=L(k,l,m,a)*R(k,l,m,i)
  DM5,DM5,0,0,0,0,         DM5,DM5,DM5,DM5,0,0,  DM5,DM5,DM5,DM5,0,0,  //Contraction 6: D(a,i)+=L(k,l,m,i)*R(a,m,l,k)
  DM5,DM5,DM5,DM5,0,0,     DM5,DM5,0,0,0,0,      DM5,DM5,DM5,DM5,0,0,  //Contraction 7: D(a,b,i,j)+=L(c,a)*R(c,b,i,j)
  DM5,DM5,DM5,DM5,0,0,     DM5,DM5,0,0,0,0,      DM5,DM5,DM5,DM5,0,0   //Contraction 8: D(a,b,i,j)+=L(c,j)*R(i,b,a,c)
 };
 int cptrn[8][8]={ //Tensor contraction patterns:
  -1,-2, 1, 2,-1,-2, 3, 4, //Contraction pattern 1
   4,-3, 3,-1,-4, 2,-2, 1, //Contraction pattern 2
  -1, 1, 2, 3,-1, 4, 5, 6, //Contraction pattern 3
  -4, 6, 5, 4, 3, 2, 1,-1, //Contraction pattern 4
  -1,-2,-3, 1,-1,-2,-3, 2, //Contraction pattern 5
  -4,-3,-2, 2, 1,-3,-2,-1, //Contraction pattern 6
  -1, 1,-1, 2, 3, 4, 0, 0, //Contraction pattern 7
  -4, 4, 3, 2, 1,-1, 0, 0  //Contraction pattern 8
 };
 double cflops[8];

//Set the number of Flops for each tensor contraction:
 cflops[0]=2.0*double(dims[0][0])*double(dims[0][1])*double(dims[0][2])*double(dims[0][3])*double(dims[1][0])*double(dims[1][1]);                    //Flops for tensor contraction 1
 cflops[1]=cflops[0];                                                                                                                                //Flops for tensor contraction 2
 cflops[2]=2.0*double(dims[6][0])*double(dims[6][1])*double(dims[6][2])*double(dims[6][3])*double(dims[6][4])*double(dims[6][5])*double(dims[7][0]); //Flops for tensor contraction 3
 cflops[3]=cflops[2];                                                                                                                                //Flops for tensor contraction 4
 cflops[4]=2.0*double(dims[12][0])*double(dims[12][1])*double(dims[13][0])*double(dims[13][1])*double(dims[13][2]);                                  //Flops for tensor contraction 5
 cflops[5]=cflops[4];                                                                                                                                //Flops for tensor contraction 6
 cflops[6]=2.0*double(dims[18][0])*double(dims[18][1])*double(dims[18][2])*double(dims[18][3])*double(dims[19][0]);                                  //Flops for tensor contraction 7
 cflops[7]=cflops[6];                                                                                                                                //Flops for tensor contraction 8

//Allocate argument buffers (initializes both the Host and all GPUs):
 errc=arg_buf_allocate(&buf_size,&max_args,first_gpu,last_gpu); //if you don't want to use Host buffer, set buf_size to a small value
 if(errc){printf("#ERROR: arg_buf_allocate failed!"); return 1;};
 printf("Allocated argument buffers: Host buffer size = %zu bytes: max args = %d \n",buf_size,max_args);

//Check all GPUs:
 for(i=first_gpu;i<=last_gpu;i++){
  errc=gpu_activate(i); if(errc){printf("#ERROR: gpu_activate %d failed!",i); return 1;};
  printf("Activated GPU %d \n",i);
 };

//Create and initialize 24 tensor blocks:
 for(j=1;j<=8;j++){ //eight tensor contractions
  my_gpu=first_gpu+(j-1)/2; //expecting 4 GPUs
  for(i=(j-1)*3;i<j*3;i++){ //tensor block number
   errc=tensBlck_create(&(tb[i])); //create an empty tensor block
   if(errc){printf("#ERROR: tensBlck_create %d failed!",i); return 1;};
   errc=tensBlck_alloc(tb[i],my_gpu,R8,rank[i],&(dims[i][0])); //allocate memory on Host (pinned) and GPU (buffer), init to zero on Host
   if(errc){printf("#ERROR: tensBlck_alloc %d failed!",i); return 1;};
   printf("Allocated tensor block %d: CPU/GPU address: %p / %p\n",i,tb[i]->elems_h,tb[i]->elems_d);
  };
 };

//Create empty CUDA task handles:
 for(i=0;i<MAX_TASKS;i++){errc=cuda_task_create(&(tsks[i])); if(errc){printf("#ERROR: cuda_task_create %d failed! \n",i); return 1;};};
 printf("Created %d empty task handles \n",MAX_TASKS);
//----------------------------------------------------------
//TENSOR OPERATIONS (examples):

//Execute tensor initializations (concurrent tasks are executed on GPUs asynchronously w.r.t. Host):
 k=0;
 for(j=1;j<=8;j++){ //tensor contraction number
  i=(j-1)*3; //destination tensor block number
  errc=gpu_tensor_block_init_(tb[i],0.0,COPY_BACK,tsks[k++]); if(errc){printf("#ERROR: gpu_tensor_block_init_ D %d failed!\n",i); return 1;};
  errc=gpu_tensor_block_init_(tb[i+1],init_val,COPY_BACK,tsks[k++]); if(errc){printf("#ERROR: gpu_tensor_block_init_ L %d failed!\n",i); return 1;};
  errc=gpu_tensor_block_init_(tb[i+2],init_val,COPY_BACK,tsks[k++]); if(errc){printf("#ERROR: gpu_tensor_block_init_ R %d failed!\n",i); return 1;};
 };
 errc=cuda_tasks_wait(k,tsks,tsk_stats); //wait on completion (or do something on Host)
 if(errc){printf("#ERROR: cuda_tasks_wait for initializations failed!"); return 1;};
 print_task_timings(k,tsks,tsk_stats); //print timings for all tasks

//Clean CUDA tasks for reuse:
 for(i=0;i<k;i++){errc=cuda_task_clean(tsks[i]); if(errc){printf("#ERROR: cuda_task_clean %d failed!",i); return 1;};};

//Mark tensor blocks as absent on GPU (to enforce PCIe copy):
 for(i=0;i<24;i++){errc=tensBlck_set_absence(tb[i]);};

//Choose the tensor transpose and matrix multiplication algorithms:
 gpu_set_transpose_algorithm(EFF_TRN_ON); //turn off the efficient tensor transpose (slow)
 gpu_set_matmult_algorithm(BLAS_ON); //switch to custom BLAS kernels (slow)

//Execute tensor contractions (concurrent tasks are executed on GPUs asynchronously w.r.t. Host):
 tms=clock(); flops=0.0; k=0;
 errc=gpu_tensor_block_contract_dlf_(&(cptrn[0][0]),tb[1],tb[2],tb[0],COPY_BACK,tsks[k++]); flops+=cflops[0];
 if(errc){printf("#ERROR: gpu_tensor_block_contract_ %d failed!",1); return 1;};
 errc=gpu_tensor_block_contract_dlf_(&(cptrn[1][0]),tb[4],tb[5],tb[3],COPY_BACK,tsks[k++]); flops+=cflops[1];
 if(errc){printf("#ERROR: gpu_tensor_block_contract_ %d failed!",2); return 1;};
// errc=gpu_tensor_block_contract_dlf_(&(cptrn[2][0]),tb[7],tb[8],tb[6],COPY_BACK,tsks[k++]); flops+=cflops[2];
// if(errc){printf("#ERROR: gpu_tensor_block_contract_ %d failed!",3); return 1;};
// errc=gpu_tensor_block_contract_dlf_(&(cptrn[3][0]),tb[10],tb[11],tb[9],COPY_BACK,tsks[k++]); flops+=cflops[3];
// if(errc){printf("#ERROR: gpu_tensor_block_contract_ %d failed!",4); return 1;};
// errc=gpu_tensor_block_contract_dlf_(&(cptrn[4][0]),tb[13],tb[14],tb[12],COPY_BACK,tsks[k++]); flops+=cflops[4];
// if(errc){printf("#ERROR: gpu_tensor_block_contract_ %d failed!",5); return 1;};
// errc=gpu_tensor_block_contract_dlf_(&(cptrn[5][0]),tb[16],tb[17],tb[15],COPY_BACK,tsks[k++]); flops+=cflops[5];
// if(errc){printf("#ERROR: gpu_tensor_block_contract_ %d failed!",6); return 1;};
// errc=gpu_tensor_block_contract_dlf_(&(cptrn[6][0]),tb[19],tb[20],tb[18],COPY_BACK,tsks[k++]); flops+=cflops[6];
// if(errc){printf("#ERROR: gpu_tensor_block_contract_ %d failed!",7); return 1;};
// errc=gpu_tensor_block_contract_dlf_(&(cptrn[7][0]),tb[22],tb[23],tb[21],COPY_BACK,tsks[k++]); flops+=cflops[7];
// if(errc){printf("#ERROR: gpu_tensor_block_contract_ %d failed!",8); return 1;};
 errc=cuda_tasks_wait(k,tsks,tsk_stats);
 tme=clock(); tmm=((double)(tme-tms))/CLOCKS_PER_SEC;
 if(errc){printf("#ERROR: cuda_tasks_wait for contractions failed!"); return 1;};
 print_task_timings(k,tsks,tsk_stats); //print timings for all tasks
 printf("Cumulative GFlop/s = %f; total time = %f sec\n",flops/(tmm*1073741824.0),tmm);

//Inspect the results:
 for(i=0;i<24;i+=3){
  printf("Destination element inspection %d: %e %e\n",i,((double*)(tb[i]->elems_h))[13],((double*)(tb[i]->elems_h))[133]);
 };
//------------------------------------------------------------------------------------------------------------------------

//Done. Release resources:
 printf("Cleaning ... ");
//Destroy CUDA task handles:
 for(i=0;i<MAX_TASKS;i++){
  errc=cuda_task_destroy(tsks[i]); if(errc){printf("#ERROR: cuda_task_destroy %d failed!",i); return 1;};
 };
//Free tensor block memory:
 for(i=0;i<24;i++){
  errc=tensBlck_free(tb[i]); if(errc){printf("#ERROR: tensBlck_free %d failed!",i); return 1;};
 };
//Destroy tensor blocks:
 for(i=0;i<24;i++){
  errc=tensBlck_destroy(tb[i]); if(errc){printf("#ERROR: tensBlck_destroy %d failed!",i); return 1;};
 };
 printf("Ok\n");
//Deallocate argument buffers (finalizes the Host and all GPUs):
 errc=arg_buf_deallocate(first_gpu,last_gpu); if(errc){printf("#ERROR: arg_buf_deallocate failed!\n"); return 1;};
 printf("Deallocated argument buffers\n");
 printf("Success!\n");

//Exit:
 return 0;
}
