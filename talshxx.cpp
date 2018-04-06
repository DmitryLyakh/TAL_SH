/** ExaTensor::TAL-SH: Device-unified user-level C++ API implementation.
REVISION: 2018/04/06

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
**/

#include <iostream>
#include <complex>
#include <initializer_list>
#include <string>
#include <assert.h>

#include "talsh.h"
#include "talsh_task.hpp"

namespace talsh{


template <typename T>
Tensor::Tensor(const std::initializer_list<std::size_t> signature, //tensor signature (identifier): signature[0:rank-1]
               const std::initializer_list<int> dims,              //tensor dimension extents: dims[0:rank-1]
               const T init_val):                                  //scalar initialization value (its type will define tensor element data kind)
 signature_(signature), used_(0)
{
 static_assert(TensorData<T>::supported,"Tensor data type is not supported!");
 int errc = talshTensorClean(&tensor_);
 assert(errc == TALSH_SUCCESS);
 const int rank = static_cast<int>(dims.size());
 errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.begin(),talshFlatDevId(DEV_HOST,0),NULL,-1,NULL,
                             realPart(init_val),imagPart(init_val));
 assert(errc == TALSH_SUCCESS && signature.size() == dims.size());
 write_task_ = nullptr;
}


template <typename T>
Tensor::Tensor(const std::initializer_list<std::size_t> signature, //tensor signature (identifier): signature[0:rank-1]
               const std::initializer_list<int> dims,              //tensor dimension extents: dims[0:rank-1]
               T * ext_mem,                                        //pointer to an external memory storage where the tensor body will reside
               const T * init_val):                                //optional scalar initialization value (provide nullptr if not needed)
 signature_(signature), used_(0)
{
 static_assert(TensorData<T>::supported,"Tensor data type is not supported!");
 int errc = talshTensorClean(&tensor_);
 assert(errc == TALSH_SUCCESS);
 assert(ext_mem != nullptr);
 const int rank = static_cast<int>(dims.size());
 if(init_val == nullptr){
  errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.begin(),talshFlatDevId(DEV_HOST,0),(void*)ext_mem);
 }else{
  std::cout << "FATAL: Initialization of tensors with external memory storage is not implemented in TAL-SH yet!" << std::endl; assert(false);
  errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.begin(),talshFlatDevId(DEV_HOST,0),(void*)ext_mem,-1,NULL,
                              realPart(*init_val),imagPart(*init_val));
 }
 assert(errc == TALSH_SUCCESS && signature.size() == dims.size());
 write_task_ = nullptr;
}


Tensor::~Tensor()
{
 assert(used_ == 0 && write_task_ == nullptr);
 int errc = talshTensorDestruct(&tensor_);
 assert(errc == TALSH_SUCCESS);
}


/** Returns the tensor rank (order in math terms). **/
int Tensor::getRank() const
{
 return talshTensorRank(&tensor_);
}
/** Returns the tensor order (rank in phys terms). **/
int Tensor::getOrder() const
{
 return this->getRank();
}


/** Use counter increment. **/
Tensor & Tensor::operator++()
{
 ++used_;
 return *this;
}


/** Use counter decrement. **/
Tensor & Tensor::operator--()
{
 assert(used_ > 0);
 --used_;
 return *this;
}


/** Synchronizes the tensor presence on a given device.
    Returns TRUE on success, FALSE if an active write task
    on this tensor has failed to complete successfully. **/
bool Tensor::sync(const int device_kind, const int device_id, void * dev_mem)
{
 bool res = this->complete_write_task();
 if(res){
  int errc = talshTensorPlace(&tensor_,device_id,device_kind,dev_mem);
  assert(errc == TALSH_SUCCESS);
 }
 return res;
}


/** Performs a tensor contraction of two tensors and accumulates the result into the current tensor:
    this += left * right * factor **/
template <typename T>
int Tensor::contractAccumulate(TensorTask * task_handle,    //out: task handle associated with this operation or nullptr (synchronous)
                               const std::string & pattern, //in: contraction pattern string
                               Tensor & left,               //in: left tensor
                               Tensor & right,              //in: right tensor
                               const int device_kind,       //in: execution device kind
                               const int device_id,         //in: execution device id
                               const T factor)              //in: alpha factor
{
 int errc = TALSH_SUCCESS;
 this->complete_write_task();
 const char * contr_ptrn = pattern.c_str();
 talsh_tens_t * dtens = this->get_talsh_tensor_ptr();
 talsh_tens_t * ltens = left.get_talsh_tensor_ptr();
 talsh_tens_t * rtens = right.get_talsh_tensor_ptr();
 if(task_handle != nullptr){ //asynchronous
  assert(task_handle->is_empty());
  talsh_task_t * task_hl = task_handle->get_talsh_task_ptr();
  //++left; ++right; ++(*this);
  errc = talshTensorContract(contr_ptrn,dtens,ltens,rtens,realPart(factor),imagPart(factor),device_id,device_kind,COPY_MTT,task_hl);
  //if(errc != TALSH_SUCCESS) std::cout << "#ERROR(talsh::Tensor::contractAccumulate): talshTensorContract error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
  if(errc == TALSH_SUCCESS) write_task_ = task_handle;
 }else{ //synchronous
  errc = talshTensorContract(contr_ptrn,dtens,ltens,rtens,realPart(factor),imagPart(factor),device_id,device_kind,COPY_MTT);
  //if(errc != TALSH_SUCCESS) std::cout << "#ERROR(talsh::Tensor::contractAccumulate): talshTensorContract error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
 }
 return errc;
}


/** Performs a matrix multiplication on two tensors and accumulates the result into the current tensor. **/
template <typename T>
int Tensor::multiplyAccumulate(TensorTask * task_handle, //out: task handle associated with this operation or nullptr (synchronous)
                               Tensor & left,            //in: left tensor
                               Tensor & right,           //in: right tensor
                               const int device_kind,    //in: execution device kind
                               const int device_id,      //in: execution device id
                               const T factor)           //in: alpha factor
{
 int errc = TALSH_SUCCESS;
 char cptrn[MAX_CONTRACTION_PATTERN_LEN];
 int dptrn[MAX_TENSOR_RANK*2];
 int drank = this->getRank();
 int lrank = left.getRank();
 int rrank = right.getRank();
 assert(lrank + rrank >= drank && (lrank + rrank - drank)%2 == 0);
 int nc = (lrank + rrank - drank)/2; //number of contracted indices
 int nl = lrank - nc; //number of left open indices
 int nr = rrank - nc; //number of right open indices
 //Create the digital contraction pattern:
 int l = 0;
 for(int i = 0; i < nl; ++i){dptrn[l++] = (i+1);}
 for(int i = 0; i < nc; ++i){dptrn[l++] = -(i+1);}
 for(int i = 0; i < nc; ++i){dptrn[l++] = -(nl+1+i);}
 for(int i = 0; i < nr; ++i){dptrn[l++] = (nl+1+i);}
 //Convert the digital contraction pattern into a symbolc one:
 int cpl;
 get_contr_pattern_sym(&lrank,&rrank,dptrn,cptrn,&cpl,&errc); cptrn[cpl]='\0';
 assert(errc == 0);
 std::string contr_ptrn(cptrn);
 std::cout << contr_ptrn << std::endl; //debug
 //Execute tensor contraction:
 errc = this->contractAccumulate(task_handle,contr_ptrn,left,right,device_kind,device_id,factor);
 return errc;
}


/** Prints the tensor. **/
void Tensor::print() const
{
 std::cout << "TAL-SH Tensor {";
 std::size_t rank = signature_.size();
 for(std::size_t i = 0; i < rank - 1; ++i) std::cout << signature_.begin()[i] << ",";
 if(rank > 0) std::cout << signature_.begin()[rank-1];
 std::cout << "} [use=" << used_ << "]:" << std::endl;
 talshTensorPrintInfo(&tensor_);
 return;
}


talsh_tens_t * Tensor::get_talsh_tensor_ptr()
{
 return &tensor_;
}


/** Completes the current write task on the tensor, if any. **/
bool Tensor::complete_write_task()
{
 bool res = true;
 if(write_task_ != nullptr){
  res = write_task_->wait();
  write_task_ = nullptr;
 }
 return res;
}


/** Initializes TAL-SH runtime. **/
void initialize(std::size_t * host_buffer_size)
{
 int num_gpu, gpu_list[MAX_GPUS_PER_NODE];
 int errc = talshGetDeviceCount(DEV_NVIDIA_GPU,&num_gpu);
 assert(errc == TALSH_SUCCESS && num_gpu >= 0);
 if(num_gpu > 0){for(int i = 0; i < num_gpu; ++i) gpu_list[i]=i;};

 int host_arg_max;
 if(host_buffer_size == nullptr){
  std::size_t buf_size = DEFAULT_HOST_BUFFER_SIZE;
  errc = talshInit(&buf_size,&host_arg_max,num_gpu,gpu_list,0,NULL,0,NULL);
 }else{
  errc = talshInit(host_buffer_size,&host_arg_max,num_gpu,gpu_list,0,NULL,0,NULL);
 }
 assert(errc == TALSH_SUCCESS);
 return;
}


/** Shutsdown TAL-SH runtime. **/
void shutdown()
{
 int errc = talshShutdown();
 assert(errc == TALSH_SUCCESS);
 return;
}


} //namespace talsh
