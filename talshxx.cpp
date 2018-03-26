/** ExaTensor::TAL-SH: Device-unified user-level C++ API implementation.
REVISION: 2018/03/26

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
#include <assert.h>

namespace talsh{


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


void shutdown()
{
 int errc = talshShutdown();
 assert(errc == TALSH_SUCCESS);
 return;
}


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
 errc = talshTaskClean(&write_task_);
 assert(errc == TALSH_SUCCESS);
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
 errc = talshTaskClean(&write_task_);
 assert(errc == TALSH_SUCCESS);
}


Tensor::~Tensor()
{
 assert(used_ == 0);
 int errc = talshTensorDestruct(&tensor_);
 assert(errc == TALSH_SUCCESS);
}


void Tensor::print()
{
 std::cout << "TAL-SH Tensor {";
 std::size_t rank = signature_.size();
 for(std::size_t i = 0; i < rank - 1; ++i) std::cout << signature_.begin()[i] << ",";
 if(rank > 0) std::cout << signature_.begin()[rank-1];
 std::cout << "} [use=" << used_ << "]:" << std::endl;
 talshTensorPrintInfo(&tensor_);
 return;
}

} //namespace talsh
