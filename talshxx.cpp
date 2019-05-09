/** ExaTensor::TAL-SH: Device-unified user-level C++ API implementation.
REVISION: 2019/05/09

Copyright (C) 2014-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2019 Oak Ridge National Laboratory (UT-Battelle)

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

#include "talshxx.hpp"

namespace talsh{

//Static constant storage:

constexpr float TensorData<float>::unity;
constexpr float TensorData<float>::zero;
constexpr double TensorData<double>::unity;
constexpr double TensorData<double>::zero;
constexpr std::complex<float> TensorData<std::complex<float>>::unity;
constexpr std::complex<float> TensorData<std::complex<float>>::zero;
constexpr std::complex<double> TensorData<std::complex<double>>::unity;
constexpr std::complex<double> TensorData<std::complex<double>>::zero;


//Helper functions:
// Generic real/imaginary part extraction:
double realPart(float number){return static_cast<double>(number);}
double realPart(double number){return number;}
double realPart(std::complex<float> number){return static_cast<double>(number.real());}
double realPart(std::complex<double> number){return number.real();}
double imagPart(float number){return 0.0f;}
double imagPart(double number){return 0.0;}
double imagPart(std::complex<float> number){return static_cast<double>(number.imag());}
double imagPart(std::complex<double> number){return number.imag();}


//Functions:

Tensor::Impl::~Impl()
{
 if(used_ != 0) std::cout << "#ERROR(Tensor::Impl::~Impl): Non-zero use count = " << used_ << std::endl;
 if(write_task_ != nullptr) std::cout << "#ERROR(Tensor::Impl::~Impl): Non-null task pointer = " << (void*)write_task_ << std::endl;
 assert(used_ == 0 && write_task_ == nullptr);
 int errc = talshTensorDestruct(&tensor_);
 assert(errc == TALSH_SUCCESS);
}


/** Returns the tensor rank (order in math terms). **/
int Tensor::getRank() const
{
 return talshTensorRank(&(pimpl_->tensor_));
}

/** Returns the tensor order (rank in phys terms). **/
int Tensor::getOrder() const
{
 return this->getRank();
}


/** Returns the tensor volume (number of elements). **/
std::size_t Tensor::getVolume() const
{
 return talshTensorVolume(&(pimpl_->tensor_));
}


/** Returns tensor dimension extents (and tensor order). **/
const int * Tensor::getDimExtents(unsigned int & num_dims) const
{
 num_dims = static_cast<unsigned int>((pimpl_->tensor_).shape_p->num_dim);
 if(num_dims == 0) return nullptr;
 return (pimpl_->tensor_).shape_p->dims;
}


/** Reshapes the tensor to a different shape of the same volume. **/
int Tensor::reshape(const std::vector<int> & dims)
{
 int rank = dims.size();
 return talshTensorReshape(&(pimpl_->tensor_),rank,dims.data());
}


/** Returns the extent of a specific tensor dimension. **/
int Tensor::getDimExtent(unsigned int dim) const
{
 int n = (pimpl_->tensor_).shape_p->num_dim;
 assert(dim < n);
 return ((pimpl_->tensor_).shape_p->dims)[dim];
}


/** Use counter increment. **/
Tensor & Tensor::operator++()
{
 ++(pimpl_->used_);
 return *this;
}

/** Use counter decrement. **/
Tensor & Tensor::operator--()
{
 assert(pimpl_->used_ > 0);
 --(pimpl_->used_);
 return *this;
}


/** Synchronizes the tensor presence on the given device.
    Returns TRUE on success, FALSE if an active write task
    on this tensor has failed to complete successfully. **/
bool Tensor::sync(const int device_kind, const int device_id, void * device_mem, bool exclusive)
{
 bool res = this->completeWriteTask();
 if(res){
  int errc;
  if(device_mem != nullptr){ //client provided an explicit buffer to place the tensor into
   errc = talshTensorPlace(&(pimpl_->tensor_),device_id,device_kind,device_mem);
  }else{ //no explicit buffer provided, use saved information (if any)
   if(device_kind == DEV_HOST){
    errc = talshTensorPlace(&(pimpl_->tensor_),device_id,device_kind,pimpl_->host_mem_);
   }else{
    errc = talshTensorPlace(&(pimpl_->tensor_),device_id,device_kind);
   }
  }
  assert(errc == TALSH_SUCCESS);
  if(exclusive){
   errc = talshTensorDiscardOther(&(pimpl_->tensor_),device_id,device_kind);
   assert(errc == TALSH_SUCCESS);
  }
 }
 return res;
}


/** Returns TRUE if the tensor is ready (has been computed).
    If ready, synchronizes the tensor presence on the given device. **/
bool Tensor::ready(int * status, const int device_kind, const int device_id, void * device_mem)
{
 *status = TALSH_TASK_EMPTY;
 bool res = this->testWriteTask(status);
 if(res){
  if(*status == TALSH_TASK_COMPLETED){
   int errc;
   if(device_mem != nullptr){ //client provided an explicit buffer to place the tensor into
    errc = talshTensorPlace(&(pimpl_->tensor_),device_id,device_kind,device_mem);
   }else{ //no explicit buffer provided, use saved information (if any)
    if(device_kind == DEV_HOST){
     errc = talshTensorPlace(&(pimpl_->tensor_),device_id,device_kind,pimpl_->host_mem_);
    }else{
     errc = talshTensorPlace(&(pimpl_->tensor_),device_id,device_kind);
    }
   }
   assert(errc == TALSH_SUCCESS);
  }else{
   assert(*status == TALSH_TASK_EMPTY);
  }
 }
 return res;
}


/** Prints the tensor info. **/
void Tensor::print() const
{
 std::cout << "TAL-SH Tensor {";
 std::size_t rank = (pimpl_->signature_).size();
 if(rank > 0){
  for(std::size_t i = 0; i < rank - 1; ++i) std::cout << (pimpl_->signature_).at(i) << ",";
  std::cout << (pimpl_->signature_).at(rank-1);
 }
 std::cout << "} [use=" << pimpl_->used_ << "]:" << std::endl;
 talshTensorPrintInfo(&(pimpl_->tensor_));
 return;
}


/** Prints the tensor info and body. **/
void Tensor::print(double thresh) const
{
 std::cout << "TAL-SH Tensor {";
 std::size_t rank = (pimpl_->signature_).size();
 if(rank > 0){
  for(std::size_t i = 0; i < rank - 1; ++i) std::cout << (pimpl_->signature_).at(i) << ",";
  std::cout << (pimpl_->signature_).at(rank-1);
 }
 std::cout << "} [use=" << pimpl_->used_ << "]:" << std::endl;
 talshTensorPrintInfo(&(pimpl_->tensor_));
 talshTensorPrintBody(&(pimpl_->tensor_),thresh);
 return;
}


talsh_tens_t * Tensor::getTalshTensorPtr()
{
 return &(pimpl_->tensor_);
}


/** Completes the current write task on the tensor, if any. **/
bool Tensor::completeWriteTask()
{
 bool res = true;
 if(pimpl_->write_task_ != nullptr){
  res = pimpl_->write_task_->wait();
  pimpl_->write_task_ = nullptr;
 }
 return res;
}


/** Tests the completion of the current write task on the tensor, if any. **/
bool Tensor::testWriteTask(int * status)
{
 bool res = true;
 *status = TALSH_TASK_EMPTY;
 if(pimpl_->write_task_ != nullptr){
  res = pimpl_->write_task_->test(status);
  if(res && *status == TALSH_TASK_COMPLETED) pimpl_->write_task_ = nullptr;
 }
 return res;
}


int Tensor::norm1(TensorTask * task_handle, //out: task handle associated with this operation or nullptr (synchronous)
                  double * tens_norm1,      //out: 1-norm of the tensor
                  const int device_kind,    //in: execution device kind
                  const int device_id)      //in: execution device id
{
 int errc = TALSH_SUCCESS;
 bool synced = this->sync(DEV_HOST,0);
 if(synced){
  talsh_tens_t * tens = this->getTalshTensorPtr();
  *tens_norm1 = talshTensorImageNorm1_cpu(tens);
 }else{
  errc = TALSH_FAILURE;
 }
 return errc;
}


int Tensor::extractSlice(TensorTask * task_handle,         //out: task handle associated with this operation or nullptr (synchronous)
                         Tensor & slice,                   //inout: extracted tensor slice
                         const std::vector<int> & offsets, //in: base offsets of the slice (0-based)
                         const int device_kind,            //in: execution device kind
                         const int device_id,              //in: execution device id
                         bool accumulative)                //in: accumulative VS overwrite (defaults to overwrite)
{
 int errc = TALSH_SUCCESS;
 int accum = NOPE; if(accumulative) accum = YEP;
 this->completeWriteTask();
 talsh_tens_t * ltens = this->getTalshTensorPtr();
 talsh_tens_t * dtens = slice.getTalshTensorPtr();
 if(task_handle != nullptr){ //asynchronous
  assert(task_handle->isEmpty());
  talsh_task_t * task_hl = task_handle->getTalshTaskPtr();
  //++left; ++right; ++(*this);
  errc = talshTensorSlice(dtens,ltens,offsets.data(),device_id,device_kind,COPY_MT,accum,task_hl);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::extractSlice): talshTensorSlice error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
  if(errc == TALSH_SUCCESS){
   pimpl_->write_task_ = task_handle;
  }else{
   task_handle->clean();
  }
 }else{ //synchronous
  errc = talshTensorSlice(dtens,ltens,offsets.data(),device_id,device_kind,COPY_MT,accum);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::extractSlice): talshTensorSlice error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
 }
 return errc;
}


int Tensor::insertSlice(TensorTask * task_handle,         //out: task handle associated with this operation or nullptr (synchronous)
                        Tensor & slice,                   //inout: inserted tensor slice
                        const std::vector<int> & offsets, //in: base offsets of the slice (0-based)
                        const int device_kind,            //in: execution device kind
                        const int device_id,              //in: execution device id
                        bool accumulative)                //in: accumulative VS overwrite (defaults to overwrite)
{
 int errc = TALSH_SUCCESS;
 int accum = NOPE; if(accumulative) accum = YEP;
 this->completeWriteTask();
 talsh_tens_t * dtens = this->getTalshTensorPtr();
 talsh_tens_t * ltens = slice.getTalshTensorPtr();
 if(task_handle != nullptr){ //asynchronous
  assert(task_handle->isEmpty());
  talsh_task_t * task_hl = task_handle->getTalshTaskPtr();
  //++left; ++right; ++(*this);
  errc = talshTensorInsert(dtens,ltens,offsets.data(),device_id,device_kind,COPY_MT,accum,task_hl);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::insertSlice): talshTensorInsert error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
  if(errc == TALSH_SUCCESS){
   pimpl_->write_task_ = task_handle;
  }else{
   task_handle->clean();
  }
 }else{ //synchronous
  errc = talshTensorInsert(dtens,ltens,offsets.data(),device_id,device_kind,COPY_MT,accum);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::insertSlice): talshTensorInsert error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
 }
 return errc;
}


/** Initializes TAL-SH runtime. **/
void initialize(std::size_t * host_buffer_size)
{
 int num_gpu, gpu_list[MAX_GPUS_PER_NODE];
 int errc = talshDeviceCount(DEV_NVIDIA_GPU,&num_gpu);
 assert(errc == TALSH_SUCCESS && num_gpu >= 0);
 if(num_gpu > 0){for(int i = 0; i < num_gpu; ++i) gpu_list[i]=i;};

 int host_arg_max;
 if(host_buffer_size == nullptr){
  std::size_t buf_size = DEFAULT_HOST_BUFFER_SIZE;
  errc = talshInit(&buf_size,&host_arg_max,num_gpu,gpu_list,0,NULL,0,NULL);
 }else{
  errc = talshInit(host_buffer_size,&host_arg_max,num_gpu,gpu_list,0,NULL,0,NULL);
 }
 if(errc != TALSH_SUCCESS) std::cout << "#ERROR(talshInit): TAL-SH initialization error " << errc << std::endl;
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


std::size_t getDeviceMaxTensorSize(const int device_kind, const int device_id)
{
 return talshDeviceTensorSize(device_id,device_kind);
}


std::size_t getDeviceMaxBufferSize(const int device_kind, const int device_id)
{
 return talshDeviceBufferSize(device_id,device_kind);
}


} //namespace talsh
