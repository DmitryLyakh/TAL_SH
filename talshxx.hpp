/** ExaTensor::TAL-SH: Device-unified user-level C++ API header.
REVISION: 2019/03/07

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

#ifndef TALSHXX_HPP_
#define TALSHXX_HPP_

#include <iostream>
#include <complex>
#include <memory>
#include <initializer_list>
#include <vector>
#include <string>

#include <assert.h>

#include "talsh.h"        //TAL-SH C header
#include "talsh_task.hpp" //TAL-SH C++ task

namespace talsh{

//Constants:

static const std::size_t DEFAULT_HOST_BUFFER_SIZE = TALSH_NO_HOST_BUFFER; //small unused buffer will be allocated


//Tensor data kind (static type VS numeric data kind constant conversions):

template <typename T>
struct TensorData{
 static constexpr int kind = NO_TYPE;
 static constexpr bool supported = false;
};

template <>
struct TensorData<float>{
 static constexpr int kind = R4;
 static constexpr bool supported = true;
 static constexpr float unity = 1.0f;
 static constexpr float zero = 0.0f;
};

template <>
struct TensorData<double>{
 static constexpr int kind = R8;
 static constexpr bool supported = true;
 static constexpr double unity = 1.0;
 static constexpr double zero = 0.0;
};

template <>
struct TensorData<std::complex<float>>{
 static constexpr int kind = C4;
 static constexpr bool supported = true;
 static constexpr std::complex<float> unity = {1.0f,0.0f};
 static constexpr std::complex<float> zero = {0.0f,0.0f};
};

template <>
struct TensorData<std::complex<double>>{
 static constexpr int kind = C8;
 static constexpr bool supported = true;
 static constexpr std::complex<double> unity = {1.0,0.0};
 static constexpr std::complex<double> zero = {0.0,0.0};
};

template <int talsh_data_kind> struct TensorDataType{using value = void;};
template <> struct TensorDataType<R4>{using value = float;};
template <> struct TensorDataType<R8>{using value = double;};
template <> struct TensorDataType<C4>{using value = std::complex<float>;};
template <> struct TensorDataType<C8>{using value = std::complex<double>;};


//Helper functions:

// Generic real/imaginary part extraction:
double realPart(float number);
double realPart(double number);
double realPart(std::complex<float> number);
double realPart(std::complex<double> number);
double imagPart(float number);
double imagPart(double number);
double imagPart(std::complex<float> number);
double imagPart(std::complex<double> number);


//Classes:

/** Dense local tensor **/
class Tensor{

public:

 /** Full Ctor with scalar initialization (TAL-SH provides tensor data storage) **/
 template <typename T>
 Tensor(const std::initializer_list<std::size_t> signature, //tensor signature (identifier): signature[0:rank-1]
        const std::initializer_list<int> dims,              //tensor dimension extents: dims[0:rank-1]
        const T init_val);                                  //scalar initialization value (its type will define tensor element data kind)
 /** Full Ctor with scalar initialization (TAL-SH provides tensor data storage) **/
 template <typename T>
 Tensor(const std::vector<std::size_t> & signature,         //tensor signature (identifier): signature[0:rank-1]
        const std::vector<int> & dims,                      //tensor dimension extents: dims[0:rank-1]
        const T init_val);                                  //scalar initialization value (its type will define tensor element data kind)

 /** Full Ctor with data import (TAL-SH provides tensor data storage) **/
 template <typename T>
 Tensor(const std::vector<std::size_t> & signature,         //tensor signature (identifier): signature[0:rank-1]
        const std::vector<int> & dims,                      //tensor dimension extents: dims[0:rank-1]
        const std::vector<T> & ext_data);                   //imported data (its type will define tensor element data kind)

 /** Full Ctor with scalar initialization (Application provides tensor data storage) **/
 template <typename T>
 Tensor(const std::initializer_list<std::size_t> signature, //tensor signature (identifier): signature[0:rank-1]
        const std::initializer_list<int> dims,              //tensor dimension extents: dims[0:rank-1]
        T * ext_mem,                                        //pointer to an external memory storage where the tensor body will reside
        const T * init_val = nullptr);                      //optional scalar initialization value (provide nullptr if not needed)
 /** Full Ctor with scalar initialization (Application provides tensor data storage) **/
 template <typename T>
 Tensor(const std::vector<std::size_t> & signature,         //tensor signature (identifier): signature[0:rank-1]
        const std::vector<int> & dims,                      //tensor dimension extents: dims[0:rank-1]
        T * ext_mem,                                        //pointer to an external memory storage where the tensor body will reside
        const T * init_val = nullptr);                      //optional scalar initialization value (provide nullptr if not needed)

 /** Short Ctor with scalar initialization (TAL-SH provides tensor data storage) **/
 template <typename T>
 Tensor(const std::vector<int> & dims,                      //tensor dimension extents: dims[0:rank-1]
        const T init_val);                                  //scalar initialization value (its type will define tensor element data kind)

 /** Short Ctor with data import (TAL-SH provides tensor data storage) **/
 template <typename T>
 Tensor(const std::vector<int> & dims,                      //tensor dimension extents: dims[0:rank-1]
        const std::vector<T> & ext_data);                   //imported data (its type will define tensor element data kind)

 /** Short Ctor with scalar initialization (Application provides tensor data storage) **/
 template <typename T>
 Tensor(const std::vector<int> & dims,                      //tensor dimension extents: dims[0:rank-1]
        T * ext_mem,                                        //pointer to an external memory storage where the tensor body will reside
        const T * init_val = nullptr);                      //optional scalar initialization value (provide nullptr if not needed)

 /** Copy ctor **/
 Tensor(const Tensor & tensor) = default;

 /** Copy assignment **/
 Tensor & operator=(const Tensor & tensor) = default;

 /** Move ctor **/
 Tensor(Tensor && tensor) = default;

 /** Move assignment **/
 Tensor & operator=(Tensor && tensor) = default;

 /** Dtor **/
 ~Tensor() = default;

 /** Returns the tensor rank (order in math terms). **/
 int getRank() const;
 /** Returns the tensor order (rank in phys terms). **/
 int getOrder() const;

 /** Returns the tensor volume (number of elements). **/
 std::size_t getVolume() const;

 /** Returns tensor dimension extents (and tensor order). **/
 const int * getDimExtents(unsigned int & num_dims) const;

 /** Returns a direct pointer to the tensor data available on Host.
     If no image is available on Host, returns false.  **/
 template<typename T>
 bool getDataAccessHost(T ** data_ptr);
 /** Returns a direct constant pointer to the tensor data available on Host.
     If no image is available on Host, returns false.  **/
 template<typename T>
 bool getDataAccessHostConst(const T ** data_ptr);

 /** Use count increment/decrement. **/
 Tensor & operator++(); //increments tensor use count
 Tensor & operator--(); //decrements tensor use count

 /** Synchronizes the tensor presence on the given device.
     Returns TRUE on success, FALSE if an active write task
     on this tensor has failed to complete successfully. **/
 bool sync(const int device_kind = DEV_HOST, //in: device kind
           const int device_id = 0,          //in: specific device of the given kind which the synchronization is done for
           void * device_mem = nullptr);     //in: optional pointer to that device's client memory where the tensor data should go

 /** Returns TRUE if the tensor is ready (has been computed).
     If ready, synchronizes its presence on the given device. **/
 bool ready(int * status,                     //out: status of the current write operation
            const int device_kind = DEV_HOST, //in: device kind
            const int device_id = 0,          //in: specific device of the given kind which the synchronization is done for
            void * device_mem = nullptr);     //in: optional pointer to that device's client memory where the tensor data should go

 /** Performs tensor initialization to some scalar value.
     Returns an error code (0:success). **/
 template <typename T>
 int setValue(TensorTask * task_handle,                    //out: task handle associated with this operation or nullptr (synchronous)
              const int device_kind = DEV_HOST,            //in: execution device kind
              const int device_id = 0,                     //in: execution device id
              const T scalar_value = TensorData<T>::zero); //in: scalar value

 /** Performs accumulation of a tensor into the current tensor:
     this += left * factor
     Returns an error code (0:success). **/
 template <typename T>
 int accumulate(TensorTask * task_handle,               //out: task handle associated with this operation or nullptr (synchronous)
                const std::string & pattern,            //in: accumulation pattern string
                Tensor & left,                          //in: left tensor
                const int device_kind = DEV_HOST,       //in: execution device kind
                const int device_id = 0,                //in: execution device id
                const T factor = TensorData<T>::unity); //in: alpha factor

 /** Performs a tensor contraction of two tensors and accumulates the result into the current tensor:
     this += left * right * factor
     Returns an error code (0:success). **/
 template <typename T>
 int contractAccumulate(TensorTask * task_handle,               //out: task handle associated with this operation or nullptr (synchronous)
                        const std::string & pattern,            //in: contraction pattern string
                        Tensor & left,                          //in: left tensor
                        Tensor & right,                         //in: right tensor
                        const int device_kind = DEV_HOST,       //in: execution device kind
                        const int device_id = 0,                //in: execution device id
                        const T factor = TensorData<T>::unity); //in: alpha factor

 /** Performs a matrix multiplication on two tensors and accumulates the result into the current tensor.
     Returns an error code (0:success). **/
 template <typename T>
 int multiplyAccumulate(TensorTask * task_handle,               //out: task handle associated with this operation or nullptr (synchronous)
                        Tensor & left,                          //in: left tensor
                        Tensor & right,                         //in: right tensor
                        const int device_kind = DEV_HOST,       //in: execution device kind
                        const int device_id = 0,                //in: execution device id
                        const T factor = TensorData<T>::unity); //in: alpha factor

 /** Prints the tensor. **/
 void print() const;

private:

 //Private methods:
 talsh_tens_t * getTalshTensorPtr();
 bool completeWriteTask();
 bool testWriteTask(int * status);

 //Implementation:
 struct Impl{

  std::vector<std::size_t> signature_; //tensor signature (unique integer multi-index identifier)
  talsh_tens_t tensor_;                //TAL-SH tensor block (dense locally stored tensor)
  TensorTask * write_task_;            //non-owning pointer to the task handle for the current asynchronous operation updating the tensor, if any
  void * host_mem_;                    //saved pointer to the original external Host memory buffer provided by the application during construction
  int used_;                           //number of unfinished (asynchronous) TAL-SH operations that are currently using the tensor

  template <typename T>
  Impl(const std::initializer_list<std::size_t> signature, //tensor signature (identifier): signature[0:rank-1]
       const std::initializer_list<int> dims,              //tensor dimension extents: dims[0:rank-1]
       const T init_val);                                  //scalar initialization value (its type will define tensor element data kind)

  template <typename T>
  Impl(const std::vector<std::size_t> & signature,         //tensor signature (identifier): signature[0:rank-1]
       const std::vector<int> & dims,                      //tensor dimension extents: dims[0:rank-1]
       const T init_val);                                  //scalar initialization value (its type will define tensor element data kind)

  template <typename T>
  Impl(const std::vector<std::size_t> & signature,         //tensor signature (identifier): signature[0:rank-1]
       const std::vector<int> & dims,                      //tensor dimension extents: dims[0:rank-1]
       const std::vector<T> & ext_data);                   //imported data (its type will define tensor element data kind)

  template <typename T>
  Impl(const std::initializer_list<std::size_t> signature, //tensor signature (identifier): signature[0:rank-1]
       const std::initializer_list<int> dims,              //tensor dimension extents: dims[0:rank-1]
       T * ext_mem,                                        //pointer to an external memory storage where the tensor body will reside
       const T * init_val = nullptr);                      //optional scalar initialization value (provide nullptr if not needed)

  template <typename T>
  Impl(const std::vector<std::size_t> & signature,         //tensor signature (identifier): signature[0:rank-1]
       const std::vector<int> & dims,                      //tensor dimension extents: dims[0:rank-1]
       T * ext_mem,                                        //pointer to an external memory storage where the tensor body will reside
       const T * init_val = nullptr);                      //optional scalar initialization value (provide nullptr if not needed)

  Impl(const Impl &) = delete;
  Impl & operator=(const Impl &) = delete;

  ~Impl();
 };

 //Data members:
 std::shared_ptr<Impl> pimpl_;
};


//Namespace API:

// TAL-SH initialization/shutdown:
void initialize(std::size_t * host_buffer_size = nullptr); //in: desired host buffer size; out: actual host buffer size
void shutdown();

// Host memory pinning/unpinning for accelerated computing:
template<typename T>
int pinHostMemory(T * host_ptr, std::size_t mem_size)
{
 return host_mem_register((void*)host_ptr,mem_size);
}

template<typename T>
int unpinHostMemory(T * host_ptr)
{
 return host_mem_unregister((void*)host_ptr);
}


//Template definitions:

template <typename T>
Tensor::Impl::Impl(const std::initializer_list<std::size_t> signature, //tensor signature (identifier): signature[0:rank-1]
                   const std::initializer_list<int> dims,              //tensor dimension extents: dims[0:rank-1]
                   const T init_val):                                  //scalar initialization value (its type will define tensor element data kind)
 signature_(signature), host_mem_(nullptr), used_(0)
{
 static_assert(TensorData<T>::supported,"Tensor data type is not supported!");
 int errc = talshTensorClean(&tensor_); assert(errc == TALSH_SUCCESS);
 const int rank = static_cast<int>(dims.size());
 errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.begin(),talshFlatDevId(DEV_HOST,0),NULL,-1,NULL,
                             realPart(init_val),imagPart(init_val));
 assert(errc == TALSH_SUCCESS && signature.size() == dims.size());
 write_task_ = nullptr;
}

template <typename T>
Tensor::Impl::Impl(const std::vector<std::size_t> & signature, //tensor signature (identifier): signature[0:rank-1]
                   const std::vector<int> & dims,              //tensor dimension extents: dims[0:rank-1]
                   const T init_val):                          //scalar initialization value (its type will define tensor element data kind)
 signature_(signature), host_mem_(nullptr), used_(0)
{
 static_assert(TensorData<T>::supported,"Tensor data type is not supported!");
 int errc = talshTensorClean(&tensor_); assert(errc == TALSH_SUCCESS);
 const int rank = static_cast<int>(dims.size());
 errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.data(),talshFlatDevId(DEV_HOST,0),NULL,-1,NULL,
                             realPart(init_val),imagPart(init_val));
 assert(errc == TALSH_SUCCESS && signature.size() == dims.size());
 write_task_ = nullptr;
}

template <typename T>
Tensor::Impl::Impl(const std::vector<std::size_t> & signature, //tensor signature (identifier): signature[0:rank-1]
                   const std::vector<int> & dims,              //tensor dimension extents: dims[0:rank-1]
                   const std::vector<T> & ext_data):           //imported data (its type will define tensor element data kind)
 signature_(signature), host_mem_(nullptr), used_(0)
{
 static_assert(TensorData<T>::supported,"Tensor data type is not supported!");
 int errc = talshTensorClean(&tensor_); assert(errc == TALSH_SUCCESS);
 const int rank = static_cast<int>(dims.size());
 errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.data(),talshFlatDevId(DEV_HOST,0),NULL);
 assert(errc == TALSH_SUCCESS && signature.size() == dims.size());
 errc = talshTensorImportData(&tensor_,TensorData<T>::kind,static_cast<const void*>(ext_data.data()));
 assert(errc == TALSH_SUCCESS);
 write_task_ = nullptr;
}

template <typename T>
Tensor::Impl::Impl(const std::initializer_list<std::size_t> signature, //tensor signature (identifier): signature[0:rank-1]
                   const std::initializer_list<int> dims,              //tensor dimension extents: dims[0:rank-1]
                   T * ext_mem,                                        //pointer to an external memory storage where the tensor body will reside
                   const T * init_val):                                //optional scalar initialization value (provide nullptr if not needed)
 signature_(signature), host_mem_(((void*)ext_mem)), used_(0)
{
 static_assert(TensorData<T>::supported,"Tensor data type is not supported!");
 int errc = talshTensorClean(&tensor_); assert(errc == TALSH_SUCCESS);
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

template <typename T>
Tensor::Impl::Impl(const std::vector<std::size_t> & signature, //tensor signature (identifier): signature[0:rank-1]
                   const std::vector<int> & dims,              //tensor dimension extents: dims[0:rank-1]
                   T * ext_mem,                                //pointer to an external memory storage where the tensor body will reside
                   const T * init_val):                        //optional scalar initialization value (provide nullptr if not needed)
 signature_(signature), host_mem_(((void*)ext_mem)), used_(0)
{
 static_assert(TensorData<T>::supported,"Tensor data type is not supported!");
 int errc = talshTensorClean(&tensor_); assert(errc == TALSH_SUCCESS);
 assert(ext_mem != nullptr);
 const int rank = static_cast<int>(dims.size());
 if(init_val == nullptr){
  errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.data(),talshFlatDevId(DEV_HOST,0),(void*)ext_mem);
 }else{
  std::cout << "FATAL: Initialization of tensors with external memory storage is not implemented in TAL-SH yet!" << std::endl; assert(false);
  errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.data(),talshFlatDevId(DEV_HOST,0),(void*)ext_mem,-1,NULL,
                              realPart(*init_val),imagPart(*init_val));
 }
 assert(errc == TALSH_SUCCESS && signature.size() == dims.size());
 write_task_ = nullptr;
}


template <typename T>
Tensor::Tensor(const std::initializer_list<std::size_t> signature, //tensor signature (identifier): signature[0:rank-1]
               const std::initializer_list<int> dims,              //tensor dimension extents: dims[0:rank-1]
               const T init_val):                                  //scalar initialization value (its type will define tensor element data kind)
 pimpl_(new Impl(signature,dims,init_val))
{
}

template <typename T>
Tensor::Tensor(const std::vector<std::size_t> & signature, //tensor signature (identifier): signature[0:rank-1]
               const std::vector<int> & dims,              //tensor dimension extents: dims[0:rank-1]
               const T init_val):                          //scalar initialization value (its type will define tensor element data kind)
 pimpl_(new Impl(signature,dims,init_val))
{
}

template <typename T>
Tensor::Tensor(const std::vector<std::size_t> & signature, //tensor signature (identifier): signature[0:rank-1]
               const std::vector<int> & dims,              //tensor dimension extents: dims[0:rank-1]
               const std::vector<T> & ext_data):           //imported data (its type will define tensor element data kind)
 pimpl_(new Impl(signature,dims,ext_data))
{
}

template <typename T>
Tensor::Tensor(const std::initializer_list<std::size_t> signature, //tensor signature (identifier): signature[0:rank-1]
               const std::initializer_list<int> dims,              //tensor dimension extents: dims[0:rank-1]
               T * ext_mem,                                        //pointer to an external memory storage where the tensor body will reside
               const T * init_val):                                //optional scalar initialization value (provide nullptr if not needed)
 pimpl_(new Impl(signature,dims,ext_mem,init_val))
{
}

template <typename T>
Tensor::Tensor(const std::vector<std::size_t> & signature, //tensor signature (identifier): signature[0:rank-1]
               const std::vector<int> & dims,              //tensor dimension extents: dims[0:rank-1]
               T * ext_mem,                                //pointer to an external memory storage where the tensor body will reside
               const T * init_val):                        //optional scalar initialization value (provide nullptr if not needed)
 pimpl_(new Impl(signature,dims,ext_mem,init_val))
{
}

template <typename T>
Tensor::Tensor(const std::vector<int> & dims,              //tensor dimension extents: dims[0:rank-1]
               const T init_val):                          //scalar initialization value (its type will define tensor element data kind)
 Tensor(std::vector<std::size_t>(dims.size(),0),dims,init_val)
{
}

template <typename T>
Tensor::Tensor(const std::vector<int> & dims,              //tensor dimension extents: dims[0:rank-1]
               const std::vector<T> & ext_data):           //imported data (its type will define tensor element data kind)
 Tensor(std::vector<std::size_t>(dims.size(),0),dims,ext_data)
{
}

template <typename T>
Tensor::Tensor(const std::vector<int> & dims,              //tensor dimension extents: dims[0:rank-1]
               T * ext_mem,                                //pointer to an external memory storage where the tensor body will reside
               const T * init_val):                        //optional scalar initialization value (provide nullptr if not needed)
 Tensor(std::vector<std::size_t>(dims.size(),0),dims,ext_mem,init_val)
{
}


/** Returns a direct pointer to the tensor data available on Host.
    If no image is available on Host, returns false.  **/
template<typename T>
bool Tensor::getDataAccessHost(T ** data_ptr)
{
 this->completeWriteTask();
 int data_kind = TensorData<T>::kind;
 talsh_tens_t * dtens = this->getTalshTensorPtr();
 assert(dtens != nullptr);
 void * body_ptr;
 int errc = talshTensorGetBodyAccess(dtens,&body_ptr,data_kind,0,DEV_HOST);
 if(errc == TALSH_SUCCESS){
  *data_ptr = static_cast<T*>(body_ptr);
 }else{
  *data_ptr = nullptr;
  return false;
 }
 return true;
}

/** Returns a direct constant pointer to the tensor data available on Host.
    If no image is available on Host, returns false.  **/
template<typename T>
bool Tensor::getDataAccessHostConst(const T ** data_ptr)
{
 this->completeWriteTask();
 int data_kind = TensorData<T>::kind;
 const talsh_tens_t * dtens = this->getTalshTensorPtr();
 assert(dtens != nullptr);
 void * body_ptr;
 int errc = talshTensorGetBodyAccessConst(dtens,&body_ptr,data_kind,0,DEV_HOST);
 if(errc == TALSH_SUCCESS){
  *data_ptr = static_cast<const T*>(body_ptr);
 }else{
  *data_ptr = nullptr;
  return false;
 }
 return true;
}


/** Performs tensor initialization to some scalar value. **/
template <typename T>
int Tensor::setValue(TensorTask * task_handle, //out: task handle associated with this operation or nullptr (synchronous)
                     const int device_kind,    //in: execution device kind
                     const int device_id,      //in: execution device id
                     const T scalar_value)     //in: scalar value
{
 int errc = TALSH_SUCCESS;
 this->completeWriteTask();
 talsh_tens_t * dtens = this->getTalshTensorPtr();
 if(task_handle != nullptr){ //asynchronous
  assert(task_handle->isEmpty());
  talsh_task_t * task_hl = task_handle->getTalshTaskPtr();
  errc = talshTensorInit(dtens,realPart(scalar_value),imagPart(scalar_value),device_id,device_kind,COPY_T,task_hl);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::setValue): talshTensorInit error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
  if(errc == TALSH_SUCCESS) pimpl_->write_task_ = task_handle;
 }else{ //synchronous
  errc = talshTensorInit(dtens,realPart(scalar_value),imagPart(scalar_value),device_id,device_kind,COPY_T);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::setValue): talshTensorInit error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
 }
 return errc;
}


/** Performs accumulation of a tensor into the current tensor:
    this += left * factor **/
template <typename T>
int Tensor::accumulate(TensorTask * task_handle,    //out: task handle associated with this operation or nullptr (synchronous)
                       const std::string & pattern, //in: accumulation pattern string
                       Tensor & left,               //in: left tensor
                       const int device_kind,       //in: execution device kind
                       const int device_id,         //in: execution device id
                       const T factor)              //in: alpha factor
{
 int errc = TALSH_SUCCESS;
 this->completeWriteTask();
 const char * contr_ptrn = pattern.c_str();
 talsh_tens_t * dtens = this->getTalshTensorPtr();
 talsh_tens_t * ltens = left.getTalshTensorPtr();
 if(task_handle != nullptr){ //asynchronous
  assert(task_handle->isEmpty());
  talsh_task_t * task_hl = task_handle->getTalshTaskPtr();
  //++left; ++right; ++(*this);
  errc = talshTensorAdd(contr_ptrn,dtens,ltens,realPart(factor),imagPart(factor),device_id,device_kind,COPY_TT,task_hl);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::accumulate): talshTensorAdd error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
  if(errc == TALSH_SUCCESS) pimpl_->write_task_ = task_handle;
 }else{ //synchronous
  errc = talshTensorAdd(contr_ptrn,dtens,ltens,realPart(factor),imagPart(factor),device_id,device_kind,COPY_TT);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::accumulate): talshTensorAdd error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
 }
 return errc;
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
 this->completeWriteTask();
 const char * contr_ptrn = pattern.c_str();
 talsh_tens_t * dtens = this->getTalshTensorPtr();
 talsh_tens_t * ltens = left.getTalshTensorPtr();
 talsh_tens_t * rtens = right.getTalshTensorPtr();
 if(task_handle != nullptr){ //asynchronous
  assert(task_handle->isEmpty());
  talsh_task_t * task_hl = task_handle->getTalshTaskPtr();
  //++left; ++right; ++(*this);
  errc = talshTensorContract(contr_ptrn,dtens,ltens,rtens,realPart(factor),imagPart(factor),device_id,device_kind,COPY_TTT,task_hl);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::contractAccumulate): talshTensorContract error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
  if(errc == TALSH_SUCCESS) pimpl_->write_task_ = task_handle;
 }else{ //synchronous
  errc = talshTensorContract(contr_ptrn,dtens,ltens,rtens,realPart(factor),imagPart(factor),device_id,device_kind,COPY_TTT);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::contractAccumulate): talshTensorContract error " << errc << std::endl; //debug
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
 int conj_bits = 0;
 get_contr_pattern_sym(&lrank,&rrank,&conj_bits,dptrn,cptrn,&cpl,&errc); cptrn[cpl]='\0';
 assert(errc == 0);
 std::string contr_ptrn(cptrn);
 std::cout << contr_ptrn << std::endl; //debug
 //Execute tensor contraction:
 errc = this->contractAccumulate(task_handle,contr_ptrn,left,right,device_kind,device_id,factor);
 return errc;
}


} //namespace talsh

#endif //TALSHXX_HPP_
