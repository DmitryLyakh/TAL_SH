/** ExaTensor::TAL-SH: Device-unified user-level C++ API header.
REVISION: 2020/06/19

Copyright (C) 2014-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2020 Oak Ridge National Laboratory (UT-Battelle)

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

#include <cassert>

#include "mem_manager.h"  //TAL-SH memory manager
#include "talsh.h"        //TAL-SH C header
#include "talsh_task.hpp" //TAL-SH C++ task

namespace talsh{

//Constants:

static const std::size_t DEFAULT_HOST_BUFFER_SIZE = TALSH_NO_HOST_BUFFER; //small unused buffer will be allocated


//Tensor data kind (static type VS numeric data kind constant conversions):

const int REAL32 = R4;
const int REAL64 = R8;
const int COMPLEX32 = C4;
const int COMPLEX64 = C8;

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

  /** Full Ctor with user-defined initialization (TAL-SH provides tensor data storage) **/
 Tensor(const std::vector<std::size_t> & signature,         //tensor signature (identifier): signature[0:rank-1]
        const std::vector<int> & dims,                      //tensor dimension extents: dims[0:rank-1]
        int data_kind,                                      //tensor data kind
        talsh_tens_init_i init_func);                       //user-defined tensor initialization function

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

 /** Short Ctor with user-defined initialization (TAL-SH provides tensor data storage) **/
 Tensor(const std::vector<int> & dims,                      //tensor dimension extents: dims[0:rank-1]
        int data_kind,                                      //tensor data kind
        talsh_tens_init_i init_func);                       //user-defined tensor initialization function

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

 /** Returns the tensor element data type: {REAL32,REAL64,COMPLEX32,COMPLEX64}. **/
 int getElementType() const;

 /** Returns the tensor rank (order in math terms). **/
 int getRank() const;
 /** Returns the tensor order (rank in phys terms). **/
 int getOrder() const;

 /** Returns the tensor volume (number of elements). **/
 std::size_t getVolume() const;

 /** Returns tensor signature (base offset for each tensor dimension).
     The default tensor signature is all zero offsets. **/
 const std::vector<std::size_t> & getDimOffsets() const;

 /** Returns tensor dimension extents (and tensor order). **/
 const int * getDimExtents(unsigned int & num_dims) const; //num_dims returns by reference (ugly)

 /** Returns the extent of a specific tensor dimension. **/
 int getDimExtent(unsigned int dim) const;

 /** Reshapes the tensor to a different shape of the same volume. **/
 int reshape(const std::vector<int> & dims); //new tensor dimension extents: dims[0:rank-1])

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
           void * device_mem = nullptr,      //in: optional pointer to that device's client memory where the tensor data should go
           bool exclusive = false);          //in: if true, tensor images on all other devices will be discarded

 /** Returns TRUE if the tensor is ready (has been computed).
     If ready, synchronizes its presence on the given device. **/
 bool ready(int * status,                     //out: status of the current write operation
            const int device_kind = DEV_HOST, //in: device kind
            const int device_id = 0,          //in: specific device of the given kind which the synchronization is done for
            void * device_mem = nullptr);     //in: optional pointer to that device's client memory where the tensor data should go

 /** Performs tensor initialization to some scalar value.
     Returns an error code (0:success). **/
 template <typename T = double>
 int setValue(TensorTask * task_handle,                    //out: task handle associated with this operation or nullptr (synchronous)
              const int device_kind = DEV_HOST,            //in: execution device kind
              const int device_id = 0,                     //in: execution device id
              const T scalar_value = TensorData<T>::zero); //in: scalar value

 /** Computes the 1-norm of the tensor. **/
 int norm1(TensorTask * task_handle,                       //out: task handle associated with this operation or nullptr (synchronous)
           double * tens_norm1,                            //out: 1-norm of the tensor (sum of absolute values of tensor elements)
           const int device_kind = DEV_HOST,               //in: execution device kind
           const int device_id = 0);                       //in: execution device id

 /** Extracts a slice from a given position in the current tensor. **/
 int extractSlice(TensorTask * task_handle,                //out: task handle associated with this operation or nullptr (synchronous)
                  Tensor & slice,                          //inout: extracted tensor slice
                  const std::vector<int> & offsets,        //in: base offsets of the slice (0-based)
                  const int device_kind = DEV_HOST,        //in: execution device kind
                  const int device_id = 0,                 //in: execution device id
                  bool accumulative = false);              //in: accumulate versus overwrite the destination tensor

 /** Inserts a slice into a given position in the current tensor. **/
 int insertSlice(TensorTask * task_handle,                //out: task handle associated with this operation or nullptr (synchronous)
                 Tensor & slice,                          //inout: inserted tensor slice
                 const std::vector<int> & offsets,        //in: base offsets of the slice (0-based)
                 const int device_kind = DEV_HOST,        //in: execution device kind
                 const int device_id = 0,                 //in: execution device id
                 bool accumulative = false);              //in: accumulate versus overwrite the destination tensor

 /** Copies the body of another congruent tensor with an optional dimension permutation:
     this = left (permuted)
     Returns an error code (0:success). **/
 int copyBody(TensorTask * task_handle,               //out: task handle associated with this operation or nullptr (synchronous)
              const std::string & pattern,            //in: permutation pattern string
              Tensor & left,                          //in: left tensor (source)
              const int device_kind = DEV_HOST,       //in: execution device kind
              const int device_id = 0);               //in: execution device id

 /** Performs accumulation of a tensor into the current tensor:
     this += left * scalar_factor
     Returns an error code (0:success). **/
 template <typename T = double>
 int accumulate(TensorTask * task_handle,               //out: task handle associated with this operation or nullptr (synchronous)
                const std::string & pattern,            //in: accumulation pattern string
                Tensor & left,                          //in: left tensor
                const int device_kind = DEV_HOST,       //in: execution device kind
                const int device_id = 0,                //in: execution device id
                const T factor = TensorData<T>::unity); //in: scalar factor

 /** Performs a tensor contraction of two tensors and accumulates the result into the current tensor:
     this += left * right * scalar_factor
     Returns an error code (0:success). **/
 template <typename T = double>
 int contractAccumulate(TensorTask * task_handle,               //out: task handle associated with this operation or nullptr (synchronous)
                        const std::string & pattern,            //in: contraction pattern string
                        Tensor & left,                          //in: left tensor
                        Tensor & right,                         //in: right tensor
                        const int device_kind = DEV_HOST,       //in: execution device kind
                        const int device_id = 0,                //in: execution device id
                        const T factor = TensorData<T>::unity,  //in: scalar factor (alpha)
                        bool accumulative = true);              //in: accumulate versus overwrite the destination tensor

 /** Performs an extra large tensor contraction of two tensors and accumulates the result into the current tensor:
     this += left * right * scalar_factor
     Regardless of the chosen execution device, this operation is blocking and the result will be available on Host.
     By providing a specific device_kind with device_id=DEV_DEFAULT, all devices of the requested kind will be used.
     Returns an error code (0:success). **/
 template <typename T = double>
 int contractAccumulateXL(TensorTask * task_handle,               //out: task handle associated with this operation or nullptr (synchronous)
                          const std::string & pattern,            //in: contraction pattern string
                          Tensor & left,                          //in: left tensor
                          Tensor & right,                         //in: right tensor
                          const int device_kind = DEV_HOST,       //in: execution device kind
                          const int device_id = 0,                //in: execution device id
                          const T factor = TensorData<T>::unity,  //in: scalar factor (alpha)
                          bool accumulative = true);              //in: accumulate versus overwrite the destination tensor

 /** Performs a matrix multiplication on two tensors and accumulates the result into the current tensor.
     Returns an error code (0:success). **/
 template <typename T = double>
 int multiplyAccumulate(TensorTask * task_handle,               //out: task handle associated with this operation or nullptr (synchronous)
                        Tensor & left,                          //in: left tensor
                        Tensor & right,                         //in: right tensor
                        const int device_kind = DEV_HOST,       //in: execution device kind
                        const int device_id = 0,                //in: execution device id
                        const T factor = TensorData<T>::unity); //in: scalar factor (alpha)

 /** Arbitrary tensor decomposition via SVD. Returns an error code (0:success).
     Example of the decomposition of tensor D(a,b,c,d,e):
      D(a,b,c,d,e)=L(d,i,c,j)*M(i,j)*R(e,a,j,b,i)
     This is a hyper-contraction, but the corresponding decomposition pattern
     is still specified as a regular tensor contraction:
      D(a,b,c,d,e)=L(d,i,c,j)*R(e,a,j,b,i)
     The middle tensor factor is returned separately. **/
 int decomposeSVD(TensorTask * task_handle,         //out: task handle associated with this operation or nullptr (synchronous)
                  const std::string & pattern,      //in: decomposition pattern string (same as the tensor contraction pattern)
                  Tensor & left,                    //out: left tensor factor
                  Tensor & right,                   //out: right tensor factor
                  Tensor & middle,                  //out: middle tensor factor (may be empty on entrance)
                  const int device_kind = DEV_HOST, //in: execution device kind
                  const int device_id = 0);         //in: execution device id

 /** Arbitrary tensor decomposition via SVD with the middle tensor
     absorbed by the left tensor factor. Returns an error code (0:success).
     Example of the decomposition of tensor D(a,b,c,d,e):
      D(a,b,c,d,e)=L(d,i,c,j)*M(i,j)*R(e,a,j,b,i)
     This is a hyper-contraction, but the corresponding decomposition pattern
     is still specified as a regular tensor contraction:
      D(a,b,c,d,e)=L(d,i,c,j)*R(e,a,j,b,i)
     The middle tensor factor is absorbed into the left tensor factor. **/
 int decomposeSVDL(TensorTask * task_handle,         //out: task handle associated with this operation or nullptr (synchronous)
                   const std::string & pattern,      //in: decomposition pattern string (same as the tensor contraction pattern)
                   Tensor & left,                    //out: left tensor factor
                   Tensor & right,                   //out: right tensor factor
                   const int device_kind = DEV_HOST, //in: execution device kind
                   const int device_id = 0);         //in: execution device id

 /** Arbitrary tensor decomposition via SVD with the middle tensor
     absorbed by the right tensor factor. Returns an error code (0:success).
     Example of the decomposition of tensor D(a,b,c,d,e):
      D(a,b,c,d,e)=L(d,i,c,j)*M(i,j)*R(e,a,j,b,i)
     This is a hyper-contraction, but the corresponding decomposition pattern
     is still specified as a regular tensor contraction:
      D(a,b,c,d,e)=L(d,i,c,j)*R(e,a,j,b,i)
     The middle tensor factor is absorbed into the right tensor factor. **/
 int decomposeSVDR(TensorTask * task_handle,         //out: task handle associated with this operation or nullptr (synchronous)
                   const std::string & pattern,      //in: decomposition pattern string (same as the tensor contraction pattern)
                   Tensor & left,                    //out: left tensor factor
                   Tensor & right,                   //out: right tensor factor
                   const int device_kind = DEV_HOST, //in: execution device kind
                   const int device_id = 0);         //in: execution device id

 /** Arbitrary tensor decomposition via SVD with the middle tensor
     absorbed by both left and right tensor factors (as the square root of it).
     Returns an error code (0:success).
     Example of the decomposition of tensor D(a,b,c,d,e):
      D(a,b,c,d,e)=L(d,i,c,j)*M(i,j)*R(e,a,j,b,i)
     This is a hyper-contraction, but the corresponding decomposition pattern
     is still specified as a regular tensor contraction:
      D(a,b,c,d,e)=L(d,i,c,j)*R(e,a,j,b,i)
     The middle tensor factor is absorbed into both left and right tensor factors. **/
 int decomposeSVDLR(TensorTask * task_handle,         //out: task handle associated with this operation or nullptr (synchronous)
                    const std::string & pattern,      //in: decomposition pattern string (same as the tensor contraction pattern)
                    Tensor & left,                    //out: left tensor factor
                    Tensor & right,                   //out: right tensor factor
                    const int device_kind = DEV_HOST, //in: execution device kind
                    const int device_id = 0);         //in: execution device id

 /** Internal tensor orthogonalization via SVD and discarding the middle tensor.
     Returns an error code (0:success).
     Example of the decomposition of tensor D(a,b,c,d,e):
      D(a,b,c,d,e)=L(d,i,c)*M(i)*R(e,a,b,i)
     This is a hyper-contraction, but the corresponding decomposition pattern
     is still specified as a regular tensor contraction:
      D(a,b,c,d,e)=L(d,i,c)*R(e,a,b,i)
     The middle tensor factor is discarded, with tensor D redefined according
     to the above equation. The symbolic tensor decomposition (contraction)
     pattern must only have one contracted index, its dimension being equal
     to the minimum of the left and right uncontracted dimension volumes,
     e.g. in the shown case min(vol(d,c),vol(e,a,b)). **/
 int orthogonalizeSVD(TensorTask * task_handle,         //out: task handle associated with this operation or nullptr (synchronous)
                      const std::string & pattern,      //in: decomposition pattern string (same as the tensor contraction pattern)
                      const int device_kind = DEV_HOST, //in: execution device kind
                      const int device_id = 0);         //in: execution device id

 /** Internal tensor orthogonalization via the Modified Gram-Schmidt procedure.
     The set of tensor dimensions provided in the isometric dimension set argument
     will form the column space of the corresponding orthogonal matrix,
     which can either be square or tall rectangular. Thus, the cumulative
     volume of the complementary tensor dimensions must not exceed the
     cumulative volume of the isometric dimension set. **/
 int orthogonalizeMGS(TensorTask * task_handle,         //out: task handle associated with this operation or nullptr (synchronous)
                      const std::vector<unsigned int> & iso_dims, //in: isometric dimension set (cannot be empty)
                      const int device_kind = DEV_HOST, //in: execution device kind
                      const int device_id = 0);         //in: execution device id

 /** Prints the tensor info. **/
 void print() const;
 /** Prints the tensor info and elements greater or equal to "thresh". **/
 void print(double thresh) const;

 /** Resets the write task on the tensor. **/
 void resetWriteTask(TensorTask * task = nullptr);
 /** Returns a non-owning pointer to the write task, or nullptr. **/
 TensorTask * getWriteTask();

 /** Returns a pointer to the underlying C TAL-SH tensor implementation. **/
 talsh_tens_t * getTalshTensorPtr();

 friend int determineOptimalDevice(Tensor & tens0);
 friend int determineOptimalDevice(Tensor & tens0, Tensor & tens1);
 friend int determineOptimalDevice(Tensor & tens0, Tensor & tens1, Tensor & tens2);

private:

 //Private methods:
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

  Impl(const std::vector<std::size_t> & signature,         //tensor signature (identifier): signature[0:rank-1]
       const std::vector<int> & dims,                      //tensor dimension extents: dims[0:rank-1]
       int data_kind,                                      //tensor data kind
       talsh_tens_init_i init_func);                       //user-defined tensor initialization function

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
int initialize(std::size_t * host_buffer_size = nullptr); //in: desired host buffer size; out: actual host buffer size
int shutdown();

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

// Query device count of a given kind:
int getDeviceCount(int dev_kind); //in: device kind

// Max allocatable tensor size (bytes) in the device buffer per specified device:
std::size_t getDeviceMaxTensorSize(const int device_kind = DEV_HOST, //in: device kind
                                   const int device_id = 0);         //in: device id

// Max device memory buffer size (bytes) per specified device:
std::size_t getDeviceMaxBufferSize(const int device_kind = DEV_HOST, //in: device kind
                                   const int device_id = 0);         //in: device id

// Determine the optimal execution device for given tensors:
int determineOptimalDevice(Tensor & tens0);
int determineOptimalDevice(Tensor & tens0,
                           Tensor & tens1);
int determineOptimalDevice(Tensor & tens0,
                           Tensor & tens1,
                           Tensor & tens2);

// Enable fast math on a given device(s):
bool enableFastMath(int device_kind,              //in: device kind
                    int device_id = DEV_DEFAULT); //in: device id

// Memory management logging:
void startMemManagerLog();
void finishMemManagerLog();

// Basic tensor operation logging:
void startTensorOpLog();
void finishTensorOpLog();

// TAL-SH statistics:
void printStatistics();


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
 errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.begin(),talshFlatDevId(DEV_HOST,0),NULL,0,
                             NULL,realPart(init_val),imagPart(init_val));
 if(errc != TALSH_SUCCESS) std::cout << "#ERROR(talsh::Tensor::Tensor): talshTensorConstruct error " << errc << std::endl << std::flush;
 assert(errc == TALSH_SUCCESS);
 assert(signature.size() == dims.size());
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
 errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.data(),talshFlatDevId(DEV_HOST,0),NULL,0,
                             NULL,realPart(init_val),imagPart(init_val));
 if(errc != TALSH_SUCCESS) std::cout << "#ERROR(talsh::Tensor::Tensor): talshTensorConstruct error " << errc << std::endl << std::flush;
 assert(errc == TALSH_SUCCESS);
 assert(signature.size() == dims.size());
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
 errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.data(),talshFlatDevId(DEV_HOST,0),NULL,0);
 if(errc != TALSH_SUCCESS) std::cout << "#ERROR(talsh::Tensor::Tensor): talshTensorConstruct error " << errc << std::endl << std::flush;
 assert(errc == TALSH_SUCCESS);
 assert(signature.size() == dims.size());
 std::size_t vol = talshTensorVolume(&tensor_); assert(vol <= ext_data.size());
 errc = talshTensorImportData(&tensor_,TensorData<T>::kind,static_cast<const void*>(ext_data.data()));
 if(errc != TALSH_SUCCESS) std::cout << "#ERROR(talsh::Tensor::Tensor): talshTensorImportData error " << errc << std::endl;
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
  std::cout << "#FATAL: Initialization of tensors with external memory storage is not implemented in TAL-SH yet!" << std::endl; assert(false);
  errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.begin(),talshFlatDevId(DEV_HOST,0),(void*)ext_mem,-1,
                              NULL,realPart(*init_val),imagPart(*init_val));
 }
 if(errc != TALSH_SUCCESS) std::cout << "#ERROR(talsh::Tensor::Tensor): talshTensorConstruct error " << errc << std::endl << std::flush;
 assert(errc == TALSH_SUCCESS);
 assert(signature.size() == dims.size());
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
  std::cout << "#FATAL: Initialization of tensors with external memory storage is not implemented in TAL-SH yet!" << std::endl; assert(false);
  errc = talshTensorConstruct(&tensor_,TensorData<T>::kind,rank,dims.data(),talshFlatDevId(DEV_HOST,0),(void*)ext_mem,-1,
                              NULL,realPart(*init_val),imagPart(*init_val));
 }
 if(errc != TALSH_SUCCESS) std::cout << "#ERROR(talsh::Tensor::Tensor): talshTensorConstruct error " << errc << std::endl << std::flush;
 assert(errc == TALSH_SUCCESS);
 assert(signature.size() == dims.size());
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
 const void * body_ptr;
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
  bool task_empty = task_handle->isEmpty(); assert(task_empty);
  talsh_task_t * task_hl = task_handle->getTalshTaskPtr();
  errc = talshTensorInit(dtens,realPart(scalar_value),imagPart(scalar_value),device_id,device_kind,COPY_M,task_hl);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::setValue): talshTensorInit error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
  if(errc == TALSH_SUCCESS){
   task_handle->used_tensors_[0] = this;
   task_handle->num_tensors_ = 1;
   this->resetWriteTask(task_handle);
  }else{
   task_handle->clean();
  }
 }else{ //synchronous
  errc = talshTensorInit(dtens,realPart(scalar_value),imagPart(scalar_value),device_id,device_kind,COPY_M);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::setValue): talshTensorInit error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
 }
 return errc;
}


/** Performs accumulation of a tensor into the current tensor:
    this += left * scalar_factor **/
template <typename T>
int Tensor::accumulate(TensorTask * task_handle,    //out: task handle associated with this operation or nullptr (synchronous)
                       const std::string & pattern, //in: accumulation pattern string
                       Tensor & left,               //in: left tensor
                       const int device_kind,       //in: execution device kind
                       const int device_id,         //in: execution device id
                       const T factor)              //in: scalar factor
{
 int errc = TALSH_SUCCESS;
 this->completeWriteTask();
 left.completeWriteTask();
 const char * contr_ptrn = pattern.c_str();
 talsh_tens_t * dtens = this->getTalshTensorPtr();
 talsh_tens_t * ltens = left.getTalshTensorPtr();
 if(task_handle != nullptr){ //asynchronous
  bool task_empty = task_handle->isEmpty(); assert(task_empty);
  talsh_task_t * task_hl = task_handle->getTalshTaskPtr();
  //++left; ++(*this);
  errc = talshTensorAdd(contr_ptrn,dtens,ltens,realPart(factor),imagPart(factor),device_id,device_kind,COPY_MT,task_hl);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::accumulate): talshTensorAdd error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
  if(errc == TALSH_SUCCESS){
   task_handle->used_tensors_[0] = this;
   task_handle->used_tensors_[1] = &left;
   task_handle->num_tensors_ = 2;
   this->resetWriteTask(task_handle);
  }else{
   task_handle->clean();
  }
 }else{ //synchronous
  errc = talshTensorAdd(contr_ptrn,dtens,ltens,realPart(factor),imagPart(factor),device_id,device_kind,COPY_MT);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::accumulate): talshTensorAdd error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
 }
 return errc;
}


/** Performs a tensor contraction of two tensors and accumulates the result into the current tensor:
    this += left * right * scalar_factor **/
template <typename T>
int Tensor::contractAccumulate(TensorTask * task_handle,    //out: task handle associated with this operation or nullptr (synchronous)
                               const std::string & pattern, //in: contraction pattern string
                               Tensor & left,               //in: left tensor
                               Tensor & right,              //in: right tensor
                               const int device_kind,       //in: execution device kind
                               const int device_id,         //in: execution device id
                               const T factor,              //in: scalar factor (alpha)
                               bool accumulative)           //in: accumulate in (default) VS overwrite destination tensor
{
 int errc = TALSH_SUCCESS;
 this->completeWriteTask();
 left.completeWriteTask();
 right.completeWriteTask();
 int accum = YEP; if(!accumulative) accum = NOPE;
 const char * contr_ptrn = pattern.c_str();
 talsh_tens_t * dtens = this->getTalshTensorPtr();
 talsh_tens_t * ltens = left.getTalshTensorPtr();
 talsh_tens_t * rtens = right.getTalshTensorPtr();
 if(task_handle != nullptr){ //asynchronous
  bool task_empty = task_handle->isEmpty(); assert(task_empty);
  talsh_task_t * task_hl = task_handle->getTalshTaskPtr();
  //++left; ++right; ++(*this);
  errc = talshTensorContract(contr_ptrn,dtens,ltens,rtens,realPart(factor),imagPart(factor),device_id,device_kind,
                             COPY_MTT,accum,task_hl);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::contractAccumulate): talshTensorContract error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
  if(errc == TALSH_SUCCESS){
   task_handle->used_tensors_[0] = this;
   task_handle->used_tensors_[1] = &left;
   task_handle->used_tensors_[2] = &right;
   task_handle->num_tensors_ = 3;
   this->resetWriteTask(task_handle);
  }else{
   task_handle->clean();
  }
 }else{ //synchronous
  errc = talshTensorContract(contr_ptrn,dtens,ltens,rtens,realPart(factor),imagPart(factor),device_id,device_kind,
                             COPY_MTT,accum);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::contractAccumulate): talshTensorContract error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
 }
 return errc;
}


/** Performs an extra large tensor contraction of two tensors and accumulates the result into the current tensor:
    this += left * right * scalar_factor **/
template <typename T>
int Tensor::contractAccumulateXL(TensorTask * task_handle,    //out: task handle associated with this operation or nullptr (synchronous)
                                 const std::string & pattern, //in: contraction pattern string
                                 Tensor & left,               //in: left tensor
                                 Tensor & right,              //in: right tensor
                                 const int device_kind,       //in: execution device kind
                                 const int device_id,         //in: execution device id
                                 const T factor,              //in: scalar factor (alpha)
                                 bool accumulative)           //in: accumulate in (default) VS overwrite destination tensor
{
 int errc = TALSH_SUCCESS;
 this->completeWriteTask();
 left.completeWriteTask();
 right.completeWriteTask();
 int accum = YEP; if(!accumulative) accum = NOPE;
 const char * contr_ptrn = pattern.c_str();
 talsh_tens_t * dtens = this->getTalshTensorPtr();
 talsh_tens_t * ltens = left.getTalshTensorPtr();
 talsh_tens_t * rtens = right.getTalshTensorPtr();
 if(task_handle != nullptr){ //asynchronous
  bool task_empty = task_handle->isEmpty(); assert(task_empty);
  //++left; ++right; ++(*this);
  errc = talshTensorContractXL(contr_ptrn,dtens,ltens,rtens,realPart(factor),imagPart(factor),device_id,device_kind,accum);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::contractAccumulateXL): talshTensorContractXL error " << errc << std::endl; //debug
  assert(errc == TALSH_SUCCESS || errc == TRY_LATER || errc == DEVICE_UNABLE);
 }else{ //synchronous
  errc = talshTensorContractXL(contr_ptrn,dtens,ltens,rtens,realPart(factor),imagPart(factor),device_id,device_kind,accum);
  if(errc != TALSH_SUCCESS && errc != TRY_LATER && errc != DEVICE_UNABLE)
   std::cout << "#ERROR(talsh::Tensor::contractAccumulateXL): talshTensorContractXL error " << errc << std::endl; //debug
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
                               const T factor)           //in: scalar factor
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
