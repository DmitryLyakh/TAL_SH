/** TAL-SH: Tensor Method Interface
REVISION: 2019/09/13

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef TENSOR_METHOD_HPP_
#define TENSOR_METHOD_HPP_

#ifdef EXATN_SERVICE

#include "byte_packet.h"

#include <vector>

//Dense locally stored tensor block for ExaTN:
struct TensorDenseBlock{
 std::vector<std::size_t> base;
 std::vector<std::size_t> dims;
 void * body;
 int data_kind;

 TensorDenseBlock(): body(nullptr), data_kind(NO_TYPE){}
 ~TensorDenseBlock(){if(body) free(body); body = nullptr;}

 inline std::size_t getVolume(){
  std::size_t vol = 1;
  for(unsigned int i = 0; i < dims.size(); ++i) vol *= dims[i];
  return vol;
 }
};

//External tensor method (identifiable):
template <typename IdentifiableConcept>
class TensorMethod: public IdentifiableConcept{
public:

 TensorMethod() = default;
 virtual ~TensorMethod() = default;

 //Packing/unpacking data members into/from a plain byte packet:
 virtual void pack(BytePacket & packet) = 0;
 virtual void unpack(BytePacket & packet) = 0;

 //Application-defined external tensor method:
 virtual int apply(TensorDenseBlock & local_tensor) = 0;

};

#endif //EXATN_SERVICE

#endif //TENSOR_METHOD_HPP_
