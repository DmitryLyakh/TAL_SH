/** TAL-SH: Tensor Method Interface
REVISION: 2019/02/26

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef TENSOR_METHOD_HPP_
#define TENSOR_METHOD_HPP_

#ifdef EXATN_SERVICE

#include "byte_packet.h"

//Dense tensor block view (interoperable):
typedef struct{
 int num_dims;      //number of dimensions
 int data_kind;     //data kind
 void * body_ptr;   //non-owning pointer to the tensor data
 long long * bases; //non-owning pointer to dimension bases
 long long * dims;  //non-owning pointer to dimension extents
} TensorDenseBlock;

unsigned long long getDenseTensorVolume(const TensorDenseBlock &);


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
 virtual int apply(const TensorDenseBlock & local_tensor) = 0;

};

#endif //EXATN_SERVICE

#endif //TENSOR_METHOD_HPP_
