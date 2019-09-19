/** TAL-SH: Tensor Method Interface (Tensor Functor)
REVISION: 2019/09/18

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef TENSOR_METHOD_HPP_
#define TENSOR_METHOD_HPP_

#ifdef EXATN_SERVICE

#include "byte_packet.h"

namespace talsh{

class Tensor;

//External tensor functor (identifiable):
template <typename IdentifiableConcept>
class TensorFunctor: public IdentifiableConcept{
public:

 TensorFunctor() = default;
 virtual ~TensorFunctor() = default;

 //Packing/unpacking data members into/from a plain byte packet:
 virtual void pack(BytePacket & packet) = 0;
 virtual void unpack(BytePacket & packet) = 0;

 //Application-defined external tensor method:
 virtual int apply(Tensor & local_tensor) = 0;

};

} //namespace talsh

#endif //EXATN_SERVICE

#endif //TENSOR_METHOD_HPP_
