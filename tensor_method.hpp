/** TAL-SH: Tensor Method Interface
REVISION: 2019/02/14

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef TENSOR_METHOD_HPP_
#define TENSOR_METHOD_HPP_

#ifdef EXATN_SERVICE

#include "identifiable.h"

#include "tensor_algebra.h"

class TensorMethod: public Identifiable{
public:

 TensorMethod() = default;
 virtual ~TensorMethod() = default;

 virtual void apply(const talsh_tens_dense_t & local_tensor) = 0;

};

#endif //EXATN_SERVICE

#endif //TENSOR_METHOD_HPP_
