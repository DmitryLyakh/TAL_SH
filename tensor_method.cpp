/** TAL-SH: Tensor Method Interface
REVISION: 2019/02/26

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifdef EXATN_SERVICE

#include "tensor_method.hpp"

unsigned long long getDenseTensorVolume(const TensorDenseBlock & tensor)
{
 unsigned long long vol = 1;
 if(tensor.num_dims > 0){
  for(unsigned long long i = 0; i < tensor.num_dims; ++i) vol*=tensor.dims[i];
 }else if(tensor.num_dims < 0){
  vol = 0;
 }
 return vol;
}

#endif //EXATN_SERVICE
