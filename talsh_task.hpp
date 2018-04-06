/** ExaTensor::TAL-SH: C++ TAL-SH task
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

#ifndef _TALSH_TASK_HPP
#define _TALSH_TASK_HPP

#include <tuple>

#include "talsh.h" //TAL-SH C header

namespace talsh{

class Tensor;

/** Tensor task. **/
class TensorTask{

public:

 TensorTask();

 TensorTask(const TensorTask & task_handle) = delete;

 TensorTask & operator=(const TensorTask & task_handle) = delete;

 ~TensorTask();

 bool is_empty();

 void clean();

 bool wait();

 bool test(int * status);

private:

 talsh_task_t * get_talsh_task_ptr();

 talsh_task_t talsh_task_;                    //TAL-SH task handle
 unsigned int num_tensors_;                   //number of participating tensors
 Tensor * used_tensors_[MAX_TENSOR_OPERANDS]; //non-owning pointers to the tensors participating in this task

 friend class Tensor;

};

} //namespace talsh

#endif //_TALSH_TASK_HPP
