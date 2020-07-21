/** ExaTensor::TAL-SH: C++ TAL-SH task
REVISION: 2020/07/21

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

#ifndef TALSH_TASK_HPP_
#define TALSH_TASK_HPP_

#include "talsh.h" //TAL-SH C header

#include <tuple>

namespace talsh{

class Tensor;

/** TAL-SH task handle (tensor operation handle). **/
class TensorTask{

public:

 TensorTask();

 TensorTask(const TensorTask & task_handle) = delete;
 TensorTask & operator=(const TensorTask & task_handle) = delete;
 TensorTask(TensorTask && task_handle) = delete;
 TensorTask & operator=(TensorTask && task_handle) = delete;
 ~TensorTask();

 /** Returns TRUE if the TAL-SH task handle is empty. **/
 bool isEmpty();

 /** Resets the TAL-SH task handle to an empty state (completes it before, if neeed). **/
 void clean();

 /** Completes the TAL-SH task and returns the completion status. **/
 bool wait();

 /** Returns the current TAL-SH task status in <status>, together with TRUE if the TAL-SH task has completed. **/
 bool test(int * status);

 /** Returns the execution device kind and its specifc id.
     Negative id on return means device is undefined. **/
 int getExecutionDevice(int * device_kind);

 /** Returns the total number of tensor arguments used in the TAL-SH task. **/
 unsigned int getNumTensorArguments() const;

 /** Returns a specific tensor argument used in the TAL-SH task or nullptr if it does not exist. **/
 const Tensor * getTensorArgument(unsigned int arg_num) const;

 /** Returns all tensor arguments and their images used in the TAL-SH task or NULL if none. **/
 const talshTensArg_t * getTensorArgumentImages(int * num_arguments);

 /** Returns the tensor argument coherence control value used in the TAL-SH task.
     Negative value on return means undefined. **/
 int getTensorArgumentCoherence();

 /** Returns a pointer to the underlying C TAL-SH task implementation. **/
 talsh_task_t * getTalshTaskPtr();

private:

//Data members:
 talsh_task_t talsh_task_;                    //TAL-SH task handle
 unsigned int num_tensors_;                   //number of participating tensors in this TAL-SH task
 Tensor * used_tensors_[MAX_TENSOR_OPERANDS]; //non-owning pointers to the tensors participating in this task

//Friends:
 friend class Tensor;

};

} //namespace talsh

#endif //TALSH_TASK_HPP_
