/** ExaTensor::TAL-SH: C++ TAL-SH task
REVISION: 2019/02/13

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

#ifndef TALSH_TASK_HPP_
#define TALSH_TASK_HPP_

#include <tuple>

#include "talsh.h" //TAL-SH C header

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

private:

//Methods:
 talsh_task_t * getTalshTaskPtr();

//Data members:
 talsh_task_t talsh_task_;                    //TAL-SH task handle
 unsigned int num_tensors_;                   //number of participating tensors in this TAL-SH task
 Tensor * used_tensors_[MAX_TENSOR_OPERANDS]; //non-owning pointers to the tensors participating in this task

//Friends:
 friend class Tensor;

};

} //namespace talsh

#endif //TALSH_TASK_HPP_
