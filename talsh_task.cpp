/** ExaTensor::TAL-SH: Device-unified user-level C++ API implementation.
REVISION: 2018/04/04

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

#include <assert.h>

#include "talsh_task.hpp"

namespace talsh{


TensorTask::TensorTask():
 num_tensors_(0)
{
 int errc = talshTaskClean(&talsh_task_);
 assert(errc == TALSH_SUCCESS);
}


TensorTask::~TensorTask()
{
 assert(this->wait());
 int errc = talshTaskDestruct(&talsh_task_);
 assert(errc == TALSH_SUCCESS);
}


bool TensorTask::is_empty()
{
 return (talshTaskIsEmpty(&talsh_task_) == YEP);
}

void TensorTask::clean()
{
 assert(this->wait());
 int errc = talshTaskClean(&talsh_task_);
 assert(errc == TALSH_SUCCESS);
 return;
}


bool TensorTask::wait()
{
 int stats = TALSH_TASK_COMPLETED;
 if(talshTaskIsEmpty(&talsh_task_) != YEP){
  int errc;
  if(talshTaskComplete(&talsh_task_,&stats,&errc) != YEP){
   errc = talshTaskWait(&talsh_task_,&stats);
   assert(errc == TALSH_SUCCESS);
  }
 }
 return (stats == TALSH_TASK_COMPLETED);
}


bool TensorTask::test(int * status)
{
 bool res = true;
 if(talshTaskIsEmpty(&talsh_task_) != YEP){
  int errc;
  res = (talshTaskComplete(&talsh_task_,status,&errc) == YEP);
 }else{ //empty task: Completed = TRUE
  *status = TALSH_TASK_EMPTY;
 }
 return res;
}


talsh_task_t * TensorTask::get_talsh_task_ptr()
{
 return &talsh_task_;
}


} //namespace talsh
