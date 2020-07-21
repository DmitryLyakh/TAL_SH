/* Timing services (threadsafe).
AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
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
along with ExaTensor. If not, see <http://www.gnu.org/licenses/>.*/

#include "timer.h"

#include <chrono>

double time_sys_sec()
{
 auto stamp = std::chrono::system_clock::now(); //current time point
 auto durat = std::chrono::duration<double>(stamp.time_since_epoch()); //duration (sec) since the begining of the clock
 return durat.count(); //number of seconds
}

double time_high_sec()
{
 auto stamp = std::chrono::high_resolution_clock::now(); //current time point
 auto durat = std::chrono::duration<double>(stamp.time_since_epoch()); //duration (sec) since the begining of the clock
 return durat.count(); //number of seconds
}
