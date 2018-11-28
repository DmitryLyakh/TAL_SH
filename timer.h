/* Timing services (threadsafe).
AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
REVISION: 2018/08/07

Copyright (C) 2014-2016 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2016 Oak Ridge National Laboratory (UT-Battelle)

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

#ifndef _TIMER_H
#define _TIMER_H

#ifdef __cplusplus
extern "C"{
#endif
 double time_sys_sec();  //system time stamp in seconds (thread-global)
 double time_high_sec(); //high-resolution time stamp in seconds
#ifdef __cplusplus
}
#endif

#endif //_TIMER_H
