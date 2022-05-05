/* Timing services (threadsafe).
AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
REVISION: 2020/07/21

Copyright (C) 2014-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2022 Oak Ridge National Laboratory (UT-Battelle)

LICENSE: BSD 3-Clause */

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
