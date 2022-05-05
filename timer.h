/* Timing services (threadsafe).
AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com
REVISION: 2018/12/06

Copyright (C) 2014-2022 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2022 Oak Ridge National Laboratory (UT-Battelle)

LICENSE: BSD 3-Clause */

#ifndef TIMER_H_
#define TIMER_H_

#ifdef __cplusplus
extern "C"{
#endif
 double time_sys_sec();  //system time stamp in seconds (thread-global)
 double time_high_sec(); //high-resolution time stamp in seconds
#ifdef __cplusplus
}
#endif

#endif //TIMER_H_
