#ifndef NVTX_PROFILE_H_
#define NVTX_PROFILE_H_

extern "C" {
 void prof_push(const char * annotation, int color);
 void prof_pop();
}

#endif /*NVTX_PROFILE_H_*/
