#ifndef NVTX_PROFILE_H_
#define NVTX_PROFILE_H_

#ifdef __cplusplus
extern "C" {
#endif
 void prof_push(const char * annotation, int color);
 void prof_pop();
#ifdef __cplusplus
}
#endif

#endif /*NVTX_PROFILE_H_*/
