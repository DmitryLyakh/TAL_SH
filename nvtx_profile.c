#include "nvtx_profile.h"

void prof_push(const char * annotation, int color)
{
 PUSH_RANGE(annotation,color)
 return;
}

void prof_pop()
{
 POP_RANGE
 return;
}
