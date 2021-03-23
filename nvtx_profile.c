/* https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx */

#include "nvtx_profile.h"

#ifndef NO_GPU

#ifndef USE_HIP

#include "nvToolsExt.h"

const uint32_t colors[] = {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff};
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors; \
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}

#define POP_RANGE nvtxRangePop();

#else

#define PUSH_RANGE(name,cid)
#define POP_RANGE

#endif

#else

#define PUSH_RANGE(name,cid)
#define POP_RANGE

#endif /*NO_GPU*/

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
