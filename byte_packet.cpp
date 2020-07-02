/** TAL-SH: Byte packet
REVISION: 2020/06/30

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "byte_packet.h"

#include <cstdlib>

void initBytePacket(BytePacket * packet,
                    unsigned long long max_size)
{
 packet->capacity = 0;
 packet->base_addr = malloc(max_size);
 if(packet->base_addr != NULL) packet->capacity = max_size;
 packet->size_bytes = 0;
 packet->position = 0;
 return;
}

void destroyBytePacket(BytePacket * packet)
{
 packet->capacity = 0;
 packet->size_bytes = 0;
 packet->position = 0;
 free(packet->base_addr);
 packet->base_addr = NULL;
 return;
}

void clearBytePacket(BytePacket * packet)
{
 packet->size_bytes = 0;
 packet->position = 0;
 return;
}

void resetBytePacket(BytePacket * packet,
                     unsigned long long new_position)
{
 assert(new_position <= packet->size_bytes);
 packet->position = new_position;
 return;
}
