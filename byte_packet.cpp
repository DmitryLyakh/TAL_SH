/** TAL-SH: Byte packet
REVISION: 2019/02/26

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#include "byte_packet.h"

#include <cstdlib>

void initBytePacket(BytePacket * packet)
{
 packet->capacity = 0;
 packet->base_addr = malloc(BYTE_PACKET_CAPACITY);
 if(packet->base_addr != NULL) packet->capacity = BYTE_PACKET_CAPACITY;
 packet->size_bytes = 0;
 packet->position = 0;
 return;
}

void clearBytePacket(BytePacket * packet)
{
 packet->capacity = 0;
 free(packet->base_addr);
 packet->base_addr = NULL;
 packet->size_bytes = 0;
 packet->position = 0;
 return;
}

void resetBytePacket(BytePacket * packet)
{
 packet->position = 0;
 return;
}
