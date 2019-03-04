/** TAL-SH: Byte packet
REVISION: 2019/02/26

Copyright (C) 2018-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2019 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef BYTE_PACKET_H_
#define BYTE_PACKET_H_

#include <assert.h>

#ifdef __cplusplus
#include <cstddef>
#endif

#define BYTE_PACKET_CAPACITY 1048576 //default byte packet capacity in bytes

//Byte packet (interoperable):
typedef struct{
 void * base_addr;              //base address (owning pointer)
 unsigned long long capacity;   //byte packet capacity in bytes
 unsigned long long size_bytes; //actual size of the byte packet in bytes
 unsigned long long position;   //current position inside the byte packet
} BytePacket;


void initBytePacket(BytePacket * packet);
void clearBytePacket(BytePacket * packet);
void resetBytePacket(BytePacket * packet);

#ifdef __cplusplus
template <typename T>
void appendToBytePacket(BytePacket * packet, const T & item)
{
 char * dst_ptr = &(((char*)(packet->base_addr))[packet->position]);
 char * src_ptr = ((char*)(&item));
 unsigned long long type_size = sizeof(T);
 assert(packet->position + type_size <= packet->capacity);
 for(unsigned long long i = 0; i < type_size; ++i) dst_ptr[i] = src_ptr[i];
 packet->position += type_size;
 if(packet->position > packet->size_bytes) packet->size_bytes = packet->position;
 return;
}

template <typename T>
void extractFromBytePacket(BytePacket * packet, T & item)
{
 char * src_ptr = &(((char*)(packet->base_addr))[packet->position]);
 char * dst_ptr = ((char*)(&item));
 unsigned long long type_size = sizeof(T);
 assert(packet->position + type_size <= packet->size_bytes);
 for(unsigned long long i = 0; i < type_size; ++i) dst_ptr[i] = src_ptr[i];
 packet->position += type_size;
 return;
}
#endif

#endif //BYTE_PACKET_H_
