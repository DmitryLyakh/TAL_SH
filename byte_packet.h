/** TAL-SH: Byte packet
REVISION: 2020/06/30

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#ifndef BYTE_PACKET_H_
#define BYTE_PACKET_H_

#include <cstddef>
#include <cassert>

#define BYTE_PACKET_CAPACITY 1048576 //default byte packet capacity in bytes

//Byte packet (interoperable):
typedef struct{
 void * base_addr;              //base address (owning pointer)
 unsigned long long capacity;   //byte packet capacity in bytes
 unsigned long long size_bytes; //actual size of the byte packet in bytes
 unsigned long long position;   //current position inside the byte packet
} BytePacket;

/** Initializes the byte packet. **/
void initBytePacket(BytePacket * packet,
                    unsigned long long max_size = BYTE_PACKET_CAPACITY);

/** Destroys the byte packet. **/
void destroyBytePacket(BytePacket * packet);

/** Clears the byte packet (reusing the buffer). **/
void clearBytePacket(BytePacket * packet);

/** Resets the position inside the byte packet. **/
void resetBytePacket(BytePacket * packet,
                     unsigned long long new_position = 0);

/** Appends an arbitrary plain old data variable at the current packet
    position, shifting it forward by the size of the variable. **/
template <typename T>
void appendToBytePacket(BytePacket * packet, const T & item)
{
 char * dst_ptr = &(((char*)(packet->base_addr))[packet->position]);
 const char * src_ptr = ((const char *)(&item));
 unsigned long long type_size = sizeof(T);
 assert(packet->position + type_size <= packet->capacity);
 for(unsigned long long i = 0; i < type_size; ++i) dst_ptr[i] = src_ptr[i];
 packet->position += type_size;
 if(packet->position > packet->size_bytes) packet->size_bytes = packet->position;
 return;
}

/** Extracts an arbitrary plain old data variable at the current packet
    position, shifting it forward by the size of the variable. **/
template <typename T>
void extractFromBytePacket(BytePacket * packet, T & item)
{
 const char * src_ptr = &(((const char *)(packet->base_addr))[packet->position]);
 char * dst_ptr = ((char*)(&item));
 unsigned long long type_size = sizeof(T);
 assert(packet->position + type_size <= packet->size_bytes);
 for(unsigned long long i = 0; i < type_size; ++i) dst_ptr[i] = src_ptr[i];
 packet->position += type_size;
 return;
}

#endif //BYTE_PACKET_H_
