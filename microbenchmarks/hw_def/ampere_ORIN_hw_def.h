#ifndef AMPERE_ORIN_HW_DEF_H
#define AMPERE_ORIN_HW_DEF_H

#include "../common/deviceQuery.h"

#define L1_SIZE (192 * 1024) // Max L1 size in bytes

#define CLK_FREQUENCY 1300 // frequency in MHz

#define WARP_SCHEDS_PER_SM 4

#define L2_SLICES_NUM 16

#define L2_BANK_WIDTH_in_BYTE 32

#endif