#include "l1_lat.h"

int main() {

  intilizeDeviceProp(0);

  float lat = l1_lat();

  std::cout << "l1 latency " << (unsigned)lat << std::endl;

  return 1;
}
