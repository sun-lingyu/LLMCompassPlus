#include "../l1_lat/l1_lat.h"
#include "l2_lat.h"

int main() {

  intilizeDeviceProp(0);

  float lat2 = l2_hit_lat();

  float lat1 = l1_lat();

  std::cout << "l2 latency " << (unsigned)(lat2 - lat1)
            << std::endl;

  return 1;
}
