#ifndef HOG_DESCRIPTOR_CPU_H
#define HOG_DESCRIPTOR_CPU_H

#include <string>

void computeHOG_cpu(const std::string& inputImagePath, const std::string& outputHistogramPath);

#endif // HOG_DESCRIPTOR_CPU_H