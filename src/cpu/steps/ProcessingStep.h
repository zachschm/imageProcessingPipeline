#ifndef PROCESSINGSTEP_H
#define PROCESSINGSTEP_H
#include "Image.h"

class ProcessingStep
{
 public:
    virtual void process(Image& image) = 0;  // Pure virtual function
    virtual ~ProcessingStep() = default;
};

#endif