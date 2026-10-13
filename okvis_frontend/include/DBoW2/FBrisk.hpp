/**
 * @file FBrisk.hpp
 * @brief Header file for the Frontend class.
 *
 * License: BSD, see https://github.com/dorian3d/DBoW2/blob/master/LICENSE.txt
 * @author Dorian Galvez-Lopez
 * @author Stefan Leutenegger
 *
 */

#ifndef __D_T_F_BRISK__
#define __D_T_F_BRISK__

#include <vector>
#include <string>

#include <DBoW2/FClass.h>

namespace DBoW2 {

class FBrisk: protected FClass
{
public:

  virtual ~FBrisk() = default;

  typedef std::vector<unsigned char> TDescriptor;

  typedef const TDescriptor *pDescriptor;

  static const int L = 48;

  static void meanValue(const std::vector<pDescriptor> &descriptors,
    TDescriptor &mean);

  static double distance(const TDescriptor &a, const TDescriptor &b);

  static std::string toString(const TDescriptor &a);

  static void fromString(TDescriptor &a, const std::string &s);

  static void toMat32F(const std::vector<TDescriptor> &descriptors,
    cv::Mat &mat);

};

} 

#endif

