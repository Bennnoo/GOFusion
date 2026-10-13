/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file source_file_pos.hpp
 * @brief This file contains some helper functions for the assert macros.
 * @author Paul Furgale
 * @author Stefan Leutenegger
 */

#ifndef OKVIS_SOURCE_FILE_POS_HPP
#define OKVIS_SOURCE_FILE_POS_HPP

#include <string>
#include <iostream>
#include <sstream>

namespace okvis {

class source_file_pos {
public:
  std::string function; 
  std::string file; 
  int line; 

  source_file_pos(std::string function, std::string file, int line) :
    function(function), file(file), line(line) {}

  operator std::string()
  {
    return toString();
  }

  std::string toString() const
  {
    std::stringstream s;
    s << file << ":" << line << ": " << function << "()";
    return s.str();
  }

};

}

inline std::ostream & operator<<(std::ostream & out, const okvis::source_file_pos & sfp)
{
  out << sfp.file << ":" << sfp.line << ": " << sfp.function << "()";
  return out;
}

#define OKVIS_SOURCE_FILE_POS okvis::source_file_pos(__FUNCTION__,__FILE__,__LINE__)

#endif 
