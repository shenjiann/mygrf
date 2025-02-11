/*-------------------------------------------------------------------------------
  Copyright (c) 2024 GRF Contributors.

  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#ifndef GRF_PREDICTIONVALUES_H
#define GRF_PREDICTIONVALUES_H

#include <cstddef>
#include <vector>

namespace mygrf {

class PredictionValues {
public:
  PredictionValues();

  PredictionValues(const std::vector<std::vector<double>>& values,
                   size_t num_types);


  double get(size_t node, size_t type) const;
  const std::vector<double>& get_values(size_t node) const;
  bool empty(size_t node) const;

  /**
   *  Returns all prediction values in this object. Values are
   *  organized first by node, then by type.
   */
  const std::vector<std::vector<double>>& get_all_values() const;
  const size_t get_num_nodes() const;
  const size_t get_num_types() const;

private:
  std::vector<std::vector<double>> values;
  size_t num_nodes;
  size_t num_types;
};

} // namespace grf

#endif //GRF_PREDICTIONVALUES_H
