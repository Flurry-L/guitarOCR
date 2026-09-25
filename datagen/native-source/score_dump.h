#pragma once

#include <string>

namespace gp { namespace core {
class Score;
namespace style { class Stylesheet; }
}}

namespace gpomr {

bool writeScoreDump(
    const gp::core::Score& score,
    const gp::core::style::Stylesheet& stylesheet,
    const std::string& outputPath,
    std::string& error,
    int selectedTrackIndex);

} // namespace gpomr
