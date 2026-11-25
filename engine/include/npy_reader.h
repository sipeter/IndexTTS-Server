// npy_reader.h — Minimal .npy file reader for float32 2D arrays
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace indextts {

/// Read a .npy file containing a float32 2D array [rows, cols].
/// Returns flat row-major data.
struct NpyArray {
    std::vector<float> data;
    int rows = 0;
    int cols = 0;

    float* row_ptr(int r) { return data.data() + r * cols; }
    const float* row_ptr(int r) const { return data.data() + r * cols; }
};

inline NpyArray load_npy_float32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open npy file: " + path);
    }

    // Read magic: \x93NUMPY
    char magic[6];
    f.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
        throw std::runtime_error("Invalid npy magic: " + path);
    }

    // Version
    uint8_t major_ver, minor_ver;
    f.read(reinterpret_cast<char*>(&major_ver), 1);
    f.read(reinterpret_cast<char*>(&minor_ver), 1);

    // Header length
    uint32_t header_len = 0;
    if (major_ver == 1) {
        uint16_t hl;
        f.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }

    // Read header string
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    // Parse shape from header like "{'descr': '<f4', 'fortran_order': False, 'shape': (12001, 1280), }"
    // Find 'shape':
    auto shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) {
        throw std::runtime_error("No shape in npy header: " + path);
    }
    auto paren_start = header.find('(', shape_pos);
    auto paren_end = header.find(')', shape_pos);
    std::string shape_str = header.substr(paren_start + 1, paren_end - paren_start - 1);

    // Parse dimensions
    std::vector<int> shape;
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ','))
            ++pos;
        if (pos >= shape_str.size()) break;
        int val = std::stoi(shape_str.substr(pos));
        shape.push_back(val);
        while (pos < shape_str.size() && shape_str[pos] != ',' && shape_str[pos] != ')')
            ++pos;
    }

    // Check dtype is float32
    bool is_float32 = (header.find("<f4") != std::string::npos ||
                       header.find("float32") != std::string::npos);
    if (!is_float32) {
        throw std::runtime_error("npy file is not float32: " + path);
    }

    NpyArray arr;
    if (shape.size() == 2) {
        arr.rows = shape[0];
        arr.cols = shape[1];
    } else if (shape.size() == 1) {
        arr.rows = shape[0];
        arr.cols = 1;
    } else {
        throw std::runtime_error("Unsupported npy ndim=" + std::to_string(shape.size()) + ": " + path);
    }

    size_t total = static_cast<size_t>(arr.rows) * arr.cols;
    arr.data.resize(total);
    f.read(reinterpret_cast<char*>(arr.data.data()), total * sizeof(float));

    return arr;
}

} // namespace indextts
