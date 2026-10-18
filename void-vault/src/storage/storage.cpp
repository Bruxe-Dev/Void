#include "../include/storage.hpp"

#include <fstream>

Storage::Storage(const std::string &file_path)
    : file_path(file_path)
{
}

bool Storage::save(const std::string &data)
{
    std::ofstream file(file_path, std::ios::binary | std::ios::trunc);

    if (!file.is_open())
    {
        return false;
    }

    file.write(data.data(),
               static_cast<std::streamsize>(data.size()));

    return file.good();
}

bool Storage::load(std::string &data)
{
    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open())
    {
        return false;
    }

    data.assign(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());

    return file.good() || file.eof();
}
