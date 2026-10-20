#include "../../include/storage.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <system_error>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace
{
#ifdef _WIN32
    std::wstring widen(const std::string &s)
    {
        return std::filesystem::path(s).wstring();
    }
#endif

    bool writeTemp(const std::string &tmp, const std::string &data)
    {
#ifdef _WIN32
        std::ofstream file(std::filesystem::path(tmp),
                           std::ios::binary | std::ios::trunc);
        if (!file.is_open())
        {
            return false;
        }
        file.write(data.data(), static_cast<std::streamsize>(data.size()));
        file.flush();
        const bool ok = file.good();
        file.close();
        return ok && !file.fail();
#else
        const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd < 0)
        {
            return false;
        }
        std::size_t written = 0;
        while (written < data.size())
        {
            const ssize_t n = ::write(fd, data.data() + written, data.size() - written);
            if (n <= 0)
            {
                ::close(fd);
                return false;
            }
            written += static_cast<std::size_t>(n);
        }
        const bool synced = ::fsync(fd) == 0;
        return (::close(fd) == 0) && synced;
#endif
    }

    bool commit(const std::string &tmp, const std::string &target, bool allow_replace)
    {
#ifdef _WIN32
        DWORD flags = MOVEFILE_WRITE_THROUGH;
        if (allow_replace)
        {
            flags |= MOVEFILE_REPLACE_EXISTING;
        }
        return MoveFileExW(widen(tmp).c_str(), widen(target).c_str(), flags) != 0;
#else
        if (allow_replace)
        {
            return ::rename(tmp.c_str(), target.c_str()) == 0;
        }
        // link() fails with EEXIST if the target exists: no overwrite.
        if (::link(tmp.c_str(), target.c_str()) != 0)
        {
            return false;
        }
        ::unlink(tmp.c_str());
        return true;
#endif
    }
}

Storage::Storage(const std::string &path)
    : file_path(path)
{
}

const std::string &Storage::path() const
{
    return file_path;
}

bool Storage::exists() const
{
    std::error_code ec;
    return std::filesystem::exists(std::filesystem::path(file_path), ec) && !ec;
}

bool Storage::atomicWrite(const std::string &path, const std::string &data, bool allow_replace)
{
    if (path.empty())
    {
        return false;
    }
    if (!allow_replace)
    {
        std::error_code ec;
        if (std::filesystem::exists(std::filesystem::path(path), ec) || ec)
        {
            return false;
        }
    }

    const std::string tmp = path + ".tmp";
    if (!writeTemp(tmp, data) || !commit(tmp, path, allow_replace))
    {
        std::error_code ec;
        std::filesystem::remove(std::filesystem::path(tmp), ec);
        return false;
    }
    return true;
}

bool Storage::readFile(const std::string &path, std::string &out)
{
    std::ifstream file(std::filesystem::path(path), std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }
    std::string buffer{std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>()};
    if (file.bad())
    {
        return false;
    }
    out = std::move(buffer);
    return true;
}

bool Storage::save(const std::string &data)
{
    return atomicWrite(file_path, data, true);
}

bool Storage::load(std::string &data)
{
    return readFile(file_path, data);
}
