
#ifndef VOID_VAULT_STORAGE_HPP
#define VOID_VAULT_STORAGE_HPP

#include <string>

class Storage {
public:
    explicit Storage(const std::string& file_path);

    // Atomically replaces the file: data is written to "<path>.tmp", flushed,
    // then renamed over the target. A failed save never truncates the
    // previously valid file.
    bool save(const std::string& data);

    // Reads the whole file. On failure the output is left untouched.
    bool load(std::string& data);

    bool exists() const;
    const std::string& path() const;

    // Shared atomic-write primitive (also used by Authorizer).
    // allow_replace == false: fails if the target already exists (no overwrite).
    static bool atomicWrite(const std::string& path,
                            const std::string& data,
                            bool allow_replace);

    // Reads a whole file into `out` only on success.
    static bool readFile(const std::string& path, std::string& out);

private:
    std::string file_path;
};

#endif

