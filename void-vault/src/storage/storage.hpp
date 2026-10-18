
#ifndef VOID_VAULT_STORAGE_HPP
#define VOID_VAULT_STORAGE_HPP

#include <string>

class Storage {
public:
    explicit Storage(const std::string& file_path);

    bool save(const std::string& data);
    bool load(std::string& data);

private:
    std::string file_path;
};

#endif

