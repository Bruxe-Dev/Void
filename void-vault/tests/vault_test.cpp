#include "../src/vault/vault.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

int main()
{
    const std::string test_file = "vault_test_data.dat";
    const std::string password = "VaultTestPassword123!";
    const std::string original = "Confidential Void Vault test data";

    std::remove(test_file.c_str());

    Authorizer authorizer;
    Storage storage(test_file);
    Vault vault(authorizer, storage);

    assert(authorizer.setPassword(password));

    assert(!vault.isUnlocked());
    assert(!vault.saveData(original));

    std::string loaded;
    assert(!vault.loadData(loaded));

    assert(vault.authenticate(password));
    assert(vault.unlock());
    assert(vault.isUnlocked());

    assert(vault.saveData(original));

    std::ifstream file(test_file, std::ios::binary);
    assert(file.is_open());

    std::string disk_data{
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()};

    file.close();

    assert(!disk_data.empty());
    assert(disk_data.find(original) == std::string::npos);

    assert(vault.loadData(loaded));
    assert(loaded == original);

    vault.lock();

    assert(!vault.isUnlocked());
    assert(!vault.saveData(original));
    assert(!vault.loadData(loaded));

    assert(vault.authenticate(password));
    assert(vault.unlock());

    std::string corrupted = disk_data;
    corrupted.back() ^= 1;

    {
        std::ofstream output(
            test_file,
            std::ios::binary | std::ios::trunc);

        assert(output.is_open());
        output.write(
            corrupted.data(),
            static_cast<std::streamsize>(corrupted.size()));

        assert(output.good());
    }

    assert(!vault.loadData(loaded));

    // 9. Clean up
    vault.lock();
    std::remove(test_file.c_str());

    std::cout << "All vault integration tests passed.\n";
    return 0;
}
