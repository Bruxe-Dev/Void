#include "../src/vault/vault.hpp"

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

void check(bool condition, const char *message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

int main()
{
    const std::string test_file = "vault_test_data.dat";
    const std::string password = "VaultTestPassword123!";
    const std::string original = "Confidential Void Vault test data";

    std::remove(test_file.c_str());

    Authorizer authorizer;
    Storage storage(test_file);
    Vault vault(authorizer, storage);

    check(
        authorizer.setPassword(password),
        "Password setup");

    check(!vault.isUnlocked(), "Vault starts locked");
    check(!vault.saveData(original), "Saving while locked is rejected");

    std::string loaded;
    check(!vault.loadData(loaded), "Loading while locked is rejected");

    check(vault.authenticate(password), "Correct password authenticates");
    check(vault.unlock(), "Authenticated vault unlocks");
    check(vault.isUnlocked(), "Vault reports unlocked");

    check(vault.saveData(original), "Saving encrypted data succeeds");

    std::ifstream file(test_file, std::ios::binary);
    check(file.is_open(), "Encrypted file exists and opens");

    const std::string disk_data{
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()};

    file.close();

    check(!disk_data.empty(), "Encrypted file is not empty");
    check(
        disk_data.find(original) == std::string::npos,
        "Plaintext is absent from encrypted file");

    check(vault.loadData(loaded), "Loading encrypted data succeeds");
    check(loaded == original, "Loaded data matches original");

    vault.lock();

    check(!vault.isUnlocked(), "Vault locks successfully");
    check(!vault.saveData(original), "Saving after locking is rejected");
    check(!vault.loadData(loaded), "Loading after locking is rejected");

    check(vault.authenticate(password), "Re-authentication succeeds");
    check(vault.unlock(), "Vault unlocks again");

    std::string corrupted = disk_data;
    corrupted.back() ^= 1;

    {
        std::ofstream output(
            test_file,
            std::ios::binary | std::ios::trunc);

        check(output.is_open(), "Corrupted file opens for writing");

        output.write(
            corrupted.data(),
            static_cast<std::streamsize>(corrupted.size()));

        check(output.good(), "Corrupted data is written");
    }

    const std::string previous_loaded = loaded;

    check(
        !vault.loadData(loaded),
        "Corrupted ciphertext is rejected");

    check(
        loaded == previous_loaded,
        "Failed decryption does not overwrite output data");

    vault.lock();
    std::remove(test_file.c_str());

    std::cout << "All vault integration tests passed.\n";
    return EXIT_SUCCESS;
}