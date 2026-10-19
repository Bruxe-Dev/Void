#include <cstdio>
#include <iostream>
#include <string>

#include "storage.hpp"

int main()
{
    const std::string test_file = "storage_test_data.tmp";
    Storage storage(test_file);

    const std::string expected = "Void Vault storage test";
    std::string loaded;

    std::cout << "=== Storage Tests ===\n";

    if (!storage.save(expected))
    {
        std::cerr << "FAIL: Could not save data\n";
        return 1;
    }

    std::cout << "PASS: Save data\n";

    if (!storage.load(loaded))
    {
        std::cerr << "FAIL: Could not load data\n";
        return 1;
    }

    std::cout << "PASS: Load data\n";

    if (loaded != expected)
    {
        std::cerr << "FAIL: Loaded content differs\n";
        return 1;
    }

    std::cout << "PASS: Content verification\n";

    if (std::remove(test_file.c_str()) != 0)
    {
        std::cerr << "WARNING: Could not remove test file\n";
    }

    std::cout << "All storage tests passed.\n";
    return 0;
}
