#include "encryption.hpp"

#include <sodium.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace
{
    constexpr char MAGIC[] = "VV01";
    constexpr std::size_t MAGIC_SIZE = sizeof(MAGIC) - 1;
    constexpr std::size_t HEADER_SIZE =
        MAGIC_SIZE + crypto_pwhash_SALTBYTES + crypto_secretbox_NONCEBYTES;

    bool deriveKey(unsigned char *key, std::string_view password, const unsigned char *salt)
    {
        return crypto_pwhash(
                   key,
                   crypto_secretbox_KEYBYTES,
                   password.data(),
                   password.size(),
                   salt,
                   crypto_pwhash_OPSLIMIT_INTERACTIVE,
                   crypto_pwhash_MEMLIMIT_INTERACTIVE,
                   crypto_pwhash_ALG_ARGON2ID13) == 0;
    }
}

Encryption::Encryption()
{
    if (sodium_init() < 0)
    {
        throw std::runtime_error("Libsodium initialization failed");
    }
}

bool Encryption::encrypt(
    std::string_view plaintext,
    std::string_view password,
    std::string &encrypted_data)
{
    if (password.empty() ||
        plaintext.size() > crypto_secretbox_MESSAGEBYTES_MAX - crypto_secretbox_MACBYTES)
    {
        return false;
    }

    unsigned char salt[crypto_pwhash_SALTBYTES];
    unsigned char nonce[crypto_secretbox_NONCEBYTES];
    unsigned char key[crypto_secretbox_KEYBYTES];

    randombytes_buf(salt, sizeof salt);
    randombytes_buf(nonce, sizeof nonce);

    if (!deriveKey(key, password, salt))
    {
        sodium_memzero(key, sizeof key);
        return false;
    }

    std::string record(HEADER_SIZE + crypto_secretbox_MACBYTES + plaintext.size(), '\0');
    auto *out = reinterpret_cast<unsigned char *>(record.data());
    std::copy(MAGIC, MAGIC + MAGIC_SIZE, record.begin());
    std::copy(salt, salt + sizeof salt, out + MAGIC_SIZE);
    std::copy(nonce, nonce + sizeof nonce, out + MAGIC_SIZE + sizeof salt);

    const int status = crypto_secretbox_easy(
        out + HEADER_SIZE,
        reinterpret_cast<const unsigned char *>(plaintext.data()),
        plaintext.size(),
        nonce,
        key);

    sodium_memzero(key, sizeof key);

    if (status != 0)
    {
        sodium_memzero(record.data(), record.size());
        return false;
    }

    encrypted_data = std::move(record);
    return true;
}

bool Encryption::decrypt(
    std::string_view encrypted_data,
    std::string_view password,
    std::string &plaintext)
{
    if (password.empty() ||
        encrypted_data.size() < HEADER_SIZE + crypto_secretbox_MACBYTES ||
        encrypted_data.compare(0, MAGIC_SIZE, MAGIC) != 0)
    {
        return false;
    }

    const auto *bytes =
        reinterpret_cast<const unsigned char *>(encrypted_data.data());
    const unsigned char *salt = bytes + MAGIC_SIZE;
    const unsigned char *nonce = salt + crypto_pwhash_SALTBYTES;
    const unsigned char *ciphertext = nonce + crypto_secretbox_NONCEBYTES;
    const std::size_t ciphertext_size = encrypted_data.size() - HEADER_SIZE;

    unsigned char key[crypto_secretbox_KEYBYTES];

    if (!deriveKey(key, password, salt))
    {
        sodium_memzero(key, sizeof key);
        return false;
    }

    std::string result(ciphertext_size - crypto_secretbox_MACBYTES, '\0');

    const int status = crypto_secretbox_open_easy(
        reinterpret_cast<unsigned char *>(result.data()),
        ciphertext,
        ciphertext_size,
        nonce,
        key);

    sodium_memzero(key, sizeof key);

    if (status != 0)
    {
        sodium_memzero(result.data(), result.size());
        return false;
    }

    // Wipe the previous plaintext held by the caller before replacing it.
    if (!plaintext.empty())
    {
        sodium_memzero(plaintext.data(), plaintext.size());
    }
    plaintext = std::move(result);
    return true;
}
