#include "encryption.hpp"

#include <sodium.h>
#include <stdexcept>
#include <string>

namespace
{
    constexpr char MAGIC[] = "VV01";
}

Encryption::Encryption()
{
    if (sodium_init() < 0)
    {
        throw std::runtime_error("Libsodium initialization failed");
    }
}

bool Encryption::encrypt(
    const std::string &plaintext,
    const std::string &password,
    std::string &encrypted_data)
{
    if (password.empty())
    {
        return false;
    }

    unsigned char salt[crypto_pwhash_SALTBYTES];
    unsigned char nonce[crypto_secretbox_NONCEBYTES];
    unsigned char key[crypto_secretbox_KEYBYTES];

    randombytes_buf(salt, sizeof salt);
    randombytes_buf(nonce, sizeof nonce);

    if (crypto_pwhash(
            key,
            sizeof key,
            password.data(),
            password.size(),
            salt,
            crypto_pwhash_OPSLIMIT_INTERACTIVE,
            crypto_pwhash_MEMLIMIT_INTERACTIVE,
            crypto_pwhash_ALG_ARGON2ID13) != 0)
    {
        sodium_memzero(key, sizeof key);
        return false;
    }

    std::string ciphertext(
        plaintext.size() + crypto_secretbox_MACBYTES, '\0');

    crypto_secretbox_easy(
        reinterpret_cast<unsigned char *>(ciphertext.data()),
        reinterpret_cast<const unsigned char *>(plaintext.data()),
        plaintext.size(),
        nonce,
        key);

    encrypted_data.assign(MAGIC, sizeof(MAGIC) - 1);
    encrypted_data.append(
        reinterpret_cast<const char *>(salt), sizeof salt);
    encrypted_data.append(
        reinterpret_cast<const char *>(nonce), sizeof nonce);
    encrypted_data.append(ciphertext);

    sodium_memzero(key, sizeof key);
    return true;
}

bool Encryption::decrypt(
    const std::string &encrypted_data,
    const std::string &password,
    std::string &plaintext)
{
    constexpr std::size_t HEADER_SIZE =
        sizeof(MAGIC) - 1 +
        crypto_pwhash_SALTBYTES +
        crypto_secretbox_NONCEBYTES;

    if (password.empty() ||
        encrypted_data.size() < HEADER_SIZE + crypto_secretbox_MACBYTES ||
        encrypted_data.compare(0, sizeof(MAGIC) - 1, MAGIC) != 0)
    {
        return false;
    }

    const auto *bytes =
        reinterpret_cast<const unsigned char *>(encrypted_data.data());

    const unsigned char *salt =
        bytes + sizeof(MAGIC) - 1;

    const unsigned char *nonce =
        salt + crypto_pwhash_SALTBYTES;

    const unsigned char *ciphertext =
        nonce + crypto_secretbox_NONCEBYTES;

    const std::size_t ciphertext_size =
        encrypted_data.size() - HEADER_SIZE;

    unsigned char key[crypto_secretbox_KEYBYTES];

    if (crypto_pwhash(
            key,
            sizeof key,
            password.data(),
            password.size(),
            salt,
            crypto_pwhash_OPSLIMIT_INTERACTIVE,
            crypto_pwhash_MEMLIMIT_INTERACTIVE,
            crypto_pwhash_ALG_ARGON2ID13) != 0)
    {
        sodium_memzero(key, sizeof key);
        return false;
    }

    std::string result(
        ciphertext_size - crypto_secretbox_MACBYTES, '\0');

    const int status = crypto_secretbox_open_easy(
        reinterpret_cast<unsigned char *>(result.data()),
        ciphertext,
        ciphertext_size,
        nonce,
        key);

    sodium_memzero(key, sizeof key);

    if (status != 0)
    {
        return false;
    }

    plaintext = std::move(result);
    return true;
}
