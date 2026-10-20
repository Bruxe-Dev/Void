#ifndef VOID_VAULT_ENCRYPTION_HPP
#define VOID_VAULT_ENCRYPTION_HPP

#include <string>
#include <string_view>

// Record format (unchanged, "VV01"):
//   "VV01" | salt (crypto_pwhash_SALTBYTES) | nonce (crypto_secretbox_NONCEBYTES)
//   | crypto_secretbox_easy(plaintext)  (MAC + ciphertext)
// Key = Argon2id13(password, salt, OPSLIMIT_INTERACTIVE, MEMLIMIT_INTERACTIVE).
class Encryption
{
public:
    Encryption();

    // On failure, encrypted_data is left unchanged.
    bool encrypt(
        std::string_view plaintext,
        std::string_view password,
        std::string &encrypted_data);

    // On failure (wrong password, tampering, truncation, bad magic) plaintext
    // is left unchanged and no partial plaintext is exposed.
    bool decrypt(
        std::string_view encrypted_data,
        std::string_view password,
        std::string &plaintext);
};

#endif
