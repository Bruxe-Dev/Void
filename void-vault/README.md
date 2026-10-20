# Void Vault

Authenticated, encrypted local secret storage implemented in C++20 using **libsodium** (Argon2id password hashing and key derivation + XSalsa20-Poly1305 authenticated encryption).

---

## 1. Architecture & Security Design

| Component | Files | Responsibility |
|---|---|---|
| **Authorizer** | `src/security/auth.hpp`, `src/security/auth.cpp` | Argon2id password verification (`crypto_pwhash_str`), persistent `VVAUTH1` account record (`vault_auth.dat`), persistent 5-attempt lockout persisted **before** verification, fail-closed corruption detection, and legacy single-line hash migration. |
| **Encryption** | `src/security/encryption.hpp`, `src/security/encryption.cpp` | `VV01` record format (`"VV01" \|\| 16B salt \|\| 24B nonce \|\| 16B Poly1305 MAC \|\| XSalsa20 ciphertext`), Argon2id key derivation (`OPSLIMIT_INTERACTIVE`, `MEMLIMIT_INTERACTIVE`), key/buffer zeroization on all exit paths. |
| **SecureBuffer** | `src/security/secure_buffer.hpp` | Non-copyable RAII buffer backed by `sodium_malloc` / `sodium_free` (guard pages + memory wiping on release) holding the active session password and interactive secrets. |
| **Storage** | `include/storage.hpp`, `src/storage/storage.cpp` | Atomic file writes (`<path>.tmp` write + flush + `MoveFileExW` / `rename` replacement), non-overwrite mode (`allow_replace=false`) for account initialization, and non-destructive reads. |
| **Vault** | `src/vault/vault.hpp`, `src/vault/vault.cpp` | Coordinates authentication, unlocking/locking, encrypted save/load, explicit `VaultError` reporting, and foreign-data overwrite protection (`VaultError::ForeignDataExists`). |
| **CLI App** | `src/main.cpp` | Interactive console workflow with echo-disabled password input (`SetConsoleMode` / `termios`), `--data-dir <dir>` support, and safe EOF/lock shutdown. |

---

## 2. Prerequisites & Dependency Installation (Windows / MSYS2 UCRT64)

In an MSYS2 UCRT64 shell (or via `C:\msys64\usr\bin\pacman.exe`):

```powershell
C:\msys64\usr\bin\pacman.exe -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-libsodium mingw-w64-ucrt-x86_64-pkgconf mingw-w64-ucrt-x86_64-make
```

> **Important PATH Note on Windows:** If Anaconda (`C:\Users\HP\anaconda3\Library\mingw-w64\bin` or `Library\bin`) precedes `C:\msys64\ucrt64\bin` on `PATH`, Anaconda's older `libstdc++-6.dll` / `libsodium.dll` will shadow MSYS2's toolchain and cause `0xc0000139` DLL entry-point errors. Both `CMakeLists.txt` (`VOID_VAULT_STATIC_RUNTIME=ON`) and `scripts/build.ps1` prevent this automatically.

---

## 3. Verified Commands

All commands below assume working directory `D:\Void\void-vault`.

### Quick Build & Full Test Suite (Recommended)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -BuildType Release -BuildDir build
```

### Manual CMake Configuration & Build

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
```

### Debug Configuration & Build

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -BuildType Debug -BuildDir build-debug
```

### Run Full Test Suite

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
ctest --test-dir build --output-on-failure
```

### Run Individual Tests

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
ctest --test-dir build -R storage_test --output-on-failure -V
ctest --test-dir build -R auth_test --output-on-failure -V
ctest --test-dir build -R encryption_test --output-on-failure -V
ctest --test-dir build -R vault_test --output-on-failure -V
ctest --test-dir build -R app_e2e_test --output-on-failure -V
```

### Run the Interactive Application

```powershell
.\build\void_vault.exe --data-dir .
```

- **First launch:** Prompts for a new password (minimum 12 characters) and confirmation with console echo disabled, then atomically creates `vault_auth.dat`.
- **Subsequent launches:** Displays menu (`1) unlock  2) save text  3) show data  4) lock  5) status  6) quit`).
- **Safe shutdown:** Choose `6` (`quit`) or send EOF (`Ctrl+Z` + Enter on Windows, `Ctrl+D` on POSIX); the vault wipes its session password, revokes authorization, and exits cleanly.

---

## 4. Data Files, Formats & Recovery

- `vault_auth.dat`: 3-line ASCII record (`VVAUTH1`, `$argon2id$...` hash, `failed=<n>`). Plaintext passwords are never written.
- `vault_data.dat`: Binary `VV01` authenticated ciphertext (`4B magic + 16B salt + 24B nonce + 16B MAC + ciphertext`).
- **Lockout recovery:** After 5 consecutive failed password attempts, `failed=5` is persisted to `vault_auth.dat` and the account remains locked across application restarts. To recover locally, verify the user's identity, delete or move `vault_auth.dat`, and re-initialize with the **original password** so existing `vault_data.dat` can still be decrypted. (Vault refuses to overwrite `vault_data.dat` if the active password cannot decrypt the existing record.)
- **Corrupted `vault_auth.dat`:** Fails closed (`hasStorageError() == true`, exit code `3`) and refuses to re-initialize over the corrupted file.

---

## 5. Security Limitations & Readiness Assessment

- **Online Gate vs. Offline Confidentiality:** `Authorizer` enforces online attempt limits, while confidentiality of `vault_data.dat` relies on the Argon2id-derived key. An attacker with read access to `vault_data.dat` can mount an offline password-cracking attack bounded by Argon2id `INTERACTIVE` parameters and password entropy.
- **Residual Memory Copies:** While `Vault` stores the session password in `sodium_malloc` memory (`SecureBuffer`) and `main.cpp` wipes temporary input strings with `sodium_memzero`, OS console buffers and `std::istream` internal buffers are outside application control.
- **File Permissions & Concurrency:** On POSIX, temporary files are created with mode `0600`; on Windows, files inherit the parent directory's NTFS ACL. Cross-process file locking is not implemented.
- **Unattended Cloud Operation:** Void Vault is an interactive local secret store (**Ready for local development and testing**). Do **not** store live broker credentials in this prototype or use interactive password files for unattended cloud workers; cloud deployments require an external KMS/HSM or workload-identity secret manager.
