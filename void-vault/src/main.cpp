// Void Vault interactive application.
//
// Passwords are never accepted on the command line. They are read from stdin
// without echo when stdin is a console; when stdin is redirected (automated
// tests) they are read as plain lines.

#include <iostream>
#include <string>
#include <filesystem>

#include <sodium.h>

#include "security/auth.hpp"
#include "security/secure_buffer.hpp"
#include "vault/vault.hpp"
#include "storage.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

namespace
{
    void wipe(std::string &s)
    {
        if (!s.empty())
        {
            sodium_memzero(s.data(), s.size());
            s.clear();
        }
    }

    // RAII: disables console echo while alive (no-op if stdin is not a console).
    class EchoGuard
    {
    public:
        EchoGuard()
        {
#ifdef _WIN32
            handle_ = GetStdHandle(STD_INPUT_HANDLE);
            if (handle_ != INVALID_HANDLE_VALUE && GetConsoleMode(handle_, &mode_))
            {
                active_ = SetConsoleMode(handle_, mode_ & ~static_cast<DWORD>(ENABLE_ECHO_INPUT)) != 0;
            }
#else
            if (isatty(STDIN_FILENO) && tcgetattr(STDIN_FILENO, &old_) == 0)
            {
                termios t = old_;
                t.c_lflag &= ~static_cast<tcflag_t>(ECHO);
                active_ = tcsetattr(STDIN_FILENO, TCSAFLUSH, &t) == 0;
            }
#endif
        }
        ~EchoGuard()
        {
            if (!active_)
            {
                return;
            }
#ifdef _WIN32
            SetConsoleMode(handle_, mode_);
#else
            tcsetattr(STDIN_FILENO, TCSAFLUSH, &old_);
#endif
            std::cout << '\n';
        }
        EchoGuard(const EchoGuard &) = delete;
        EchoGuard &operator=(const EchoGuard &) = delete;

    private:
        bool active_ = false;
#ifdef _WIN32
        HANDLE handle_ = INVALID_HANDLE_VALUE;
        DWORD mode_ = 0;
#else
        termios old_{};
#endif
    };

    // Returns false on EOF. The secret goes straight into a SecureBuffer and
    // the temporary std::string is wiped.
    bool readSecret(const char *prompt, SecureBuffer &out)
    {
        std::cout << prompt << std::flush;
        std::string line;
        bool ok;
        {
            EchoGuard guard;
            ok = static_cast<bool>(std::getline(std::cin, line));
        }
        if (ok && !line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (ok)
        {
            out.assign(line);
        }
        wipe(line);
        return ok;
    }

    bool readLine(const char *prompt, std::string &out)
    {
        std::cout << prompt << std::flush;
        if (!std::getline(std::cin, out))
        {
            return false;
        }
        if (!out.empty() && out.back() == '\r')
        {
            out.pop_back();
        }
        return true;
    }

    void printHelp()
    {
        std::cout << "Usage: void_vault [--data-dir <directory>] [--help]\n"
                     "  Stores vault_auth.dat (password hash + failed-attempt counter)\n"
                     "  and vault_data.dat (encrypted record) in the data directory\n"
                     "  (default: current directory). Passwords are only read interactively.\n";
    }

    int firstTimeSetup(Authorizer &authorizer)
    {
        std::cout << "No account found. First-time setup.\n"
                  << "Choose a password of at least " << Authorizer::MIN_PASSWORD_LENGTH
                  << " characters.\n";
        for (int tries = 0; tries < 3; ++tries)
        {
            SecureBuffer first, second;
            if (!readSecret("New password: ", first) || !readSecret("Confirm password: ", second))
            {
                std::cout << "Input closed; setup aborted.\n";
                return 1;
            }
            if (first.view().size() < Authorizer::MIN_PASSWORD_LENGTH)
            {
                std::cout << "Password too short.\n";
                continue;
            }
            if (first.view() != second.view())
            {
                std::cout << "Passwords do not match.\n";
                continue;
            }
            if (!authorizer.setPassword(first.view()))
            {
                std::cerr << "Account setup failed (could not write the auth file safely).\n";
                return 1;
            }
            std::cout << "Account created.\n";
            return 0;
        }
        std::cout << "Too many invalid setup attempts.\n";
        return 1;
    }
}

int main(int argc, char **argv)
{
    std::filesystem::path data_dir = ".";
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printHelp();
            return 0;
        }
        if (arg == "--data-dir" && i + 1 < argc)
        {
            data_dir = argv[++i];
            continue;
        }
        std::cerr << "Unknown or incomplete argument: " << arg << '\n';
        printHelp();
        return 2;
    }

    std::error_code ec;
    if (!std::filesystem::is_directory(data_dir, ec))
    {
        std::cerr << "Data directory does not exist: " << data_dir.string() << '\n';
        return 2;
    }

    std::cout << "=== Void Vault ===\n";

    Authorizer authorizer((data_dir / "vault_auth.dat").string());
    Storage storage((data_dir / "vault_data.dat").string());

    if (authorizer.hasStorageError())
    {
        std::cerr << "The auth file exists but is unreadable or corrupted. Refusing to continue.\n"
                     "Recovery: restore vault_auth.dat from backup, or move it aside and re-create the\n"
                     "account with the ORIGINAL password (data stays encrypted with that password).\n";
        return 3;
    }

    if (!authorizer.isInitialized())
    {
        const int rc = firstTimeSetup(authorizer);
        if (rc != 0)
        {
            return rc;
        }
    }

    Vault vault(authorizer, storage);

    while (true)
    {
        if (authorizer.isLockedOut())
        {
            std::cerr << "Account is locked after " << Authorizer::MAX_FAILED_ATTEMPTS
                      << " failed attempts (persists across restarts).\n"
                         "Recovery is manual; see README (Lockout recovery).\n";
            vault.lock();
            return 4;
        }

        std::cout << "\n[" << (vault.isUnlocked() ? "UNLOCKED" : "LOCKED") << "] "
                  << "1) unlock  2) save text  3) show data  4) lock  5) status  6) quit\n";
        std::string choice;
        if (!readLine("> ", choice))
        {
            std::cout << "\nInput closed. Locking and exiting.\n";
            vault.lock();
            return 0;
        }

        if (choice == "1")
        {
            SecureBuffer password;
            if (!readSecret("Password: ", password))
            {
                continue;
            }
            if (vault.authenticate(password.view()) && vault.unlock())
            {
                std::cout << "Vault unlocked.\n";
            }
            else
            {
                std::cout << "Authentication failed. Remaining attempts before lockout: "
                          << (Authorizer::MAX_FAILED_ATTEMPTS - authorizer.failedAttempts()) << '\n';
            }
        }
        else if (choice == "2")
        {
            if (!vault.isUnlocked())
            {
                std::cout << "Rejected: " << vaultErrorMessage(VaultError::Locked) << '\n';
                continue;
            }
            std::string text;
            if (!readLine("Text to store (single line): ", text))
            {
                continue;
            }
            if (vault.saveData(text))
            {
                std::cout << "Saved (encrypted).\n";
            }
            else
            {
                std::cout << "Save failed: " << vaultErrorMessage(vault.lastError()) << '\n';
            }
            wipe(text);
        }
        else if (choice == "3")
        {
            std::string text;
            if (vault.loadData(text))
            {
                std::cout << "Stored data: " << text << '\n';
            }
            else
            {
                std::cout << "Load failed: " << vaultErrorMessage(vault.lastError()) << '\n';
            }
            wipe(text);
        }
        else if (choice == "4")
        {
            vault.lock();
            std::cout << "Vault locked.\n";
        }
        else if (choice == "5")
        {
            std::cout << "Unlocked: " << (vault.isUnlocked() ? "yes" : "no")
                      << " | failed attempts: " << authorizer.failedAttempts() << '/'
                      << Authorizer::MAX_FAILED_ATTEMPTS
                      << " | data file present: " << (storage.exists() ? "yes" : "no") << '\n';
        }
        else if (choice == "6" || choice == "q" || choice == "quit")
        {
            vault.lock();
            std::cout << "Vault locked. Goodbye.\n";
            return 0;
        }
        else if (!choice.empty())
        {
            std::cout << "Unknown option.\n";
        }
    }
}
