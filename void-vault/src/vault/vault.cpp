#include "vault.hpp"

Vault::Vault()
    : is_unlocked(false)
{
}

bool Vault::isUnlocked() const {
    return is_unlocked;
}

