#ifndef VOID_VAULT_VAULT_HPP
#define VOID_VAULT_VAULT_HPP

class Vault{
	public:
		Vault();
		
		bool isUnlocked() const;
	private:
		bool is_unlocked;
};

#endif
