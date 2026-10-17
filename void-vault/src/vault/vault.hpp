#ifndef VOID_VAULT_VAULT_HPP
#define VOID_VAULT_VAULT_HPP

class Vault{
	public:
		Vault();
		
		bool is_unlocked() const;
	private:
		bool is_unlocked;
};

#endif
