#ifndef VOID_VAULT_SECURE_BUFFER_HPP
#define VOID_VAULT_SECURE_BUFFER_HPP

#include <sodium.h>

#include <cstring>
#include <new>
#include <string_view>

// Holds a secret in sodium_malloc'd memory (guard pages, mlock'd where the OS
// allows, wiped on free). Non-copyable so the secret is not duplicated.
// Limitation: data that passed through std::string / console buffers before
// being assigned here may still have copies elsewhere in memory.
class SecureBuffer
{
public:
    SecureBuffer() = default;
    SecureBuffer(const SecureBuffer &) = delete;
    SecureBuffer &operator=(const SecureBuffer &) = delete;

    SecureBuffer(SecureBuffer &&other) noexcept : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    SecureBuffer &operator=(SecureBuffer &&other) noexcept
    {
        if (this != &other)
        {
            clear();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~SecureBuffer() { clear(); }

    void assign(std::string_view value)
    {
        clear();
        if (value.empty())
        {
            return;
        }
        void *mem = sodium_malloc(value.size());
        if (mem == nullptr)
        {
            throw std::bad_alloc();
        }
        std::memcpy(mem, value.data(), value.size());
        data_ = static_cast<char *>(mem);
        size_ = value.size();
    }

    void clear() noexcept
    {
        if (data_ != nullptr)
        {
            sodium_free(data_);   // wipes before freeing
            data_ = nullptr;
            size_ = 0;
        }
    }

    bool empty() const noexcept { return size_ == 0; }
    std::string_view view() const noexcept { return {data_, size_}; }

private:
    char *data_ = nullptr;
    std::size_t size_ = 0;
};

#endif
