#pragma once

#ifdef _WIN32

#include <cstdlib>
#ifdef _WIN32
#include <basetsd.h>// for SSIZE_T
typedef SSIZE_T ssize_t;
#else
#include <unistd.h>
#endif

// Platform agnostic definations
#ifdef __linux__
typedef unsigned long ULONG;
typedef int HRESULT;
#endif

#include <string>

namespace luminous {
namespace fileio {

#ifdef _WIN32
typedef ::HRESULT HRESULT;
#else
typedef unsigned long HRESULT;
#endif
enum { S_RESULT_SUCCESS = 0 };

struct IUnknown12 {
    virtual ULONG Release() = 0;
    virtual ULONG AddRef() = 0;
};

template <typename T>
class ComPtr
{
public:
    typedef T InterfaceType;

protected:
    InterfaceType *ptr_;
    template<class U> friend class ComPtr;

    void InternalAddRef() const throw()
    {
        if (ptr_ != nullptr)
        {
            ptr_->AddRef();
        }
    }

    unsigned long InternalRelease() throw()
    {
        unsigned long ref = 0;
        T* temp = ptr_;

        if (temp != nullptr)
        {
            ptr_ = nullptr;
            ref = temp->Release();
        }

        return ref;
    }

public:
#pragma region constructors
    ComPtr() throw() : ptr_(nullptr)
    {
    }

    ComPtr(nullptr_t) throw() : ptr_(nullptr)
    {
    }

    template<class U>
    ComPtr(U *other) throw() : ptr_(other)
    {
        InternalAddRef();
    }

    ComPtr(const ComPtr& other) throw() : ptr_(other.ptr_)
    {
        InternalAddRef();
    }

    // copy constructor that allows to instantiate class when U* is convertible to T*
    template<class U>
    ComPtr(const ComPtr<U> &other, std::enable_if_t<std::is_convertible_v<U*, T*>, void*> = 0) throw() :
        ptr_(other.ptr_)
    {
        InternalAddRef();
    }

    ComPtr(ComPtr &&other) throw() : ptr_(nullptr)
    {
        if (this != reinterpret_cast<ComPtr*>(&reinterpret_cast<unsigned char&>(other)))
        {
            Swap(other);
        }
    }

    // Move constructor that allows instantiation of a class when U* is convertible to T*
    template<class U>
    ComPtr(ComPtr<U>&& other, std::enable_if_t<std::is_convertible_v<U*, T*>, void*> = 0) throw() :
        ptr_(other.ptr_)
    {
        other.ptr_ = nullptr;
    }
#pragma endregion

#pragma region destructor
    ~ComPtr() throw()
    {
        InternalRelease();
    }
#pragma endregion

#pragma region assignment
    ComPtr& operator=(nullptr_t) throw()
    {
        InternalRelease();
        return *this;
    }

    ComPtr& operator=(T *other) throw()
    {
        if (ptr_ != other)
        {
            ComPtr(other).Swap(*this);
        }
        return *this;
    }

    template <typename U>
    ComPtr& operator=(U *other) throw()
    {
        ComPtr(other).Swap(*this);
        return *this;
    }

    ComPtr& operator=(const ComPtr &other) throw()
    {
        if (ptr_ != other.ptr_)
        {
            ComPtr(other).Swap(*this);
        }
        return *this;
    }

    template<class U>
    ComPtr& operator=(const ComPtr<U>& other) throw()
    {
        ComPtr(other).Swap(*this);
        return *this;
    }

    ComPtr& operator=(ComPtr &&other) throw()
    {
        ComPtr(static_cast<ComPtr&&>(other)).Swap(*this);
        return *this;
    }

    template<class U>
    ComPtr& operator=(ComPtr<U>&& other) throw()
    {
        ComPtr(static_cast<ComPtr<U>&&>(other)).Swap(*this);
        return *this;
    }
#pragma endregion

#pragma region modifiers
    void Swap(ComPtr&& r) throw()
    {
        T* tmp = ptr_;
        ptr_ = r.ptr_;
        r.ptr_ = tmp;
    }

    void Swap(ComPtr& r) throw()
    {
        T* tmp = ptr_;
        ptr_ = r.ptr_;
        r.ptr_ = tmp;
    }
#pragma endregion

    operator bool() const throw()
    {
        return Get() != nullptr;
    }

    T* Get() const throw()
    {
        return ptr_;
    }

    InterfaceType* operator->() const throw()
    {
        return ptr_;
    }

    T** operator&() throw()
    {
        return GetAddressOf();
    }

    T* const* operator&() const throw()
    {
        return GetAddressOf();
    }

    T* const* GetAddressOf() const throw()
    {
        return &ptr_;
    }

    T** GetAddressOf() throw()
    {
        return &ptr_;
    }

    T** ReleaseAndGetAddressOf() throw()
    {
        InternalRelease();
        return &ptr_;
    }

    T* Detach() throw()
    {
        T* ptr = ptr_;
        ptr_ = nullptr;
        return ptr;
    }

    void Attach(InterfaceType* other) throw()
    {
        if (ptr_ != nullptr)
        {
            auto ref = ptr_->Release();
            // Attaching to the same object only works if duplicate references are being coalesced. Otherwise
            // re-attaching will cause the pointer to be released and may cause a crash on a subsequent dereference.
            LM_ASSERT(ref != 0 || ptr_ != other);
        }

        ptr_ = other;
    }

    unsigned long Reset()
    {
        return InternalRelease();
    }

    HRESULT CopyTo(InterfaceType** ptr) const throw()
    {
        InternalAddRef();
        *ptr = ptr_;
        return 0;
    }

    template<typename U>
    HRESULT CopyTo(U** ptr, std::enable_if_t<std::is_convertible_v<U*, T*>, void*> = 0) const throw()
    {
        InternalAddRef();
        *ptr = ptr_;
    }
#pragma endregion
};    // ComPtr

std::string GetFormatErrorMessage(HRESULT hr);

struct IFileDataBlob: public IUnknown12 {
    virtual void *GetBufferPointer() const = 0;
    virtual size_t GetBufferSize() const = 0;
};

HRESULT CreateFileDataBlob(
        size_t iReqestSizeInBytes,
        BOOL bAllowWrite,
        IFileDataBlob **ppResult);

unsigned long long GetTotalToElpasedMicroseconds();

HRESULT ReadFileDirectly(const char *pFileName,
                         ssize_t iOffsetInBytes,  // File offset in content block,
                         size_t iRequestSizeInBytes,// File content size, when 0 is specified, read to file end
                         IFileDataBlob **ppResult);

};// namespace fileio
};// namespace luminous

#endif
