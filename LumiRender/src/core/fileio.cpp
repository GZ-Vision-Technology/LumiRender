#ifdef _WIN32
#include <windows.h>
#include <cstdint>
#include <algorithm>
#include "fileio.h"
#include <comdef.h>
#include <chrono>

namespace luminous {
namespace fileio {

#define ALIGN_UP(ptr, alignment) \
    (uintptr_t)(((uintptr_t) (ptr) + (uintptr_t) (alignment) -1) & ~((uintptr_t) (alignment) -1))
#define ALIGN_DOWN(ptr, alignment) (uintptr_t)((uintptr_t) (ptr) & ~((uintptr_t) (alignment) -1))

// IO no buffering
#define DIRECTIO_NO_BUFFERING 1

#define BUFFERING_IO_SIZE_THRESHOLD     ((size_t)(1 << 16))

#define DIRECTIO_IO_SIZE_THRESHOLD    ((size_t)(1 << 20))

enum IO_RESULT_TYPE {
    IO_RESULT_TYPE_NONE,
    IO_RESULT_TYPE_HEAP,
    IO_RESULT_TYPE_MAPPING,
    IO_RESULT_TYPE_DIRECT_IO
};

struct FileDataBlobImpl : public IFileDataBlob {

    static FileDataBlobImpl *CreateFromInplaceHeap(size_t heapSize) {
        const size_t alignment = sizeof(void *);
        size_t mempos = reinterpret_cast<size_t>(new BYTE[sizeof(FileDataBlobImpl) + heapSize + alignment - 1 + sizeof(void *)]);
        size_t buffpos = (mempos + sizeof(void *) + alignment - 1) & ~(alignment - 1);
        void **pbuffpos = reinterpret_cast<void **>(buffpos);
        pbuffpos[-1] = reinterpret_cast<void *>(mempos);

        FileDataBlobImpl *pHeader = reinterpret_cast<FileDataBlobImpl *>(buffpos);
        ::new (pHeader) FileDataBlobImpl();
        pHeader->IoType = IO_RESULT_TYPE_HEAP;
        pHeader->Data = (BYTE *) pHeader + sizeof(FileDataBlobImpl);
        pHeader->Size = heapSize;
        pHeader->Heap.pHeap = pHeader;
        return pHeader;
    }

    static FileDataBlobImpl *CreateFromFileMappingView(HANDLE hView, void *pView, SSIZE_T iOffset, SIZE_T iSize) {
        FileDataBlobImpl *pHeader = new FileDataBlobImpl;
        pHeader->IoType = IO_RESULT_TYPE_MAPPING;
        pHeader->Data = (PBYTE) pView + iOffset;
        pHeader->Size = iSize;
        pHeader->Mapped.hFileMapping = hView;
        pHeader->Mapped.pMappedView = pView;
        return pHeader;
    }

    static FileDataBlobImpl* CreateFromCommittedPages(void *pv, SSIZE_T iOffset, SIZE_T iSize) {
        FileDataBlobImpl *pHeader = new FileDataBlobImpl;
        pHeader->IoType = IO_RESULT_TYPE_DIRECT_IO;
        pHeader->Data = (PBYTE)pv + iOffset;
        pHeader->Size = iSize;
        pHeader->Pages.pAlloc = pv;
        return pHeader;
    }

    ULONG Release() override {
        LONG refcnt = InterlockedDecrement(&Refcnt);
        if (refcnt == 0) {
            switch (IoType) {
                case IO_RESULT_TYPE_HEAP:
                    this->~FileDataBlobImpl();
                    delete[](static_cast<BYTE *>(static_cast<void **>(this->Heap.pHeap)[-1]));
                    break;
                case IO_RESULT_TYPE_MAPPING:
                    CloseHandle(Mapped.hFileMapping);
                    delete this;
                    break;
                case IO_RESULT_TYPE_DIRECT_IO:
                    VirtualFree(Pages.pAlloc, 0, MEM_FREE);
                    delete this;
                    break;
            }
        }
        return refcnt;
    }
    ULONG AddRef() override {
        return InterlockedIncrement(&Refcnt);
    }
    void *GetBufferPointer() const override {
        return Data;
    }
    size_t GetBufferSize() const override {
        return Size;
    }

private:
    volatile ULONG Refcnt;
    IO_RESULT_TYPE IoType;
    void *Data;
    size_t Size;
    union {
        struct {
            HANDLE hFileMapping;
            VOID *pMappedView;
        } Mapped;
        struct {
            VOID *pAlloc;
        } Pages;
        struct {
            VOID *pHeap;
        } Heap;
    };

private:
    FileDataBlobImpl() {
        Refcnt = 1;
        IoType = IO_RESULT_TYPE_NONE;
        Data = nullptr;
        Size = 0;
    }
    ~FileDataBlobImpl() {
    }
};

HRESULT _MapFileDirectly(LPCSTR pFileName, SSIZE_T iOffsetInBytes, SIZE_T iReqSizeInBytes,
                         IFileDataBlob **ppResult) {

    HANDLE hFile;
    SIZE_T fileSize;
    SYSTEM_INFO sysInfo;
    LARGE_INTEGER startOffset, endOffset;
    LARGE_INTEGER reqSize;
    HANDLE hFileMapping = NULL;
    PVOID pMappedView = NULL;
    FileDataBlobImpl *pResult;

    hFile = CreateFileA(pFileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
        return HRESULT_FROM_WIN32(GetLastError());

    // validate arguments
    GetFileSizeEx(hFile, (PLARGE_INTEGER) &fileSize);
    if (iOffsetInBytes < 0 || (size_t) iOffsetInBytes > fileSize ||
        ((size_t) iOffsetInBytes + iReqSizeInBytes) > fileSize) {
        CloseHandle(hFile);
        return E_INVALIDARG;
    }

    GetSystemInfo(&sysInfo);
    startOffset.QuadPart = ALIGN_DOWN(iOffsetInBytes, sysInfo.dwAllocationGranularity);
    if (iReqSizeInBytes) {
        endOffset.QuadPart = iOffsetInBytes + iReqSizeInBytes;
    } else {
        GetFileSizeEx(hFile, &endOffset);
        iReqSizeInBytes = endOffset.QuadPart - iOffsetInBytes;
    }

    reqSize.QuadPart = (endOffset.QuadPart - startOffset.QuadPart);

    hFileMapping = CreateFileMappingW(hFile, NULL, PAGE_READONLY, reqSize.HighPart, reqSize.LowPart, NULL);
    CloseHandle(hFile);
    if (hFileMapping == NULL)
        return HRESULT_FROM_WIN32(GetLastError());

    pMappedView = MapViewOfFile(hFileMapping, FILE_MAP_READ, startOffset.HighPart, startOffset.LowPart,
                                ALIGN_UP(reqSize.QuadPart, sysInfo.dwPageSize));
    if (pMappedView == NULL) {
        CloseHandle(hFileMapping);
        return HRESULT_FROM_WIN32(GetLastError());
    }

    pResult = FileDataBlobImpl::CreateFromFileMappingView(hFileMapping, pMappedView, iOffsetInBytes - startOffset.QuadPart,
                                                          iReqSizeInBytes);
    *ppResult = pResult;

    return S_OK;
}

HRESULT _ReadFileBuffering(LPCSTR pFileName, SSIZE_T iOffsetInBytes, SIZE_T iReqSizeInBytes,
                            IFileDataBlob **ppResult) {

    HRESULT hr = S_OK;
    HANDLE hFile;
    LARGE_INTEGER startOffset, endOffset;
    FileDataBlobImpl *pResult;
    DWORD bytesToRead, bytesXfer;
    PBYTE pBuffer;

    hFile = CreateFileA(pFileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                        FILE_FLAG_SEQUENTIAL_SCAN,
                        NULL);
    if (hFile == INVALID_HANDLE_VALUE)
        return HRESULT_FROM_WIN32(GetLastError());

    startOffset.QuadPart = iOffsetInBytes;
    endOffset.QuadPart = iOffsetInBytes + (SSIZE_T) iReqSizeInBytes;

    pResult = FileDataBlobImpl::CreateFromInplaceHeap(iReqSizeInBytes);
    pBuffer = reinterpret_cast<PBYTE>(pResult->GetBufferPointer());

    while (startOffset.QuadPart < endOffset.QuadPart) {
        bytesToRead = (DWORD) std::min((SIZE_T)(endOffset.QuadPart - startOffset.QuadPart), (SIZE_T)(DWORD) (-1));
        if (!ReadFile(hFile, pBuffer, bytesToRead, &bytesXfer, NULL)) {
            CloseHandle(hFile);
            pResult->Release();
            return HRESULT_FROM_WIN32(GetLastError());
        }
        startOffset.QuadPart += bytesXfer;
        pBuffer += bytesXfer;
    }

    *ppResult = pResult;
    CloseHandle(hFile);
    return hr;
}

HRESULT _ReadFileDirectly(LPCSTR pFileName, SSIZE_T iOffsetInBytes, SIZE_T iReqSizeInBytes, BOOL bMandatory,
                          IFileDataBlob **ppResult) {

    // Internal config
    CONST UINT MaxOverlappedCount = 4;

    HRESULT hr = S_OK;
    HANDLE hFile;
    SIZE_T fileSize;
    SYSTEM_INFO sysInfo;
    LARGE_INTEGER startOffset, endOffset;
    LARGE_INTEGER reqSize;
    LARGE_INTEGER offset;
    PBYTE pv;
    LONGLONG blockSize;
    OVERLAPPED ov[MaxOverlappedCount];
    HANDLE hEvents[MaxOverlappedCount];
    INT i, ovCount;
    DWORD endBits;
    INT rc;
    FileDataBlobImpl *pResult;

    GetSystemInfo(&sysInfo);
    blockSize = 2 * sysInfo.dwPageSize;

    hFile = CreateFileA(pFileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
#if DIRECTIO_NO_BUFFERING
                        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
#else
                        FILE_FLAG_OVERLAPPED,
#endif
                        NULL);
    if (hFile == INVALID_HANDLE_VALUE)
        return HRESULT_FROM_WIN32(GetLastError());

    // validate arguments
    GetFileSizeEx(hFile, (PLARGE_INTEGER)&fileSize);
    if (iOffsetInBytes < 0 || (SIZE_T) iOffsetInBytes > fileSize ||
        ((SIZE_T) iOffsetInBytes + iReqSizeInBytes) > fileSize) {
        CloseHandle(hFile);
        return E_INVALIDARG;
    }

    // modify request file content size
    startOffset.QuadPart = iOffsetInBytes;
    if (iReqSizeInBytes) {
        endOffset.QuadPart = iOffsetInBytes + (SSIZE_T)iReqSizeInBytes;
    } else {
        endOffset.QuadPart = fileSize;
        iReqSizeInBytes = endOffset.QuadPart - iOffsetInBytes;
    }

    // For too samll file block, just read it using system buffering.
    if (iReqSizeInBytes < std::max(BUFFERING_IO_SIZE_THRESHOLD, (SIZE_T)(MaxOverlappedCount * blockSize))) {
        CloseHandle(hFile);
        return _ReadFileBuffering(pFileName, iOffsetInBytes, iReqSizeInBytes, ppResult);
    }

    // For too big file block, delegate to file mapping if not mandatory
    if(!bMandatory && iReqSizeInBytes >= DIRECTIO_IO_SIZE_THRESHOLD) {
        CloseHandle(hFile);
        hr = _MapFileDirectly(pFileName, iOffsetInBytes, iReqSizeInBytes, ppResult);
        if(FAILED(hr))
            hr = _ReadFileDirectly(pFileName, iOffsetInBytes, iReqSizeInBytes, TRUE, ppResult);
        return hr;
    }

#if DIRECTIO_NO_BUFFERING
    startOffset.QuadPart = ALIGN_DOWN(iOffsetInBytes, sysInfo.dwPageSize);
    endOffset.QuadPart = ALIGN_UP(iOffsetInBytes + (SSIZE_T)iReqSizeInBytes, sysInfo.dwPageSize);
#endif

    reqSize.QuadPart = ALIGN_UP(endOffset.QuadPart - startOffset.QuadPart, sysInfo.dwPageSize);

    pv = (PBYTE) VirtualAlloc(NULL, reqSize.QuadPart, MEM_COMMIT, PAGE_READWRITE);
    if (pv == NULL) {
        CloseHandle(hFile);
        return HRESULT_FROM_WIN32(GetLastError());
    }

    ZeroMemory(ov, sizeof(ov));
    ZeroMemory(hEvents, sizeof(hEvents));

    offset = startOffset;
    ovCount = 0;
    endBits = 0;
    for (i = 0; i < MaxOverlappedCount; ++i, ++ovCount) {
        if (offset.QuadPart >= endOffset.QuadPart)
            break;

        if ((ov[i].hEvent = CreateEventExW(NULL, NULL, CREATE_EVENT_MANUAL_RESET, EVENT_ALL_ACCESS)) == NULL) {
            hr = HRESULT_FROM_WIN32(GetLastError());
            goto cleanup;
        }
        hEvents[i] = ov[i].hEvent;
        ov[i].Offset = offset.LowPart;
        ov[i].OffsetHigh = offset.HighPart;

        if (!ReadFile(hFile, pv + (offset.QuadPart - startOffset.QuadPart),
                      (DWORD) std::min(blockSize, endOffset.QuadPart - offset.QuadPart), NULL, &ov[i]) &&
            (rc = GetLastError()) != ERROR_IO_PENDING) {
            CloseHandle(hEvents[i]);
            hr = HRESULT_FROM_WIN32(rc);
            goto cleanup;
        }

        offset.QuadPart += blockSize;
        endBits |= (1 << i);
    }

    while (endBits) {

        rc = WaitForMultipleObjects(ovCount, hEvents, FALSE, INFINITE);
        if (rc == WAIT_TIMEOUT) {
            hr = E_FAIL;
            goto cleanup;
        } else if (rc == WAIT_FAILED) {
            hr = HRESULT_FROM_WIN32(GetLastError());
            goto cleanup;
        }

        for (i = rc - WAIT_OBJECT_0; i < ovCount; ++i) {
            rc = WaitForSingleObject(hEvents[i], 0);
            if (rc != WAIT_OBJECT_0)
                continue;
            if (!HasOverlappedIoCompleted(&ov[i])) {
                hr = E_FAIL;
                goto cleanup;
            }

            ResetEvent(hEvents[i]);
            offset.LowPart = ov[i].Offset;
            offset.HighPart = ov[i].OffsetHigh;
            offset.QuadPart += ovCount * blockSize;

            if (offset.QuadPart < endOffset.QuadPart) {
                ov[i].Offset = offset.LowPart;
                ov[i].OffsetHigh = offset.HighPart;

                if (!ReadFile(hFile, pv + (offset.QuadPart - startOffset.QuadPart),
                              (DWORD) std::min(blockSize, endOffset.QuadPart - offset.QuadPart), NULL, &ov[i]) &&
                    (rc = GetLastError()) != ERROR_IO_PENDING) {
                    hr = HRESULT_FROM_WIN32(rc);
                    goto cleanup;
                }
            } else {
                endBits &= ~(1 << i);
            }
        }
    }

#if DIRECTIO_NO_BUFFERING
    pResult = FileDataBlobImpl::CreateFromCommittedPages(pv, startOffset.QuadPart - iOffsetInBytes, iReqSizeInBytes);
#else
    pResult = FileDataBlobImpl::CreateFromCommittedPages(pv, 0, iReqSizeInBytes);
#endif
    *ppResult = pResult;
    pv = NULL;

cleanup:
    if(pv)
      VirtualFree(pv, 0, MEM_FREE);
    for (i = 0; i < ovCount; ++i)
        CloseHandle(hEvents[i]);
    CloseHandle(hFile);
    return hr;
}

std::string GetFormatErrorMessage(HRESULT hr) {
    _com_error err(hr);
    LPCSTR pErrMsg = err.ErrorMessage();

    std::string err_str;

    err_str = pErrMsg;
    LocalFree((HLOCAL) pErrMsg);
    return err_str;
}

static unsigned long long _total_ms_elapsed = 0;

unsigned long long GetTotalToElpasedMicroseconds() {
    return _total_ms_elapsed;
}

_Use_decl_annotations_
        HRESULT
        ReadFileDirectly(_In_ const char *pFileName, _In_ ssize_t iOffsetInBytes, _In_ size_t iRequestSizeInBytes,
                         IFileDataBlob **ppResult) {

    HRESULT hr;

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    if (ppResult == nullptr)
        return E_INVALIDARG;

    *ppResult = NULL;

    hr = _ReadFileDirectly(pFileName, iOffsetInBytes, iRequestSizeInBytes, FALSE, ppResult);

    std::chrono::microseconds ms_elpased = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
    _total_ms_elapsed += ms_elpased.count();

    return hr;
}

}// namespace fileio
}// namespace luminous

#endif