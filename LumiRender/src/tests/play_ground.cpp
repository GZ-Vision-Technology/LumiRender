#define _CRTDBG_MAP_ALLOC // Do not include <malloc.h>
#include "crtdbg.h"
#include <malloc.h>
#ifdef _DEBUG
#define DEBUG_NEW   new( _NORMAL_BLOCK, __FILE__, __LINE__)
#else
#define DEBUG_NEW
#endif

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

class mem_leak {

public:
    mem_leak() {
        p = malloc(100);
    }

private:
    void *p;
};

mem_leak g_a;

int main() {

    int flags = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    _CrtSetDbgFlag(flags | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF);

    int *a = new int[4];
}