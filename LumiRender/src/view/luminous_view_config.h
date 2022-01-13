#pragma once


#ifdef LUMINOUS_VIEW_BUILD
#ifdef _WIN32
#define LUMINOUS_VIEW_LIB_VISIBILITY  __declspec(dllexport)
#else
#define LUMINOUS_VIEW_LIB_VISIBILITY __attribute__((visibility("default")))
#endif
#else
#ifdef _WIN32
#define LUMINOUS_VIEW_LIB_VISIBILITY __declspec(dllimport)
#else
#define LUMINOUS_VIEW_LIB_VISIBILITY
#endif
#endif