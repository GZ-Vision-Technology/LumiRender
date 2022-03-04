#pragma once
#include "clock.h"
#include <string>
#include <thread>
#include <atomic>
#include <cuda_runtime.h>
#include <vector>

namespace luminous {
inline namespace utility {

using psocket_t = uint64_t;

class ProgressReporter {
public:
    static void init();
    static void cleanup();

    ProgressReporter();
    ~ProgressReporter();

    /**
     * @brief Reset the progress bar state.
     * 
     * @param title 
     * @param total_work 
     * @param quiet 
     * @param gpu 
     * @param port Used for send progress bar status to socket.
     */
    void reset(std::string title, uint64_t total_work,  bool quiet = true, bool gpu= false, short port = 0);

    bool is_valid() const;

    void update(uint64_t num = 1);
    void done();

    float elapsed_seconds() const;

private:
  void update_progress_bar(int fd_count, const psocket_t fds[2], char *out_buffer, int terminal_width, uint64_t sn);
  friend void update_progress_proc(ProgressReporter *progressor);

  uint64_t _total_work;
  std::atomic<uint32_t> _work_done;
  HpClock _timer;

  bool _quiet;
  short _client_port;
  std::string _title;
  std::thread _update_thread;
  std::atomic<bool> _terminate_update;

  // gpu timeline
  std::vector<cudaEvent_t> _cu_profile_events;
  std::atomic<ptrdiff_t> _cu_event_launched_offset;
  ptrdiff_t _cu_event_finished_offset;
};

inline bool ProgressReporter::is_valid() const {
    return _total_work != 0;
}

inline float ProgressReporter::elapsed_seconds() const {
    return static_cast<float>(_timer.get_time());
}

};// namespace utility
};// namespace luminous
