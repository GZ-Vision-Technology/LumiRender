#include "progressreporter.h"
#include <cmath>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#elif defined(__linux__)
#include <sys/unistd.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#define closesocket(fd) close(fd)

#endif

#ifdef _WIN32
#include <Windows.h>
#include <WinSock2.h>
#include <WS2tcpip.h>

#ifdef _WIN32
#define perror(msg) fprintf(stderr, "%s error: %u\n", msg, WSAGetLastError());
#endif

#endif

#include <string.h>

namespace luminous {
inline namespace utility {

    static int get_terminal_width() {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE || !h) {
        fprintf(stderr, "GetStdHandle() call failed");
        return 80;
    }
    CONSOLE_SCREEN_BUFFER_INFO bufferInfo = {0};
    GetConsoleScreenBufferInfo(h, &bufferInfo);
    return bufferInfo.dwSize.X;
#else
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) < 0) {
        // ENOTTY is fine and expected, e.g. if output is being piped to a file.
        if (errno != ENOTTY) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                fprintf(stderr, "Error in ioctl() in TerminalWidth(): %d\n", errno);
            }
        }
        return 80;
    }
    return w.ws_col;
#endif
}

int create_udp_connection(short port, psocket_t fds[2]) {

    int ret_count = 0;
    int fd_count = 0;
    struct addrinfo hints;
    struct addrinfo *res = NULL, *ptr;
    char service_port[8];
    int i;
    int max_buff_len = 1024;

    if(port == 0)
        return ret_count;

    snprintf(service_port, 8, "%u", port);

    memset(&hints, 0, sizeof(hints));
    hints.ai_flags = AI_NUMERICSERV;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = 0;

    if (getaddrinfo(NULL, service_port, &hints, &res)) {
        perror("getaddrinfo");
        goto cleanup;
    }

    for (ptr = res, i = 0; ptr != NULL && i < 2; ptr = ptr->ai_next, ++i) {

        if ((fds[i] = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol)) ==
            0) {
            perror("socket");
            goto cleanup;
        }
        fd_count = i + 1;

        if ((setsockopt(fds[i], SOL_SOCKET, SO_SNDBUF, (const char *) &max_buff_len,
                        sizeof(max_buff_len))) < 0) {
            perror("setsockop");
            goto cleanup;
        }

        if (connect(fds[i], ptr->ai_addr, (int) ptr->ai_addrlen) < 0) {
            perror("connect");
            goto cleanup;
        }
    }

    ret_count = fd_count;
    fd_count = 0;

cleanup:
    for (i = 0; i < fd_count; ++i)
        closesocket(fds[i]);
    freeaddrinfo(res);

    return ret_count;
}

void close_udp_connection(int fd_count, const psocket_t fds[2]) {

    for (int i = 0; i < fd_count; ++i)
        closesocket(fds[i]);
}

void write_progress_info_to_file(
    int fd_count,
    const psocket_t fds[2],
    bool terminate,
    uint64_t sn,
    const char *title,
    float percentage,
    float ELA,
    float ETA) {

    if(fd_count == 0)
        return;

    const int max_buff_len = 1024;
    char buffer[max_buff_len];
    int buff_len;
    char progress_json_fmt[] =
            R"PROGRESSINFO({ "type":"ProgressInfo","sn":%llu,"title":"%s","percentage":%.3f,"ELA":%.3f,"ETA":%.3f })PROGRESSINFO";
    char progress_end_json_fmt[] =
            R"PROGRESSEND({ "type":"ProgressEndSpecifier","sn":%llu,"title":"%s" })PROGRESSEND";
    int i;

    if(!terminate)
        buff_len = snprintf(buffer, max_buff_len, progress_json_fmt, sn, title, percentage, ELA, ETA);
    else
        buff_len = snprintf(buffer, max_buff_len, progress_end_json_fmt, sn, title);

    for (i = 0; i < fd_count; ++i)
        send(fds[i], buffer, buff_len, 0);
}


void update_progress_proc(ProgressReporter *progressor) {

    std::chrono::milliseconds sleep_duration{progressor->_cu_profile_events.size() ? 50 : 250};
    uint64_t iter_count = 0;
    long terminal_width = get_terminal_width();
    std::string out_buffer;
    int fd_count;
    psocket_t fds[2];

    out_buffer.resize(terminal_width * 8);

    fd_count = create_udp_connection(progressor->_client_port, fds);

    while (!progressor->_terminate_update) {
        progressor->update_progress_bar(fd_count, fds, out_buffer.data(), terminal_width, iter_count);
        std::this_thread::sleep_for(sleep_duration);

        // Periodically increase sleepDuration to reduce overhead of
        // updates.
        ++iter_count;
        if (iter_count == 70)
            // Up to 0.5s after ~2.5s elapsed
            sleep_duration *= 2;
        else if (iter_count == 520)
            // Up to 1s after an additional ~30s have elapsed.
            sleep_duration *= 2;
        else if (iter_count == 4140)
            // After 15m, jump up to 2s intervals
            sleep_duration *= 2;
    }

    // Update for the last time
    progressor->update_progress_bar(fd_count, fds, out_buffer.data(), terminal_width, iter_count);

    if(fd_count) {
        std::this_thread::sleep_for(std::chrono::milliseconds(0));
        // Write end specifier for 5 times
        for(iter_count = 0; iter_count < 5; ++iter_count) {
            write_progress_info_to_file(fd_count, fds, true, iter_count, progressor->_title.c_str(), .0f, .0f, .0f);
            std::this_thread::sleep_for(std::chrono::milliseconds(0));
        }
    }

    close_udp_connection(fd_count, fds);
}

static struct ProgressReporterGuard {

    ProgressReporterGuard() {
        ProgressReporter::init();
    }

    ~ProgressReporterGuard() {
        ProgressReporter::cleanup();
    }
} _ProgressReporterGuard;

void ProgressReporter::init() {
#ifdef _WIN32
  int rc;
  WSADATA wsaData;
if ((rc = WSAStartup(MAKEWORD(2, 2), &wsaData))) {
    fprintf(stderr, "WSAStartup failed: %u\n", rc);
  }
#endif
}

void ProgressReporter::cleanup() {
#ifdef _WIN32
  if (WSACleanup()) {
    fprintf(stderr, "WSACleanup failed: %u\n", WSAGetLastError());
  }
#endif
}

ProgressReporter::ProgressReporter()
    : _quiet(true), _total_work(0) {
}

ProgressReporter::~ProgressReporter() {
    done();
}

void ProgressReporter::reset(std::string title, uint64_t total_work, bool quiet, bool gpu, short port) {
    done();

    _title = std::move(title);
    _total_work = total_work;
    _quiet = quiet;

    if(!is_valid())
        return;

    _work_done = 0;
    if (gpu) {
        _cu_event_launched_offset = 0;
        _cu_event_finished_offset = 0;
        _cu_profile_events.resize(_total_work);
        for (auto &cu_ev : _cu_profile_events) {
            cudaEventCreate(&cu_ev);
        }
    }

    _timer.reset();

    if (!_quiet) {
        _terminate_update = false;
        _update_thread = std::thread{
                [this] {
                    update_progress_proc(this);
                }};

        _client_port = port;
    }
}

void ProgressReporter::done() {

    if (!_cu_profile_events.empty()) {
        while (_cu_event_finished_offset < _cu_event_launched_offset) {
            cudaError_t err = cudaEventSynchronize(_cu_profile_events[_cu_event_finished_offset]);
            if (err == cudaSuccess)
                ++_cu_event_finished_offset;
            else
                break;
        }
        _work_done = _cu_profile_events.size();
        _timer.stop();
    }

    if (!_quiet) {
        if (_update_thread.joinable()) {
            bool fa = false;
            while (!_terminate_update.compare_exchange_weak(fa, true,
                                                            std::memory_order::memory_order_relaxed, std::memory_order_relaxed))
                ;

            _update_thread.join();
            fprintf(stdout, "\n");
        }
    }

    for (auto &cu_ev : _cu_profile_events) {
        cudaEventDestroy(cu_ev);
    }
    _cu_profile_events.clear();
}

void ProgressReporter::update(uint64_t num) {

    if (!is_valid()) return;

    if (!_cu_profile_events.empty()) {
        if (_cu_event_launched_offset + num <= _cu_profile_events.size()) {
            while (num-- > 0) {
                cudaEventRecord(_cu_profile_events[_cu_event_launched_offset]);
                ++_cu_event_launched_offset;
            }
        }
        return;
    }
    if (num == 0 || _quiet)
        return;
    _work_done += num;
    if (_work_done == _total_work)
        _timer.stop();
}

void ProgressReporter::update_progress_bar(int fd_count, const psocket_t fds[2], char *out_buffer, int terminal_width, uint64_t sn) {

    if (!_cu_profile_events.empty()) {

        while (_cu_event_finished_offset < _cu_event_launched_offset) {
            cudaError_t err = cudaEventQuery(_cu_profile_events[_cu_event_finished_offset]);
            if (err == cudaSuccess)
                ++_cu_event_finished_offset;
            else if (err == cudaErrorNotReady)
                break;
        }

        _work_done = _cu_event_finished_offset;
        if (_work_done == _total_work)
            _timer.stop();
    }

    float percentage = std::min((float) _work_done / _total_work, 1.0f) * 100.0f;
    auto elapsed = _timer.get_time();
    double estimated_total((long long) (elapsed / std::max(percentage / 100.0f, 1.0E-3f)));
    auto remaining = estimated_total - elapsed;

    write_progress_info_to_file(fd_count, fds, false, sn, _title.c_str(),
        percentage,
        static_cast<float>(elapsed),
        static_cast<float>(remaining));

    {
        const int PROGRESS_BAR_WIDTH = std::max<int>(2, terminal_width - _title.size() - 36);

        constexpr char BAR_START[] = "\u2595";
        constexpr char BAR_STOP[] = "\u258F";
        constexpr char PROGRESS_BLOCK[] = "\u2588";
        constexpr char CONSOLE_COLOR_GREEN[] = "\033[0;42;32m";
        constexpr char CONSOLE_COLOR_GRAY[] = "\033[0;100;90m";
        constexpr char CONSOLE_COLOR_NONE[] = "\033[0m";

        constexpr char subprogress_blocks[][6] = {"", "\u258F", "\u258E", "\u258D",
                                                  "\u258C", "\u258B", "\u258A", "\u2589"};
        constexpr char shading_block[] = " ";

        constexpr int NUM_SUBBLOCKS = (sizeof(subprogress_blocks) / sizeof(subprogress_blocks[0]));

        size_t i;
        size_t total_blocks = PROGRESS_BAR_WIDTH * NUM_SUBBLOCKS;
        size_t done = (size_t) std::ceil(percentage / 100.0f * total_blocks);
        size_t num_blocks = done / NUM_SUBBLOCKS;
        size_t num_subblocks = done % NUM_SUBBLOCKS;

        int out_buffer_pos = 0;

        out_buffer_pos += sprintf(out_buffer + out_buffer_pos, "\033[1K\r%s: %s%s", _title.c_str(), CONSOLE_COLOR_GREEN, BAR_START);

        for (i = 0; i < num_blocks; i++) {
            out_buffer_pos += sprintf(out_buffer + out_buffer_pos, "%s", PROGRESS_BLOCK);
        }

        if (num_subblocks) {
            out_buffer_pos += sprintf(out_buffer + out_buffer_pos, "%s", subprogress_blocks[num_subblocks]);
            i++;
        }

        if(done != total_blocks) {
            out_buffer_pos += sprintf(out_buffer + out_buffer_pos, "%s", CONSOLE_COLOR_GRAY);
            for (; i < PROGRESS_BAR_WIDTH; i++) {
                out_buffer_pos += sprintf(out_buffer + out_buffer_pos, shading_block);
            }
        }

        out_buffer_pos += sprintf(out_buffer + out_buffer_pos, "%s%s", BAR_STOP, CONSOLE_COLOR_NONE);

        // 4 + 6 + 3 + 6 + 3 + 1
        if (percentage < 100.0) {
            out_buffer_pos += sprintf(out_buffer + out_buffer_pos, " %4.1f%%|ELA:%3.1fs|ETA:%3.1fs",
                                      percentage,
                                      elapsed,
                                      remaining);
        } else {
            out_buffer_pos += sprintf(out_buffer + out_buffer_pos, "%4.1f%%|ELA:%3.1fs",
                                      percentage,
                                      elapsed);
        }

        out_buffer_pos += sprintf(out_buffer + out_buffer_pos, "\033[0K");

        fputs(out_buffer, stdout);
        fflush(stdout);
    }
}

}// namespace utility
}// namespace luminous::utility