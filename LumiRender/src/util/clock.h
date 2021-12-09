//
// Created by Zero on 2021/4/30.
//


#pragma once

#include <chrono>

namespace luminous {

    inline namespace utility {

    class HpClock {
    public:
        HpClock() {
            reset();
        }

        // resets the timer
        void reset() {
            _paused = false;
            _last_stamp = _start_stamp = hp_timer::now();
            _stop_stamp = time_point{hp_timer::duration::zero()};
        }
        // starts the timer
        void resume() {
            if (_paused) {
                hp_timer::duration elapsed = hp_timer::now() - _stop_stamp;
                if (elapsed < hp_timer::duration::zero())
                    elapsed = hp_timer::duration::zero();

                _start_stamp += elapsed;
                _last_stamp = _start_stamp;
                _stop_stamp = time_point{hp_timer::duration::zero()};
                _paused = false;
            }
        }
        // stop (or pause) the timer
        void stop() {
            if (!_paused) {
                _stop_stamp = hp_timer::now();
                _paused = true;
            }
        }
        // returns true if timer stopped
        bool is_paused() const {
            return _paused;
        }
        // get the current time in seconds after the lastest Reset() or Resume() call.
        double get_time() const {
            duration secs;
            if (_paused)
                secs = duration_cast(_stop_stamp - _start_stamp);
            else
                secs = duration_cast(hp_timer::now() - _start_stamp);

            // This can happen because high resolution timer is used here.
            if (secs < duration::zero())
                secs = duration::zero();

            return secs.count();
        }
        void tick() {
            if (!_paused) {
                _last_stamp = hp_timer::now();
            }
        }
        // get the time in seconds that elapsed after the lastest Tick() call.
        double get_elapsed_time() const {
            duration secs;

            if (_paused) {
                secs = duration_cast(_stop_stamp - _last_stamp);
            } else
                secs = duration_cast(hp_timer::now() - _last_stamp);

            // This can happen because high resolution timer is used here.
            if (secs < duration::zero())
                secs = duration::zero();

            return secs.count();
        }

    private:
        using duration = std::chrono::duration<double>; // second.
        using hp_timer = std::chrono::steady_clock;
        using time_point = std::chrono::steady_clock::time_point;

        // disable copy/move/assignment
        HpClock(const HpClock &) = delete;
        HpClock &operator=(const HpClock &) = delete;

        // Convert resolution from nanosecond to second.
        static duration duration_cast(const hp_timer::duration &interval) {
            return std::chrono::duration_cast<duration>(interval);
        }

        time_point _start_stamp;
        time_point _stop_stamp;
        time_point _last_stamp;
        bool _paused;
    };

    using Clock = HpClock;
    } // luminous::utility
}// luminous