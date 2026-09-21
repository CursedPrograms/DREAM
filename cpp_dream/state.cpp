#include "state.h"

#include "common.h"

std::string SharedState::state() const {
    std::lock_guard<std::mutex> lock(m_);
    return state_;
}

void SharedState::setState(const std::string& s) {
    std::lock_guard<std::mutex> lock(m_);
    state_ = s;
}

std::string SharedState::forceVideo() const {
    std::lock_guard<std::mutex> lock(m_);
    return force_;
}

void SharedState::setForceVideo(const std::string& path) {
    std::lock_guard<std::mutex> lock(m_);
    force_ = path;
}

SharedState& S() {
    static SharedState s;
    return s;
}

void touchInteraction() {
    S().lastWakeTs = dream::nowSec();
    S().flirtPlayed = false;
}
