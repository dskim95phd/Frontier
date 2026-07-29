#pragma once

namespace frontier::entities {

struct ExecutionTime {
  double dense_compute_ms = 0.0;
  double tp_communication_ms = 0.0;
  double pp_communication_ms = 0.0;

  [[nodiscard]] double total_ms() const noexcept {
    return dense_compute_ms + tp_communication_ms +
        pp_communication_ms;
  }

  friend bool operator==(
      const ExecutionTime&,
      const ExecutionTime&) = default;
};

}  // namespace frontier::entities
