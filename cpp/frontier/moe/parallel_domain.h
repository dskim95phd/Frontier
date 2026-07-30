#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace frontier::moe {

class ParallelDomainError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct ExpertRange {
  std::uint64_t begin = 0;
  std::uint64_t end = 0;

  [[nodiscard]] std::uint64_t size() const noexcept {
    return end - begin;
  }

  friend bool operator==(const ExpertRange&, const ExpertRange&) = default;
};

class ExpertParallelDomain {
 public:
  ExpertParallelDomain(
      std::uint64_t total_experts,
      std::uint64_t expert_parallel_size);

  [[nodiscard]] std::uint64_t total_experts() const noexcept {
    return total_experts_;
  }
  [[nodiscard]] std::uint64_t size() const noexcept {
    return expert_parallel_size_;
  }
  [[nodiscard]] std::uint64_t experts_per_lane() const noexcept {
    return total_experts_ / expert_parallel_size_;
  }
  [[nodiscard]] ExpertRange expert_range(
      std::uint64_t lane) const;
  [[nodiscard]] std::uint64_t owner(
      std::uint64_t expert_id) const;
  [[nodiscard]] std::vector<std::vector<std::uint64_t>>
  partition(const std::vector<std::uint64_t>& global_counts) const;

 private:
  std::uint64_t total_experts_;
  std::uint64_t expert_parallel_size_;
};

}  // namespace frontier::moe
