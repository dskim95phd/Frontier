#include "frontier/request_generator/workload.h"

#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace frontier::request_generator {
namespace {

constexpr std::string_view kSessionStartAt = "session_start_at";
constexpr std::string_view kThinkTime = "think_time";
constexpr std::string_view kNumPrefillTokens = "num_prefill_tokens";
constexpr std::string_view kNumDecodeTokens = "num_decode_tokens";
constexpr std::string_view kSessionId = "session_id";
constexpr std::string_view kSessionTurnIndex = "session_turn_index";

std::string_view trim(std::string_view value) {
    const std::size_t first = value.find_first_not_of(" \t\r");
    if (first == std::string_view::npos) {
        return {};
    }
    const std::size_t last = value.find_last_not_of(" \t\r");
    return value.substr(first, last - first + 1);
}

bool is_missing_time(SimTime value) noexcept {
    return !value.valid() && value.seconds() == SimTime{}.seconds();
}

std::vector<std::string_view> split_row(std::string_view row,
                                        std::size_t line_number) {
    if (row.find('"') != std::string_view::npos) {
        throw WorkloadError(
            "CSV quoting is unsupported in normalized workload input at line " +
            std::to_string(line_number));
    }

    std::vector<std::string_view> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = row.find(',', start);
        if (comma == std::string_view::npos) {
            fields.push_back(trim(row.substr(start)));
            break;
        }
        fields.push_back(trim(row.substr(start, comma - start)));
        start = comma + 1;
    }
    return fields;
}

double parse_finite_double(std::string_view text, std::string_view field,
                           std::size_t line_number) {
    if (text.empty()) {
        throw WorkloadError(std::string{field} + " is empty at line " +
                            std::to_string(line_number));
    }

    double value = 0.0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value,
                        std::chars_format::general);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw WorkloadError(std::string{field} + " must be numeric at line " +
                            std::to_string(line_number) + ": '" +
                            std::string{text} + "'");
    }
    if (!std::isfinite(value)) {
        throw WorkloadError(std::string{field} + " must be finite at line " +
                            std::to_string(line_number));
    }
    return value;
}

std::uint64_t parse_positive_integer(std::string_view text,
                                     std::string_view field,
                                     std::size_t line_number) {
    std::uint64_t value = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error == std::errc{} && end == text.data() + text.size()) {
        if (value == 0) {
            throw WorkloadError(std::string{field} +
                                " must be positive at line " +
                                std::to_string(line_number));
        }
        return value;
    }

    const double numeric = parse_finite_double(text, field, line_number);
    if (numeric <= 0.0) {
        throw WorkloadError(std::string{field} + " must be positive at line " +
                            std::to_string(line_number));
    }
    throw WorkloadError(std::string{field} + " must be an integer at line " +
                        std::to_string(line_number));
}

std::uint64_t parse_nonnegative_integer(std::string_view text,
                                        std::string_view field,
                                        std::size_t line_number) {
    std::uint64_t value = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error == std::errc{} && end == text.data() + text.size()) {
        return value;
    }

    const double numeric = parse_finite_double(text, field, line_number);
    if (numeric < 0.0) {
        throw WorkloadError(std::string{field} +
                            " must be nonnegative at line " +
                            std::to_string(line_number));
    }
    throw WorkloadError(std::string{field} + " must be an integer at line " +
                        std::to_string(line_number));
}

std::optional<std::uint64_t> parse_optional_nonnegative_integer(
    std::string_view text, std::string_view field, std::size_t line_number) {
    if (text.empty()) {
        return std::nullopt;
    }
    return parse_nonnegative_integer(text, field, line_number);
}

std::string_view
field_at(const std::vector<std::string_view> &fields,
         const std::unordered_map<std::string, std::size_t> &column_indices,
         std::string_view column) {
    return fields.at(column_indices.at(std::string{column}));
}

void validate_session_timing(const std::vector<WorkloadRequest> &requests) {
    std::unordered_set<SessionId::ValueType> seen_sessions;
    std::unordered_map<SessionId::ValueType, std::uint64_t>
        last_turn_by_session;
    for (const WorkloadRequest &request : requests) {
        if (!request.session_start_at.valid() &&
            !is_missing_time(request.session_start_at)) {
            throw WorkloadError(
                "session_start_at must be finite and nonnegative when "
                "present; request_id=" +
                std::to_string(request.request_id.value()));
        }
        if (!request.think_time.valid()) {
            throw WorkloadError("think_time must be finite and nonnegative; "
                                "request_id=" +
                                std::to_string(request.request_id.value()));
        }
        if (!request.session_id.valid()) {
            if (!request.session_start_at.valid()) {
                throw WorkloadError(
                    "a non-session request requires session_start_at; "
                    "request_id=" +
                    std::to_string(request.request_id.value()));
            }
            if (request.think_time.seconds() != 0.0) {
                throw WorkloadError("a non-session request must have "
                                    "think_time=0; request_id=" +
                                    std::to_string(request.request_id.value()));
            }
            continue;
        }

        const SessionId::ValueType session_id = request.session_id.value();
        const bool first_turn = seen_sessions.insert(session_id).second;
        if (first_turn) {
            if (!request.session_start_at.valid()) {
                throw WorkloadError(
                    "the first turn of a session requires session_start_at; "
                    "session_id=" +
                    std::to_string(session_id));
            }
            if (request.think_time.seconds() != 0.0) {
                throw WorkloadError(
                    "the first turn of a session must have think_time=0; "
                    "session_id=" +
                    std::to_string(session_id));
            }
        } else if (request.session_start_at.valid()) {
            throw WorkloadError(
                "a successor turn must omit session_start_at and use "
                "think_time; session_id=" +
                std::to_string(session_id));
        }

        if (request.session_turn_index.has_value()) {
            const auto last_turn = last_turn_by_session.find(session_id);
            if (last_turn != last_turn_by_session.end() &&
                request.session_turn_index.value() <= last_turn->second) {
                throw WorkloadError("session_turn_index must be strictly "
                                    "increasing; session_id=" +
                                    std::to_string(session_id));
            }
            last_turn_by_session[session_id] =
                request.session_turn_index.value();
        }
    }
}

} // namespace

std::vector<WorkloadRequest> parse_workload_csv(std::string_view csv_text) {
    std::istringstream input{std::string{csv_text}};
    input.imbue(std::locale::classic());

    std::string line;
    if (!std::getline(input, line)) {
        throw WorkloadError("workload CSV is empty");
    }
    if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
        line.erase(0, 3);
    }

    const std::vector<std::string_view> header_fields = split_row(line, 1);
    std::unordered_map<std::string, std::size_t> column_indices;
    column_indices.reserve(header_fields.size());
    for (std::size_t index = 0; index < header_fields.size(); ++index) {
        const std::string column{header_fields[index]};
        if (column.empty()) {
            throw WorkloadError("workload CSV contains an empty header column");
        }
        if (column == "block_hash_ids") {
            throw WorkloadError(
                "block_hash_ids is outside the C++ MVP; use session_id");
        }
        if (!column_indices.emplace(column, index).second) {
            throw WorkloadError("workload CSV contains duplicate column '" +
                                column + "'");
        }
    }

    const std::unordered_set<std::string> allowed_columns{
        std::string{kSessionStartAt},   std::string{kThinkTime},
        std::string{kNumPrefillTokens}, std::string{kNumDecodeTokens},
        std::string{kSessionId},        std::string{kSessionTurnIndex},
    };
    for (const auto &[column, index] : column_indices) {
        static_cast<void>(index);
        if (allowed_columns.find(column) == allowed_columns.end()) {
            throw WorkloadError("workload CSV contains unknown column '" +
                                column + "'");
        }
    }

    for (const std::string_view required :
         {kSessionStartAt, kThinkTime, kNumPrefillTokens, kNumDecodeTokens}) {
        if (column_indices.find(std::string{required}) ==
            column_indices.end()) {
            throw WorkloadError("workload CSV is missing required column '" +
                                std::string{required} + "'");
        }
    }

    std::vector<WorkloadRequest> requests;
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (trim(line).empty()) {
            continue;
        }
        const std::vector<std::string_view> fields =
            split_row(line, line_number);
        if (fields.size() != header_fields.size()) {
            throw WorkloadError(
                "workload CSV field count does not match the header at line " +
                std::to_string(line_number));
        }

        const std::string_view session_start_text =
            field_at(fields, column_indices, kSessionStartAt);
        std::optional<double> session_start_at;
        if (!session_start_text.empty()) {
            session_start_at = parse_finite_double(
                session_start_text, kSessionStartAt, line_number);
        }
        if (session_start_at.has_value() && session_start_at.value() < 0.0) {
            throw WorkloadError(
                "session_start_at must be nonnegative at line " +
                std::to_string(line_number));
        }
        const double think_time =
            parse_finite_double(field_at(fields, column_indices, kThinkTime),
                                kThinkTime, line_number);
        if (think_time < 0.0) {
            throw WorkloadError("think_time must be nonnegative at line " +
                                std::to_string(line_number));
        }

        std::optional<std::uint64_t> session_id;
        if (column_indices.find(std::string{kSessionId}) !=
            column_indices.end()) {
            session_id = parse_optional_nonnegative_integer(
                field_at(fields, column_indices, kSessionId), kSessionId,
                line_number);
        }

        std::optional<std::uint64_t> session_turn_index;
        if (column_indices.find(std::string{kSessionTurnIndex}) !=
            column_indices.end()) {
            session_turn_index = parse_optional_nonnegative_integer(
                field_at(fields, column_indices, kSessionTurnIndex),
                kSessionTurnIndex, line_number);
        }
        if (session_turn_index.has_value() && !session_id.has_value()) {
            throw WorkloadError(
                "session_turn_index requires session_id at line " +
                std::to_string(line_number));
        }
        if (session_id.has_value() &&
            session_id.value() >
                static_cast<std::uint64_t>(
                    std::numeric_limits<SessionId::ValueType>::max())) {
            throw WorkloadError(
                "session_id is outside the supported range at line " +
                std::to_string(line_number));
        }

        requests.push_back([&]() {
            WorkloadRequest value{};
            value.request_id =
                RequestId{static_cast<RequestId::ValueType>(requests.size())};
            value.session_start_at =
                session_start_at.has_value()
                    ? SimTime::from_seconds(session_start_at.value())
                    : SimTime{};
            value.think_time = SimTime::from_seconds(think_time);
            value.num_prefill_tokens = parse_positive_integer(
                field_at(fields, column_indices, kNumPrefillTokens),
                kNumPrefillTokens, line_number);
            value.num_decode_tokens = parse_positive_integer(
                field_at(fields, column_indices, kNumDecodeTokens),
                kNumDecodeTokens, line_number);
            value.session_id =
                session_id.has_value()
                    ? SessionId{static_cast<SessionId::ValueType>(
                          session_id.value())}
                    : SessionId{};
            value.session_turn_index = session_turn_index;
            return value;
        }());
    }

    validate_session_timing(requests);
    return requests;
}

std::string
serialize_workload_csv(const std::vector<WorkloadRequest> &requests) {
    validate_session_timing(requests);
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "session_start_at,think_time,num_prefill_tokens,"
              "num_decode_tokens,session_id,session_turn_index\n";

    for (std::size_t index = 0; index < requests.size(); ++index) {
        const WorkloadRequest &request = requests[index];
        if (!request.request_id.valid() ||
            request.request_id.index() != index) {
            throw WorkloadError(
                "workload request IDs must be contiguous and start at zero");
        }
        if (!request.session_start_at.valid() &&
            !is_missing_time(request.session_start_at)) {
            throw WorkloadError("cannot serialize an invalid session start");
        }
        if (!request.think_time.valid()) {
            throw WorkloadError("cannot serialize an invalid think time");
        }
        if (request.num_prefill_tokens == 0 || request.num_decode_tokens == 0) {
            throw WorkloadError("cannot serialize nonpositive token counts");
        }
        if (request.session_turn_index.has_value() &&
            !request.session_id.valid()) {
            throw WorkloadError("session_turn_index requires session_id");
        }

        if (request.session_start_at.valid()) {
            output << request.session_start_at.seconds();
        }
        output << ',' << request.think_time.seconds() << ','
               << request.num_prefill_tokens << ',' << request.num_decode_tokens
               << ',';
        if (request.session_id.valid()) {
            output << request.session_id.value();
        }
        output << ',';
        if (request.session_turn_index.has_value()) {
            output << request.session_turn_index.value();
        }
        output << '\n';
    }
    return output.str();
}

void validate_workload_for_config(const std::vector<WorkloadRequest> &requests,
                                  const config::SimulationConfig &config) {
    validate_session_timing(requests);
    for (const WorkloadRequest &request : requests) {
        if (config.prefix_cache.enabled && !request.session_id.valid()) {
            throw WorkloadError("session_id is required when session prefix "
                                "caching is enabled; "
                                "request_id=" +
                                std::to_string(request.request_id.value()));
        }
    }
}

std::vector<WorkloadRequest>
materialize_workload_for_config(const std::vector<WorkloadRequest> &raw,
                                const config::SimulationConfig &config) {
    validate_workload_for_config(raw, config);
    if (!config.prefix_cache.enabled) {
        return raw;
    }

    std::unordered_map<SessionId::ValueType, std::uint64_t>
        context_tokens_by_session;
    std::vector<WorkloadRequest> result = raw;
    for (WorkloadRequest &request : result) {
        std::uint64_t &context =
            context_tokens_by_session[request.session_id.value()];
        if (context > std::numeric_limits<std::uint64_t>::max() -
                          request.num_prefill_tokens) {
            throw WorkloadError(
                "effective session prefill token count overflows uint64; "
                "session_id=" +
                std::to_string(request.session_id.value()));
        }
        request.num_prefill_tokens += context;
        if (request.num_prefill_tokens >
            std::numeric_limits<std::uint64_t>::max() -
                request.num_decode_tokens) {
            throw WorkloadError(
                "resulting session context token count overflows uint64; "
                "session_id=" +
                std::to_string(request.session_id.value()));
        }
        context = request.num_prefill_tokens + request.num_decode_tokens;
    }
    return result;
}

} // namespace frontier::request_generator
