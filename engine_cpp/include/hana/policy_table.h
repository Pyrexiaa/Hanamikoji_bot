#pragma once

// Writing policy tables from C++, in the formats Python's PolicyTableAgent
// reads (engine/policy_table.py and engine/binary_policy_table.py):
//
//   write_policy_table_json   policy_table_v1 JSON, keyed by RoundInfoState::to_string()
//   BinaryPolicyTableWriter   binary_policy_table_v1, keyed by to_string() or by
//                             RoundInfoState::compact_key(); for large tables
//
// A table maps information-state keys to distributions over actions. It may be
// partial: whoever loads it decides what to play at information sets it does
// not list. Binary layout (see engine/binary_policy_table.py for the full
// description): a one-line JSON header, then per entry
//     u32 key length, key bytes, u8 number of actions,
//     per action: u8 action type, u8 number of cards, u8 suit values,
//                 u8 choice (255 = none), f64 probability
// with integers and doubles little-endian.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "action.h"
#include "enums.h"
#include "suit.h"

namespace hana {

// Key encodings: what the keys of a table are.
inline constexpr const char* POLICY_TABLE_KEY_ENCODING         = "round_info_state_str_v1";
inline constexpr const char* MARKOV_POLICY_TABLE_KEY_ENCODING  = "markov_round_info_state_str_v1";
inline constexpr const char* COMPACT_POLICY_TABLE_KEY_ENCODING = "round_info_state_compact_v1";

// One distribution: (action, probability) pairs; probabilities sum to 1.
using PolicyDistribution = std::vector<std::pair<Action, double>>;

inline std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Write a policy_table_v1 JSON artifact with string keys (to_string(), or
// markov keys for MARKOV_POLICY_TABLE_KEY_ENCODING). Probabilities use %.17g,
// which round-trips doubles exactly; keys are written in sorted order, as
// Python's json.dump(sort_keys=True) does.
inline void write_policy_table_json(const std::string& path,
                                    const std::map<std::string, PolicyDistribution>& table,
                                    const std::string& variant,
                                    const char* key_encoding = POLICY_TABLE_KEY_ENCODING) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::fprintf(f, "{\n  \"entries\": {");
    bool first_key = true;
    for (const auto& [key, dist] : table) {
        std::fprintf(f, "%s\n    \"%s\": [", first_key ? "" : ",", json_escape(key).c_str());
        first_key = false;
        for (std::size_t j = 0; j < dist.size(); j++) {
            const Action& a = dist[j].first;
            std::fprintf(f, "%s\n      {\"action_type\": \"%s\", \"cards\": [",
                         j ? "," : "", action_type_value_str(a.action_type));
            for (int c = 0; c < a.suit_count; c++)
                std::fprintf(f, "%s%d", c ? ", " : "", suit_value(a.suits[c]));
            if (a.choice == -1) std::fprintf(f, "], \"choice\": null");
            else                std::fprintf(f, "], \"choice\": %d", static_cast<int>(a.choice));
            std::fprintf(f, ", \"prob\": %.17g}", dist[j].second);
        }
        std::fprintf(f, "\n    ]");
    }
    std::fprintf(f, "\n  },\n  \"format\": \"policy_table_v1\",\n"
                    "  \"key_encoding\": \"%s\",\n"
                    "  \"variant\": \"%s\"\n}\n",
                 key_encoding, json_escape(variant).c_str());
    std::fclose(f);
}

// Writes a binary_policy_table_v1 file one entry at a time, so a table never
// has to be held in memory as a whole:
//
//     hana::BinaryPolicyTableWriter writer("strategy.bin",
//                                          hana::COMPACT_POLICY_TABLE_KEY_ENCODING, "medium_hana");
//     writer.add(info_state.compact_key(), {{action, 0.25}, {other_action, 0.75}});
//     writer.close();   // or let the destructor close it
//
// Keys must be unique; entries may come in any order.
class BinaryPolicyTableWriter {
public:
    BinaryPolicyTableWriter(const std::string& path, const std::string& key_encoding,
                            const std::string& variant)
        : file_(std::fopen(path.c_str(), "wb")) {
        if (!file_) throw std::runtime_error("cannot open " + path);
        // The same header line as Python's json.dumps(header, sort_keys=True).
        const std::string header = "{\"format\": \"binary_policy_table_v1\", \"key_encoding\": \"" +
                                   json_escape(key_encoding) + "\", \"variant\": \"" +
                                   json_escape(variant) + "\"}\n";
        write(header.data(), header.size());
    }

    BinaryPolicyTableWriter(const BinaryPolicyTableWriter&) = delete;
    BinaryPolicyTableWriter& operator=(const BinaryPolicyTableWriter&) = delete;

    ~BinaryPolicyTableWriter() { if (file_) std::fclose(file_); }

    void add(const std::string& key, const PolicyDistribution& dist) {
        if (dist.empty() || dist.size() > 255)
            throw std::invalid_argument("a distribution needs 1 to 255 actions");
        put_u32(static_cast<uint32_t>(key.size()));
        write(key.data(), key.size());
        put_u8(static_cast<uint8_t>(dist.size()));
        for (const auto& [action, prob] : dist) {
            put_u8(static_cast<uint8_t>(action.action_type));
            put_u8(static_cast<uint8_t>(action.suit_count));
            for (int c = 0; c < action.suit_count; c++)
                put_u8(static_cast<uint8_t>(suit_value(action.suits[c])));
            put_u8(action.choice == -1 ? 255 : static_cast<uint8_t>(action.choice));
            put_f64(prob);
        }
        count_++;
    }

    void close() {
        if (file_ && std::fclose(file_) != 0) {
            file_ = nullptr;
            throw std::runtime_error("error closing the policy table file");
        }
        file_ = nullptr;
    }

    // Entries added so far.
    std::size_t size() const { return count_; }

private:
    FILE* file_;
    std::size_t count_ = 0;

    void write(const void* data, std::size_t n) {
        if (!file_) throw std::logic_error("BinaryPolicyTableWriter is closed");
        if (std::fwrite(data, 1, n, file_) != n) throw std::runtime_error("write failed");
    }
    void put_u8(uint8_t v) { write(&v, 1); }
    void put_u32(uint32_t v) {
        const uint8_t b[4] = {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)};
        write(b, 4);
    }
    void put_f64(double v) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        uint8_t b[8];
        for (int i = 0; i < 8; i++) b[i] = uint8_t(bits >> (8 * i));
        write(b, 8);
    }
};

}  // namespace hana
