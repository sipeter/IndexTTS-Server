// text_processor.cpp — Text normalization + BPE tokenization for IndexTTS
#include "text_processor.h"
#include "constants.h"

#include <sentencepiece_processor.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <regex>

namespace indextts {

// ============================================================================
// UTF-8 helpers
// ============================================================================

/// Decode one UTF-8 code point, advance iterator. Returns 0xFFFD on error.
static uint32_t utf8_decode(const char*& p, const char* end) {
    if (p >= end) return 0xFFFD;
    uint8_t c = static_cast<uint8_t>(*p);
    uint32_t cp;
    int extra;
    if (c < 0x80)       { cp = c;          extra = 0; }
    else if (c < 0xC0)  { ++p; return 0xFFFD; }
    else if (c < 0xE0)  { cp = c & 0x1F;   extra = 1; }
    else if (c < 0xF0)  { cp = c & 0x0F;   extra = 2; }
    else if (c < 0xF8)  { cp = c & 0x07;   extra = 3; }
    else                 { ++p; return 0xFFFD; }
    ++p;
    for (int i = 0; i < extra && p < end; ++i, ++p) {
        uint8_t b = static_cast<uint8_t>(*p);
        if ((b & 0xC0) != 0x80) return 0xFFFD;
        cp = (cp << 6) | (b & 0x3F);
    }
    return cp;
}

/// Encode one code point to UTF-8 and append to string.
static void utf8_encode(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

/// Check if a code point is CJK Unified Ideograph (or common CJK ranges).
static bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00  && cp <= 0x9FFF)   // CJK Unified Ideographs
        || (cp >= 0x3400  && cp <= 0x4DBF)   // CJK Extension A
        || (cp >= 0x20000 && cp <= 0x2A6DF)  // CJK Extension B
        || (cp >= 0x2A700 && cp <= 0x2B73F)  // Extension C
        || (cp >= 0x2B740 && cp <= 0x2B81F)  // Extension D
        || (cp >= 0x2B820 && cp <= 0x2CEAF)  // Extension E
        || (cp >= 0xF900  && cp <= 0xFAFF)   // CJK Compatibility Ideographs
        || (cp >= 0x2F800 && cp <= 0x2FA1F); // CJK Compatibility Supplement
}

// ============================================================================
// Character replacement map (from IndexTTS char_rep_map / zh_char_rep_map)
// ============================================================================

static const std::unordered_map<uint32_t, std::string>& get_char_map() {
    // Maps: Chinese punctuation → ASCII equivalents
    static const std::unordered_map<uint32_t, std::string> m = {
        {0xFF1A, ","},   // ：
        {0xFF1B, ","},   // ；
        {0xFF0C, ","},   // ，
        {0x3002, "."},   // 。
        {0xFF01, "!"},   // ！
        {0xFF1F, "?"},   // ？
        {0x00B7, "-"},   // ·
        {0x3001, ","},   // 、
        {0x2026, "..."},// …
        {0x201C, "'"},   // "
        {0x201D, "'"},   // "
        {0x2018, "'"},   // '
        {0x2019, "'"},   // '
        {0xFF08, "'"},   // （
        {0xFF09, "'"},   // ）
        {0x300A, "'"},   // 《
        {0x300B, "'"},   // 》
        {0x3010, "'"},   // 【
        {0x3011, "'"},   // 】
        {0x2014, "-"},   // —
        {0xFF5E, "-"},   // ～
        {0x300C, "'"},   // 「
        {0x300D, "'"},   // 」
        {0xFF04, "."},   // ＄ (full-width dollar)
    };
    return m;
}

// ASCII-level replacements
static const std::unordered_map<char, std::string>& get_ascii_map() {
    static const std::unordered_map<char, std::string> m = {
        {';',  ","},
        {'~',  "-"},
        {'(',  "'"},
        {')',  "'"},
        {'[',  "'"},
        {']',  "'"},
        {':',  ","},
        {'$',  "."},
    };
    return m;
}

// ============================================================================
// Implementation
// ============================================================================

struct TextProcessor::Impl {
    sentencepiece::SentencePieceProcessor sp;
    bool loaded = false;

    // Punctuation token strings that trigger sentence splitting
    std::unordered_set<std::string> split_puncts = {
        ".", "!", "?",
        "\xE2\x96\x81" ".",    // ▁.   (SentencePiece uses ▁ as word boundary)
        "\xE2\x96\x81" "?",    // ▁?
        "\xE2\x96\x81" "...",  // ▁...
    };

    // Comma tokens for secondary splitting
    std::unordered_set<std::string> comma_puncts = {
        ",",
        "\xE2\x96\x81" ",",   // ▁,
    };
};

TextProcessor::TextProcessor() : impl_(std::make_unique<Impl>()) {}
TextProcessor::~TextProcessor() = default;

bool TextProcessor::load(const std::string& bpe_model_path) {
    auto status = impl_->sp.Load(bpe_model_path);
    if (!status.ok()) {
        return false;
    }
    impl_->loaded = true;
    return true;
}

std::string TextProcessor::normalize(const std::string& text) const {
    std::string result;
    result.reserve(text.size() * 2);

    const auto& cmap = get_char_map();
    const auto& amap = get_ascii_map();
    const char* p = text.data();
    const char* end = p + text.size();

    // First: replace 嗯→恩, 呣→母
    std::string temp = text;
    {
        // Simple UTF-8 string replacements
        std::string out;
        out.reserve(temp.size());
        const char* tp = temp.data();
        const char* te = tp + temp.size();
        while (tp < te) {
            const char* start = tp;
            uint32_t cp = utf8_decode(tp, te);
            if (cp == 0x55EF) {        // 嗯
                out += "\xe6\x81\xa9"; // 恩
            } else if (cp == 0x5463) { // 呣
                out += "\xe6\xaf\x8d"; // 母
            } else {
                out.append(start, tp);
            }
        }
        temp = std::move(out);
    }

    // Apply character map
    p = temp.data();
    end = p + temp.size();
    while (p < end) {
        const char* start = p;
        uint32_t cp = utf8_decode(p, end);

        if (cp == '\n') {
            result += ' ';
            continue;
        }

        // Check multi-byte char map
        auto it = cmap.find(cp);
        if (it != cmap.end()) {
            result += it->second;
            continue;
        }

        // Check ASCII map
        if (cp < 128) {
            auto ait = amap.find(static_cast<char>(cp));
            if (ait != amap.end()) {
                result += ait->second;
                continue;
            }
        }

        // Keep as-is
        result.append(start, p);
    }

    // Collapse multiple commas/dots
    // ,,, → …   ...→…
    // Simple post-processing
    std::string cleaned;
    cleaned.reserve(result.size());
    for (size_t i = 0; i < result.size(); ++i) {
        cleaned += result[i];
        if ((result[i] == ',' || result[i] == '.') && i + 2 < result.size()
            && result[i+1] == result[i] && result[i+2] == result[i]) {
            // Replace 3+ consecutive with …
            cleaned.pop_back();
            cleaned += "\xe2\x80\xa6"; // …
            // Skip all consecutive same chars
            char ch = result[i];
            while (i + 1 < result.size() && result[i+1] == ch) ++i;
        }
    }

    return cleaned;
}

std::string TextProcessor::tokenize_by_cjk(const std::string& text) const {
    std::string result;
    result.reserve(text.size() * 2);

    const char* p = text.data();
    const char* end = p + text.size();

    while (p < end) {
        const char* start = p;
        uint32_t cp = utf8_decode(p, end);

        if (is_cjk(cp)) {
            // Insert space before CJK char if not already space
            if (!result.empty() && result.back() != ' ') {
                result += ' ';
            }
            utf8_encode(cp, result);
            // Space after
            result += ' ';
        } else if (cp >= 'a' && cp <= 'z') {
            // Uppercase
            result += static_cast<char>(cp - 'a' + 'A');
        } else {
            // Keep as-is (already upper, digits, punctuation, etc.)
            result.append(start, p);
        }
    }

    // Trim trailing spaces
    while (!result.empty() && result.back() == ' ') result.pop_back();

    return result;
}

TokenIds TextProcessor::encode(const std::string& text) const {
    if (!impl_->loaded) {
        throw std::runtime_error("TextProcessor: BPE model not loaded");
    }
    std::vector<int> ids;
    impl_->sp.Encode(text, &ids);
    return TokenIds(ids.begin(), ids.end());
}

std::vector<TokenIds> TextProcessor::split_sentences(const TokenIds& tokens, int max_tokens) const {
    if (!impl_->loaded) {
        throw std::runtime_error("TextProcessor: BPE model not loaded");
    }

    // Convert each token id back to its piece string for punctuation matching
    std::vector<std::string> pieces;
    pieces.reserve(tokens.size());
    for (auto id : tokens) {
        pieces.push_back(impl_->sp.IdToPiece(id));
    }

    // Find split points at punctuation
    std::vector<size_t> split_points;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (impl_->split_puncts.count(pieces[i])) {
            split_points.push_back(i);
        }
    }

    // Build sentences from split points
    std::vector<TokenIds> sentences;
    size_t start = 0;
    for (auto sp : split_points) {
        if (sp + 1 > start) {
            TokenIds sent(tokens.begin() + start, tokens.begin() + sp + 1);
            sentences.push_back(std::move(sent));
            start = sp + 1;
        }
    }
    // Remaining tokens
    if (start < tokens.size()) {
        TokenIds sent(tokens.begin() + start, tokens.end());
        sentences.push_back(std::move(sent));
    }

    // If any sentence exceeds max_tokens, try splitting by comma
    std::vector<TokenIds> result;
    for (auto& sent : sentences) {
        if (static_cast<int>(sent.size()) <= max_tokens) {
            result.push_back(std::move(sent));
            continue;
        }
        // Secondary split by comma
        std::vector<size_t> comma_splits;
        for (size_t i = 0; i < sent.size(); ++i) {
            std::string piece = impl_->sp.IdToPiece(sent[i]);
            if (impl_->comma_puncts.count(piece)) {
                comma_splits.push_back(i);
            }
        }
        // A comma segment can itself still exceed max_tokens (sparse commas,
        // no sentence punctuation) — the position embedding table is sized to
        // max_tokens, so every emitted segment must be hard-capped.
        auto emit_capped = [&](TokenIds::const_iterator b, TokenIds::const_iterator e) {
            while (b < e) {
                auto len = std::min<std::ptrdiff_t>(max_tokens, e - b);
                result.emplace_back(b, b + len);
                b += len;
            }
        };
        if (!comma_splits.empty()) {
            size_t s = 0;
            for (auto cp : comma_splits) {
                if (cp + 1 > s) {
                    emit_capped(sent.begin() + s, sent.begin() + cp + 1);
                    s = cp + 1;
                }
            }
            if (s < sent.size()) {
                emit_capped(sent.begin() + s, sent.end());
            }
        } else {
            // Hard split by max_tokens
            emit_capped(sent.begin(), sent.end());
        }
    }

    // Merge very short sentences (< 5 tokens) into previous
    if (result.size() > 1) {
        std::vector<TokenIds> merged;
        merged.push_back(std::move(result[0]));
        for (size_t i = 1; i < result.size(); ++i) {
            if (result[i].size() < 5 &&
                static_cast<int>(merged.back().size() + result[i].size()) <= max_tokens) {
                merged.back().insert(merged.back().end(),
                                     result[i].begin(), result[i].end());
            } else {
                merged.push_back(std::move(result[i]));
            }
        }
        result = std::move(merged);
    }

    return result;
}

std::vector<TokenIds> TextProcessor::process(const std::string& text, int max_tokens) const {
    // 1. Normalize
    std::string norm = normalize(text);

    // 2. CJK tokenization + uppercase
    std::string cjk = tokenize_by_cjk(norm);

    // 3. BPE encode
    TokenIds all_ids = encode(cjk);

    // 4. Split into sentences
    return split_sentences(all_ids, max_tokens);
}

} // namespace indextts
