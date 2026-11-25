#pragma once
// Text normalizer + SentencePiece BPE tokenizer for IndexTTS

#include <string>
#include <vector>
#include <memory>

namespace indextts {

/// Holds one sentence as a sequence of BPE token‐ids.
using TokenIds = std::vector<int32_t>;

class TextProcessor {
public:
    TextProcessor();
    ~TextProcessor();

    /// Load SentencePiece model file.
    /// @param bpe_model_path  Path to bpe.model
    /// @return true on success
    bool load(const std::string& bpe_model_path);

    /// Full pipeline: normalize → CJK space insertion → BPE encode → split sentences.
    /// @param text          Raw user input text (UTF-8)
    /// @param max_tokens    Max tokens per sentence (default 100)
    /// @return              Vector of sentences, each a vector of token ids
    std::vector<TokenIds> process(const std::string& text, int max_tokens = 100) const;

    /// Normalize text: character replacements, English contractions, etc.
    std::string normalize(const std::string& text) const;

    /// CJK character tokenization: insert spaces around CJK characters, uppercase all.
    std::string tokenize_by_cjk(const std::string& text) const;

    /// BPE encode text into token id sequence.
    TokenIds encode(const std::string& text) const;

    /// Split token sequence into sentences by punctuation tokens.
    std::vector<TokenIds> split_sentences(const TokenIds& tokens, int max_tokens) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace indextts
