package org.kosowskiinowak.service;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.model.SingleArticle;
import rita.RiTa;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class FeatureVectorExtractor {

    private static final String FINANCIAL_SYMBOLS = "$%¥£&/#*";
    private static final String VOWELS = "aeiouyAEIOUY";
    private static final Pattern NUMBER_PATTERN = Pattern.compile("\\d+([.,]\\d+)?");

    // One EnglishAnalyzer per thread - avoids recreating a heavyweight object per article.
    // EnglishAnalyzer loads stopword lists on construction; reusing it is a significant speedup
    // when feature extraction runs via parallelStream.
    private static final ThreadLocal<Analyzer> THREAD_LOCAL_ANALYZER =
            ThreadLocal.withInitial(EnglishAnalyzer::new);

    // RiTa.syllables() is deterministic and slow - cache results globally across all articles.
    private static final ConcurrentHashMap<String, Integer> SYLLABLE_CACHE =
            new ConcurrentHashMap<>(4096);

    public FeatureVector extractAllFeaturesFromArticle(SingleArticle article) throws IOException {
        String text = article.text();
        Map<String, Integer> wordsCountMap = countWords(text);

        // Single pass over characters replaces three separate stream/loop passes
        // (getUppercaseLetterRatio, getFinancialSignDensity, getVowelToConsonantRatio).
        long[] charStats = computeCharStats(text);
        long uppercaseCount  = charStats[0];
        long totalLetters    = charStats[1];
        long vowelCount      = charStats[2];
        long consonantCount  = charStats[3];
        long financialCount  = charStats[4];

        double uppercaseLetterRatio = totalLetters > 0 ? (double) uppercaseCount / totalLetters : 0.0;
        double financialSignDensity = !text.isEmpty() ? (double) financialCount / text.length() : 0.0;
        double vowelToConsonantRatio = consonantCount > 0 ? (double) vowelCount / consonantCount : vowelCount;

        return new FeatureVector(
                article.countryLabel(),
                getLongestWord(wordsCountMap),
                getMostFrequentWord(wordsCountMap),
                getAverageWordLength(wordsCountMap),
                getVocabularyRichness(wordsCountMap),
                getAverageSentenceLength(text),
                uppercaseLetterRatio,
                financialSignDensity,
                getFleschReadingEaseIndex(text),
                vowelToConsonantRatio,
                getSumOfAllNumericValues(text)
        );
    }

    // [0]=uppercase, [1]=totalLetters, [2]=vowels, [3]=consonants, [4]=financialSigns
    private static long[] computeCharStats(String text) {
        long[] s = new long[5];
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (Character.isLetter(ch)) {
                s[1]++;
                if (Character.isUpperCase(ch)) s[0]++;
                if (VOWELS.indexOf(ch) >= 0) s[2]++;
                else s[3]++;
            } else if (FINANCIAL_SYMBOLS.indexOf(ch) >= 0) {
                s[4]++;
            }
        }
        return s;
    }

    /**
     * Finds the most frequent word in the given word frequency map.
     *
     * @param frequencies a map of stemmed words to their occurrence counts
     * @return the most frequent word, or {@code null} if the map is empty
     */
    String getMostFrequentWord(Map<String, Integer> frequencies) {

        String mostFrequentWord = null;
        int maxCount = 0;

        for (Map.Entry<String, Integer> entry : frequencies.entrySet()) {
            if (entry.getValue() > maxCount) {
                mostFrequentWord = entry.getKey();
                maxCount = entry.getValue();
            }
        }

        return mostFrequentWord;
    }

    /**
     * Finds the longest word in the given word frequency map.
     *
     * @param frequencies a map of stemmed words to their occurrence counts
     * @return the longest word, or {@code null} if the map is empty
     */
    String getLongestWord(Map<String, Integer> frequencies) {
        String longestWord = null;

        for (String word : frequencies.keySet()) {
            if (longestWord == null || word.length() > longestWord.length()) {
                longestWord = word;
            }
        }

        return longestWord;
    }

    /**
     * Calculates the weighted average word length across all occurrences.
     *
     * @param frequencies a map of stemmed words to their occurrence counts
     * @return the average word length, or {@code 0.0} if the map is empty
     */
    double getAverageWordLength(Map<String, Integer> frequencies) {
        int totalWords = frequencies.values().stream().mapToInt(Integer::intValue).sum();
        int totalLength = frequencies.entrySet().stream()
                .mapToInt(entry -> entry.getKey().length() * entry.getValue())
                .sum();

        if (totalWords == 0) {
            return 0.0;
        }

        return (double) totalLength / totalWords;
    }

    /**
     * Calculates vocabulary richness as the ratio of unique words to total words (type-token ratio).
     *
     * @param frequencies a map of stemmed words to their occurrence counts
     * @return a value in [0, 1] where higher means more diverse vocabulary; {@code 0.0} if empty
     */
    double getVocabularyRichness(Map<String, Integer> frequencies) {
        int totalWords = frequencies.values().stream().mapToInt(Integer::intValue).sum();
        int uniqueWords = frequencies.size();

        if (totalWords == 0) {
            return 0.0;
        }

        return (double) uniqueWords / totalWords;
    }

    /**
     * Calculates the average sentence length in the article text.
     *
     * @param text the article text
     * @return the average sentence length in the article text
     */
    double getAverageSentenceLength(String text) {
        String[] sentences = text.split("[.!?]+");
        int totalSentences = sentences.length;
        int totalWords = text.split("\\s+").length;

        if (totalSentences == 0) {
            return 0.0;
        }

        return (double) totalWords / totalSentences;
    }

    /**
     * Calculates the Flesch Reading Ease Index for the article text.
     *
     * @param text the article text
     * @return the Flesch Reading Ease Index for the article text
     */
    double getFleschReadingEaseIndex(String text) {
        if (text == null || text.isBlank()) return 0.0;

        // Calculation: 206.835 - 1.015 * (Total Words / Total Sentences) - 84.6 * (Total Syllables / Total Words)

        String[] sentences = text.split("[.!?]+");
        int totalSentences = Math.max(1, sentences.length);

        // After replaceAll("[^a-z ]"," "), every non-empty token is already [a-z]+,
        // so no further regex check is needed inside the loop.
        String[] words = text.toLowerCase().replaceAll("[^a-z ]", " ").split("\\s+");
        int totalWords = 0;
        int totalSyllables = 0;

        for (String word : words) {
            if (word.isEmpty()) continue;
            totalWords++;
            // Cache RiTa results - same word always yields the same syllable count,
            // and Reuters articles share vocabulary heavily across documents.
            totalSyllables += SYLLABLE_CACHE.computeIfAbsent(word, w -> {
                try {
                    String syllables = RiTa.syllables(w);
                    return (syllables != null && !syllables.isBlank()) ? syllables.split("/").length : 1;
                } catch (Exception e) {
                    return 1;
                }
            });
        }

        if (totalWords == 0) return 0.0;

        return 206.835 - 1.015 * ((double) totalWords / totalSentences) - 84.6 * ((double) totalSyllables / totalWords);
    }

    /**
     * Calculates the sum of all numeric values in the article text.
     *
     * @param text the article text
     * @return the sum of all numeric values in the article text
     */
    double getSumOfAllNumericValues(String text) {
        if (text == null || text.isEmpty()) {
            return 0.0;
        }

        double sum = 0.0;
        Matcher matcher = NUMBER_PATTERN.matcher(text);

        while (matcher.find()) {
            try {
                String match = matcher.group();
                String cleanMatch = match.replace(',', '.');
                sum += Double.parseDouble(cleanMatch);

            } catch (NumberFormatException e) {
                // NUMBER_PATTERN already validates the format; this is a defensive fallback only
            }
        }

        return sum;
    }

    private Map<String, Integer> countWords(String text) throws IOException {
        Map<String, Integer> frequencies = new HashMap<>();
        Analyzer analyzer = THREAD_LOCAL_ANALYZER.get();
        TokenStream tokenStream = analyzer.tokenStream("field", text);
        CharTermAttribute termAttr = tokenStream.addAttribute(CharTermAttribute.class);
        tokenStream.reset();
        while (tokenStream.incrementToken()) {
            String word = termAttr.toString();
            frequencies.merge(word, 1, Integer::sum);
        }
        tokenStream.end();
        tokenStream.close();
        return frequencies;
    }


}
