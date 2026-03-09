package org.kosowskiinowak.service;

import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.model.SingleArticle;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import rita.RiTa;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.awt.SystemColor.text;

public class FeatureVectorExtractor {

    private static final String FINANCIAL_SYMBOLS = "$%¥£&/#*";
    private static final String VOWELS = "aeiouyAEIOUY";
    // \\d+ matches one or more digits
    private static final Pattern NUMBER_PATTERN = Pattern.compile("\\d+([.,]\\d+)?");

    public FeatureVector extractAllFeaturesFromArticle(SingleArticle article) throws IOException {

        Map<String, Integer> wordsCountMap = countWords(article.text());

        String label = article.countryLabel();

        String longestWord = getLongestWord(wordsCountMap);
        String mostFrequentWord = getMostFrequentWord(wordsCountMap);

        double averageWordLength = getAverageWordLength(wordsCountMap);
        double vocabularyRichness = getVocabularyRichness(wordsCountMap);
        double averageSentenceLength = getAverageSentenceLength(article.text());
        double uppercaseLetterRatio = getUppercaseLetterRatio(article.text());
        double financialSignDensity = getFinancialSignDensity(article.text());
        double fleschReadingEaseIndex = getFleschReadingEaseIndex(article.text());
        double vowelToConsonantRatio = getVowelToConsonantRatio(article.text());
        double sumOfAllNumericValues = getSumOfAllNumericValues(article.text());

        FeatureVector vector = new FeatureVector(
                label,
                longestWord,
                mostFrequentWord,
                averageWordLength,
                vocabularyRichness,
                averageSentenceLength,
                uppercaseLetterRatio,
                financialSignDensity,
                fleschReadingEaseIndex,
                vowelToConsonantRatio,
                sumOfAllNumericValues
        );

        return vector;
    }

    /**
     * Finds the most frequent word in the article text using Apache Lucene's EnglishAnalyzer.
     *
     * @param frequencies a map of words and their corresponding frequencies in the article text
     * @return the most frequent word in the article text
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
     * Finds the longest word in the article text using Apache Lucene's EnglishAnalyzer.
     *
     * @param frequencies a map of words and their corresponding frequencies in the article text
     * @return the longest word in the article text
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
     * Calculates the average word length in the article text using Apache Lucene's EnglishAnalyzer.
     *
     * @param frequencies a map of words and their corresponding frequencies in the article text
     * @return the average word length in the article text
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
     * Calculates the vocabulary richness of the article text using Apache Lucene's EnglishAnalyzer.
     *
     * @param frequencies a map of words and their corresponding frequencies in the article text
     * @return the vocabulary richness of the article text
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
     * Calculates the ratio of uppercase letters to total letters in the article text.
     *
     * @param text the article text
     * @return the ratio of uppercase letters to total letters in the article text
     */
    double getUppercaseLetterRatio(String text) {
        long uppercaseCount = text.chars().filter(Character::isUpperCase).count();
        long totalLetters = text.chars().filter(Character::isLetter).count();

        if (totalLetters == 0) {
            return 0.0;
        }

        return (double) uppercaseCount / totalLetters;
    }

    /**
     * Calculates the density of financial signs in the article text.
     *
     * @param text the article text
     * @return the density of financial signs in the article text
     */
    double getFinancialSignDensity(String text) {
        if (text == null || text.isEmpty()) {
            return 0.0;
        }

        long financialSignCount = text.chars()
                .filter(ch -> FINANCIAL_SYMBOLS.indexOf(ch) != -1)
                .count();

        return (double) financialSignCount / text.length();
    }


    /**
     * Calculates the Flesch Reading Ease Index for the article text.
     *
     * @param text the article text
     * @return the Flesch Reading Ease Index for the article text
     */
    double getFleschReadingEaseIndex(String text) {
        // Calculation: 206.835 - 1.015 * (Total Words / Total Sentences) - 84.6 * (Total Syllables / Total Words)

        String[] sentences = text.split("[.!?]+");
        int totalSentences = sentences.length;

        // \\s+ matches one or more whitespace characters
        String[] words = text.toLowerCase().replaceAll("[^a-z ]", "").split("\\s+");
        int totalWords = 0;
        int totalSyllables = 0;

        for (String word : words) {
            if (word.isBlank()) continue;
            totalWords++;

            String syllables = RiTa.syllables(word);
            if (syllables != null && !syllables.isBlank()) {
                totalSyllables += syllables.split("/").length;
            } else {
                totalSyllables += 1;
            }
        }

        if (totalWords == 0) return 0.0;

        return 206.835 - 1.015 * ((double) totalWords / totalSentences) - 84.6 * ((double) totalSyllables / totalWords);
    }

    /**
     * Calculates the ratio of vowels to consonants in the article text.
     *
     * @param text the article text
     * @return the ratio of vowels to consonants in the article text
     */
    double getVowelToConsonantRatio(String text) {
        if (text == null || text.isEmpty()) {
            return 0.0;
        }

        int vowels = 0;
        int consonants = 0;

        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);

            if (Character.isLetter(ch)) {
                if (VOWELS.indexOf(ch) != -1) {
                    vowels++;
                } else {
                    consonants++;
                }
            }
        }

        if (consonants == 0) {
            return (double) vowels;
        }

        return (double) vowels / consonants;
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
            }
        }

        return sum;
    }

    private Map<String, Integer> countWords(String text) throws IOException {
        Map<String, Integer> frequencies = new HashMap<>();

        try (Analyzer analyzer = new EnglishAnalyzer();
             TokenStream tokenStream = analyzer.tokenStream("field", text)) {

            CharTermAttribute termAttr = tokenStream.addAttribute(CharTermAttribute.class);

            tokenStream.reset();

            while (tokenStream.incrementToken()) {
                String word = termAttr.toString();
                frequencies.put(word, frequencies.getOrDefault(word, 0) + 1);
            }

            tokenStream.end();
        }

        return frequencies;
    }


}
