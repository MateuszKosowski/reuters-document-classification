package org.kosowskiinowak.model;

public enum FeatureName {
    LONGEST_WORD("longestWord"),
    MOST_FREQUENT_WORD("mostFrequentWord"),
    AVERAGE_WORD_LENGTH("averageWordLength"),
    VOCABULARY_RICHNESS("vocabularyRichness"),
    AVERAGE_SENTENCE_LENGTH("averageSentenceLength"),
    UPPERCASE_LETTER_RATIO("uppercaseLetterRatio"),
    FINANCIAL_SIGN_DENSITY("financialSignDensity"),
    FLESCH_READING_EASE_INDEX("fleschReadingEaseIndex"),
    VOWEL_TO_CONSONANT_RATIO("vowelToConsonantRatio"),
    SUM_OF_ALL_NUMERIC_VALUES("sumOfAllNumericValues");

    private final String csvLabel;

    FeatureName(String csvLabel) {
        this.csvLabel = csvLabel;
    }

    public String csvLabel() {
        return csvLabel;
    }
}
