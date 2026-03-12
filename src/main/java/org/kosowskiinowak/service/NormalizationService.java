package org.kosowskiinowak.service;

import org.kosowskiinowak.model.FeatureVector;

import java.util.ArrayList;
import java.util.List;

public class NormalizationService {
    public List<FeatureVector> normalize(List<FeatureVector> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            return new ArrayList<>();
        }

        double minAvgWordLen = Double.MAX_VALUE, maxAvgWordLen = Double.NEGATIVE_INFINITY;
        double minVocabRichness = Double.MAX_VALUE, maxVocabRichness = Double.NEGATIVE_INFINITY;
        double minAvgSentLen = Double.MAX_VALUE, maxAvgSentLen = Double.NEGATIVE_INFINITY;
        double minUppercaseRatio = Double.MAX_VALUE, maxUppercaseRatio = Double.NEGATIVE_INFINITY;
        double minFinancialDensity = Double.MAX_VALUE, maxFinancialDensity = Double.NEGATIVE_INFINITY;
        double minFleschIndex = Double.MAX_VALUE, maxFleschIndex = Double.NEGATIVE_INFINITY;
        double minVowelConsonantRatio = Double.MAX_VALUE, maxVowelConsonantRatio = Double.NEGATIVE_INFINITY;
        double minSumNumeric = Double.MAX_VALUE, maxSumNumeric = Double.NEGATIVE_INFINITY;

        for (FeatureVector vector : vectors) {
            minAvgWordLen = Math.min(minAvgWordLen, vector.averageWordLength());
            maxAvgWordLen = Math.max(maxAvgWordLen, vector.averageWordLength());

            minVocabRichness = Math.min(minVocabRichness, vector.vocabularyRichness());
            maxVocabRichness = Math.max(maxVocabRichness, vector.vocabularyRichness());

            minAvgSentLen = Math.min(minAvgSentLen, vector.averageSentenceLength());
            maxAvgSentLen = Math.max(maxAvgSentLen, vector.averageSentenceLength());

            minUppercaseRatio = Math.min(minUppercaseRatio, vector.uppercaseLetterRatio());
            maxUppercaseRatio = Math.max(maxUppercaseRatio, vector.uppercaseLetterRatio());

            minFinancialDensity = Math.min(minFinancialDensity, vector.financialSignDensity());
            maxFinancialDensity = Math.max(maxFinancialDensity, vector.financialSignDensity());

            minFleschIndex = Math.min(minFleschIndex, vector.fleschReadingEaseIndex());
            maxFleschIndex = Math.max(maxFleschIndex, vector.fleschReadingEaseIndex());

            minVowelConsonantRatio = Math.min(minVowelConsonantRatio, vector.vowelToConsonantRatio());
            maxVowelConsonantRatio = Math.max(maxVowelConsonantRatio, vector.vowelToConsonantRatio());

            minSumNumeric = Math.min(minSumNumeric, vector.sumOfAllNumericValues());
            maxSumNumeric = Math.max(maxSumNumeric, vector.sumOfAllNumericValues());
        }

        List<FeatureVector> normalizedVectors = new ArrayList<>();

        for (FeatureVector vector : vectors) {
            double normAvgWordLen = normalize(vector.averageWordLength(), minAvgWordLen, maxAvgWordLen);
            double normVocabRichness = normalize(vector.vocabularyRichness(), minVocabRichness, maxVocabRichness);
            double normAvgSentLen = normalize(vector.averageSentenceLength(), minAvgSentLen, maxAvgSentLen);
            double normUppercaseRatio = normalize(vector.uppercaseLetterRatio(), minUppercaseRatio, maxUppercaseRatio);
            double normFinancialDensity = normalize(vector.financialSignDensity(), minFinancialDensity, maxFinancialDensity);
            double normFleschIndex = normalize(vector.fleschReadingEaseIndex(), minFleschIndex, maxFleschIndex);
            double normVowelConsonantRatio = normalize(vector.vowelToConsonantRatio(), minVowelConsonantRatio, maxVowelConsonantRatio);
            double normSumNumeric = normalize(vector.sumOfAllNumericValues(), minSumNumeric, maxSumNumeric);

            normalizedVectors.add(new FeatureVector(
                    vector.label(),
                    vector.longestWord(),
                    vector.mostFrequentWord(),
                    normAvgWordLen,
                    normVocabRichness,
                    normAvgSentLen,
                    normUppercaseRatio,
                    normFinancialDensity,
                    normFleschIndex,
                    normVowelConsonantRatio,
                    normSumNumeric
            ));
        }

        return normalizedVectors;
    }

    private double normalize(double value, double min, double max) {
        if (max - min == 0) {
            return 0.0;
        }
        return (value - min) / (max - min);
    }
}
