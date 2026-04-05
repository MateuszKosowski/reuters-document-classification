package org.kosowskiinowak.classifier.metric;

import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.service.TextSimilarityService;

public class EuclideanMetric implements Metric {

    private final TextSimilarityService textSimilarityService;

    public EuclideanMetric(TextSimilarityService textSimilarityService) {
        this.textSimilarityService = textSimilarityService;
    }

    @Override
    public double calculate(FeatureVector a, FeatureVector b) {

        if (a == null || b == null) {
            return 0.0;
        }

        if (a.equals(b)) {
            return 0.0;
        }

        double dLongestWord = textSimilarityService.calculateTextDistance(a.longestWord(), b.longestWord(), 2);
        double dMostFreqWord = textSimilarityService.calculateTextDistance(a.mostFrequentWord(), b.mostFrequentWord(), 2);
        double dAvgLen = a.averageWordLength() - b.averageWordLength();
        double dVocab = a.vocabularyRichness() - b.vocabularyRichness();
        double dSentLen = a.averageSentenceLength() - b.averageSentenceLength();
        double dUppercase = a.uppercaseLetterRatio() - b.uppercaseLetterRatio();
        double dFinancial = a.financialSignDensity() - b.financialSignDensity();
        double dFlesch = a.fleschReadingEaseIndex() - b.fleschReadingEaseIndex();
        double dVowel = a.vowelToConsonantRatio() - b.vowelToConsonantRatio();
        double dNumeric = a.sumOfAllNumericValues() - b.sumOfAllNumericValues();

        return Math.sqrt(
                dLongestWord * dLongestWord +
                dMostFreqWord * dMostFreqWord +
                dAvgLen * dAvgLen +
                dVocab * dVocab +
                dSentLen * dSentLen +
                dUppercase * dUppercase +
                dFinancial * dFinancial +
                dFlesch * dFlesch +
                dVowel * dVowel +
                dNumeric * dNumeric
        );
    }
}
