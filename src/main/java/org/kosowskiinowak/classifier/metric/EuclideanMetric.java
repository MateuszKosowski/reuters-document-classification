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

        double longestWord = textSimilarityService.calculateTextDistance(a.longestWord(), b.longestWord(), 2);
        double mostFrequentWord = textSimilarityService.calculateTextDistance(a.mostFrequentWord(), b.mostFrequentWord(), 2);

        return Math.sqrt(
                Math.pow(longestWord, 2) +
                        Math.pow(mostFrequentWord, 2) +
                        Math.pow(a.averageWordLength() - b.averageWordLength(), 2) +
                        Math.pow(a.vocabularyRichness() - b.vocabularyRichness(), 2) +
                        Math.pow(a.averageSentenceLength() - b.averageSentenceLength(), 2) +
                        Math.pow(a.uppercaseLetterRatio() - b.uppercaseLetterRatio(), 2) +
                        Math.pow(a.financialSignDensity() - b.financialSignDensity(), 2) +
                        Math.pow(a.fleschReadingEaseIndex() - b.fleschReadingEaseIndex(), 2) +
                        Math.pow(a.vowelToConsonantRatio() - b.vowelToConsonantRatio(), 2) +
                        Math.pow(a.sumOfAllNumericValues() - b.sumOfAllNumericValues(), 2)
        );
    }
}
