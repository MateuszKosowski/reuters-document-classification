package org.kosowskiinowak.classifier.metric;

import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.service.TextSimilarityService;

public class ChebyshevMetric implements Metric {

    private final TextSimilarityService textSimilarityService;

    public ChebyshevMetric(TextSimilarityService textSimilarityService) {
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

        double max = textSimilarityService.calculateTextDistance(a.longestWord(), b.longestWord(), 2);
        double d;
        d = textSimilarityService.calculateTextDistance(a.mostFrequentWord(), b.mostFrequentWord(), 2);
        if (d > max) max = d;
        d = Math.abs(a.averageWordLength() - b.averageWordLength());
        if (d > max) max = d;
        d = Math.abs(a.vocabularyRichness() - b.vocabularyRichness());
        if (d > max) max = d;
        d = Math.abs(a.averageSentenceLength() - b.averageSentenceLength());
        if (d > max) max = d;
        d = Math.abs(a.uppercaseLetterRatio() - b.uppercaseLetterRatio());
        if (d > max) max = d;
        d = Math.abs(a.financialSignDensity() - b.financialSignDensity());
        if (d > max) max = d;
        d = Math.abs(a.fleschReadingEaseIndex() - b.fleschReadingEaseIndex());
        if (d > max) max = d;
        d = Math.abs(a.vowelToConsonantRatio() - b.vowelToConsonantRatio());
        if (d > max) max = d;
        d = Math.abs(a.sumOfAllNumericValues() - b.sumOfAllNumericValues());
        if (d > max) max = d;
        return max;
    }
}
