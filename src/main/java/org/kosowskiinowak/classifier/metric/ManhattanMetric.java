package org.kosowskiinowak.classifier.metric;

import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.service.TextSimilarityService;

public class ManhattanMetric implements Metric {

    private final TextSimilarityService textSimilarityService;

    public ManhattanMetric(TextSimilarityService textSimilarityService) {
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

        double longestWordDist = textSimilarityService.calculateTextDistance(a.longestWord(), b.longestWord(), 2);
        double mostFrequentWordDist = textSimilarityService.calculateTextDistance(a.mostFrequentWord(), b.mostFrequentWord(), 2);

        return longestWordDist +
                mostFrequentWordDist +
                Math.abs(a.averageWordLength() - b.averageWordLength()) +
                Math.abs(a.vocabularyRichness() - b.vocabularyRichness()) +
                Math.abs(a.averageSentenceLength() - b.averageSentenceLength()) +
                Math.abs(a.uppercaseLetterRatio() - b.uppercaseLetterRatio()) +
                Math.abs(a.financialSignDensity() - b.financialSignDensity()) +
                Math.abs(a.fleschReadingEaseIndex() - b.fleschReadingEaseIndex()) +
                Math.abs(a.vowelToConsonantRatio() - b.vowelToConsonantRatio()) +
                Math.abs(a.sumOfAllNumericValues() - b.sumOfAllNumericValues());
    }
}
