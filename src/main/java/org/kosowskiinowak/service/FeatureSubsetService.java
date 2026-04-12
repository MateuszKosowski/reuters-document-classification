package org.kosowskiinowak.service;

import org.kosowskiinowak.model.FeatureName;
import org.kosowskiinowak.model.FeatureVector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class FeatureSubsetService {

    public List<EnumSet<FeatureName>> generateFeatureSubsets() {
        FeatureName[] features = FeatureName.values();
        List<EnumSet<FeatureName>> subsets = new ArrayList<>();

        int maxMask = (1 << features.length) - 1;
        for (int mask = 1; mask <= maxMask; mask++) {
            int featureCount = Integer.bitCount(mask);
            if (featureCount == features.length) {
                continue;
            }

            EnumSet<FeatureName> subset = EnumSet.noneOf(FeatureName.class);
            for (int bitIndex = 0; bitIndex < features.length; bitIndex++) {
                if ((mask & (1 << bitIndex)) != 0) {
                    subset.add(features[bitIndex]);
                }
            }

            subsets.add(subset);
        }

        subsets.sort(Comparator
                .<EnumSet<FeatureName>>comparingInt(Set::size)
                .reversed()
                .thenComparing(this::formatFeatureSet));

        return subsets;
    }

    public List<FeatureVector> maskDataset(List<FeatureVector> dataset, Set<FeatureName> activeFeatures) {
        return dataset.stream()
                .map(vector -> maskVector(vector, activeFeatures))
                .toList();
    }

    public FeatureVector maskVector(FeatureVector vector, Set<FeatureName> activeFeatures) {
        return new FeatureVector(
                vector.label(),
                activeFeatures.contains(FeatureName.LONGEST_WORD) ? safeText(vector.longestWord()) : "",
                activeFeatures.contains(FeatureName.MOST_FREQUENT_WORD) ? safeText(vector.mostFrequentWord()) : "",
                activeFeatures.contains(FeatureName.AVERAGE_WORD_LENGTH) ? vector.averageWordLength() : 0.0,
                activeFeatures.contains(FeatureName.VOCABULARY_RICHNESS) ? vector.vocabularyRichness() : 0.0,
                activeFeatures.contains(FeatureName.AVERAGE_SENTENCE_LENGTH) ? vector.averageSentenceLength() : 0.0,
                activeFeatures.contains(FeatureName.UPPERCASE_LETTER_RATIO) ? vector.uppercaseLetterRatio() : 0.0,
                activeFeatures.contains(FeatureName.FINANCIAL_SIGN_DENSITY) ? vector.financialSignDensity() : 0.0,
                activeFeatures.contains(FeatureName.FLESCH_READING_EASE_INDEX) ? vector.fleschReadingEaseIndex() : 0.0,
                activeFeatures.contains(FeatureName.VOWEL_TO_CONSONANT_RATIO) ? vector.vowelToConsonantRatio() : 0.0,
                activeFeatures.contains(FeatureName.SUM_OF_ALL_NUMERIC_VALUES) ? vector.sumOfAllNumericValues() : 0.0
        );
    }

    public boolean isFullFeatureSet(Set<FeatureName> activeFeatures) {
        return activeFeatures.size() == FeatureName.values().length;
    }

    public String formatFeatureSet(Set<FeatureName> activeFeatures) {
        return activeFeatures.stream()
                .sorted(Comparator.comparingInt(Enum::ordinal))
                .map(FeatureName::csvLabel)
                .collect(Collectors.joining(","));
    }

    private String safeText(String value) {
        return value == null ? "" : value;
    }
}
