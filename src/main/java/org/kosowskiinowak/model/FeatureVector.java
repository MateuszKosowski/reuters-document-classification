package org.kosowskiinowak.model;

public record FeatureVector
        (
            String label,

            String longestWord,
            String mostFrequentWord,

            double averageWordLength,
            double vocabularyRichness,
            double averageSentenceLength,
            double uppercaseLetterRatio,
            double financialSignDensity,
            double fleschReadingEaseIndex,
            double vowelToConsonantRatio,
            double sumOfAllNumericValues
        ){}
