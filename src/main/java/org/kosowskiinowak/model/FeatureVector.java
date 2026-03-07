package org.kosowskiinowak.model;

public record FeatureVector
        (
            String label,

            String longestWord,
            String mostFrequentWord,

            double averageWordLength,
            double vocabularyRichness,
            double bodyLength,
            double wordCount,
            double numberDensity,
            double capitalLettersRatio,
            double avgSentenceLength
        ){}
