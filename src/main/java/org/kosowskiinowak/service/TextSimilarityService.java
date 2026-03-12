package org.kosowskiinowak.service;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

public class TextSimilarityService {

    public double calculateTextDistance(String s1, String s2, int n) {
        return 1.0 - calculateNGramSimilarity(s1, s2, n);
    }

    private double calculateNGramSimilarity(String s1, String s2, int n) {
        if (s1 == null || s2 == null) return 0.0;
        if (s1.equals(s2)) return 1.0;
        if (s1.length() < 2 || s2.length() < 2) return 0.0;

        Set<String> nGrams1 = generateNGrams(s1, n);
        Set<String> nGrams2 = generateNGrams(s2, n);

        int initialSize1 = nGrams1.size();
        int initialSize2 = nGrams2.size();

        Set<String> intersection = nGrams1.stream()
                .filter(nGrams2::contains)
                .collect(Collectors.toSet());
        int common = intersection.size();

        // Sorensena-Dice'a
        return (2.0 * common) / (initialSize1 + initialSize2);
    }

    private Set<String> generateNGrams(String text, int n) {
        Set<String> nGrams = new HashSet<>();
        for (int i = 0; i <= text.length() - n; i++) {
            nGrams.add(text.substring(i, i + n));
        }
        return nGrams;
    }
}
