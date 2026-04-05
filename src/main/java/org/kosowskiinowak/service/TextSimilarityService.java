package org.kosowskiinowak.service;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class TextSimilarityService {

    // N-gram sets are recomputed from scratch for every test×training pair, which
    // causes millions of HashSet allocations per experiment. Cache by (text, n) so
    // each unique word is processed only once across all classifier calls.
    private final ConcurrentHashMap<String, Set<String>> nGramCache = new ConcurrentHashMap<>();

    public double calculateTextDistance(String s1, String s2, int n) {
        return 1.0 - calculateNGramSimilarity(s1, s2, n);
    }

    private double calculateNGramSimilarity(String s1, String s2, int n) {
        if (s1 == null || s2 == null) return 0.0;
        if (s1.equals(s2)) return 1.0;
        if (s1.length() < 2 || s2.length() < 2) return 0.0;

        Set<String> nGrams1 = generateNGrams(s1, n);
        Set<String> nGrams2 = generateNGrams(s2, n);

        int common = 0;
        for (String ngram : nGrams1) {
            if (nGrams2.contains(ngram)) {
                common++;
            }
        }

        // Sørensen–Dice coefficient
        return (2.0 * common) / (nGrams1.size() + nGrams2.size());
    }

    private Set<String> generateNGrams(String text, int n) {
        String cacheKey = n + "\u0000" + text;
        return nGramCache.computeIfAbsent(cacheKey, key -> {
            Set<String> nGrams = new HashSet<>();
            for (int i = 0; i <= text.length() - n; i++) {
                nGrams.add(text.substring(i, i + n));
            }
            return Collections.unmodifiableSet(nGrams);
        });
    }
}
