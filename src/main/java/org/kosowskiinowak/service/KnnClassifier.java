package org.kosowskiinowak.service;

import org.kosowskiinowak.classifier.metric.Metric;
import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.model.Neighbor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class KnnClassifier {

    /**
     * Zwraca maksymalne dopuszczalne k dla danego zbioru treningowego.
     * k nie może przekraczać 2 * liczebność_najmniejszej_klasy + 1,
     * bo wtedy najmniejsza klasa nie może nigdy wygrać głosowania.
     */
    public int maxAllowedK(List<FeatureVector> trainingSet) {
        Map<String, Long> classCounts = trainingSet.stream()
                .collect(Collectors.groupingBy(FeatureVector::label, Collectors.counting()));

        long minClassSize = classCounts.values().stream()
                .min(Long::compare)
                .orElse(1L);

        return (int) (2 * minClassSize + 1);
    }

    public String classify(FeatureVector testVector, List<FeatureVector> trainingSet, int k, Metric metric) {
        if (k < 1) {
            throw new IllegalArgumentException("k musi być >= 1, podano: " + k);
        }

        int kMax = maxAllowedK(trainingSet);
        if (k > kMax) {
            throw new IllegalArgumentException(
                    "k=" + k + " jest za duże. Dla tego zbioru maksymalne sensowne k = " + kMax +
                    " (2 * najmniejsza klasa + 1). Najmniejsza klasa zawiera " + (kMax - 1) / 2 + " elementów."
            );
        }

        List<Neighbor> neighbors = new ArrayList<>();

        for (FeatureVector trainVector : trainingSet) {
            double distance = metric.calculate(testVector, trainVector);
            neighbors.add(new Neighbor(distance, trainVector.label()));
        }

        neighbors.sort(Comparator.comparingDouble(Neighbor::distance));

        List<Neighbor> kNearest = neighbors.subList(0, Math.min(k, neighbors.size()));

        Map<String, Long> votes = kNearest.stream()
                .collect(Collectors.groupingBy(Neighbor::label, Collectors.counting()));

        long maxVotes = votes.values().stream()
                .max(Long::compare)
                .orElse(0L);

        List<String> winners = votes.entrySet().stream()
                .filter(entry -> entry.getValue() == maxVotes)
                .map(Map.Entry::getKey)
                .toList();

        if (winners.size() == 1) {
            return winners.get(0);
        }

        return winners.stream()
                .min(Comparator.comparingDouble(label ->
                        kNearest.stream()
                                .filter(n -> n.label().equals(label))
                                .mapToDouble(Neighbor::distance)
                                .average()
                                .orElse(Double.MAX_VALUE)
                ))
                .orElse(winners.get(0));
    }
}
