package org.kosowskiinowak.service;

import org.kosowskiinowak.classifier.metric.Metric;
import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.model.Neighbor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.stream.Collectors;

public class KnnClassifier {

    /**
     * Returns a heuristic upper bound for k based on the smallest class size.
     *
     * In k-NN voting, if k exceeds 2 * minClassSize, the minority class can never
     * accumulate enough votes to win against a larger class — it is effectively
     * eliminated from classification. The bound 2 * minClassSize + 1 guarantees
     * that even the smallest class can achieve a majority when all k nearest
     * neighbours belong to it.
     *
     * <p>Note: not called in the experiment pipeline — kept as a utility for
     * manual configuration guidance when choosing the k search range.
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
            throw new IllegalArgumentException("k must be >= 1, got: " + k);
        }

        if (trainingSet == null || trainingSet.isEmpty()) {
            throw new IllegalArgumentException("Training set must not be empty.");
        }

        int effectiveK = Math.min(k, trainingSet.size());
        PriorityQueue<Neighbor> nearestNeighbors = new PriorityQueue<>(
                effectiveK,
                Comparator.comparingDouble(Neighbor::distance).reversed()
        );

        for (FeatureVector trainVector : trainingSet) {
            double distance = metric.calculate(testVector, trainVector);
            Neighbor neighbor = new Neighbor(distance, trainVector.label());

            if (nearestNeighbors.size() < effectiveK) {
                nearestNeighbors.offer(neighbor);
                continue;
            }

            Neighbor farthestNeighbor = nearestNeighbors.peek();
            if (farthestNeighbor != null && distance < farthestNeighbor.distance()) {
                nearestNeighbors.poll();
                nearestNeighbors.offer(neighbor);
            }
        }

        List<Neighbor> kNearest = new ArrayList<>(nearestNeighbors);

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

        // Sort only when tie-breaking is actually needed (rare case)
        kNearest.sort(Comparator.comparingDouble(Neighbor::distance));

        return winners.stream()
                .min(Comparator.comparingDouble(label ->
                        kNearest.stream()
                                .filter(neighbor -> neighbor.label().equals(label))
                                .mapToDouble(Neighbor::distance)
                                .average()
                                .orElse(Double.MAX_VALUE)
                ))
                .orElse(winners.get(0));
    }
}
