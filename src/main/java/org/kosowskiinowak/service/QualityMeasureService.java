package org.kosowskiinowak.service;

import org.kosowskiinowak.model.ClassMetrics;
import org.kosowskiinowak.model.DetailedMetrics;

import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;

public class QualityMeasureService {

    public Map<String, Double> calculateAccuracy(List<ClassificationResult> results) {
        if (results.isEmpty()) return Map.of("Ogólny (Accuracy)", 0.0);
        long correct = results.stream().filter(r -> r.actual().equals(r.predicted())).count();
        return Map.of("Ogólny (Accuracy)", (double) correct / results.size());
    }

    public Map<String, Double> calculatePrecision(List<ClassificationResult> results) {
        Set<String> classes = getAllClasses(results);
        Map<String, Double> precisionMap = new HashMap<>();

        for (String cls : classes) {
            long tp = results.stream().filter(r -> r.actual().equals(cls) && r.predicted().equals(cls)).count();
            long fp = results.stream().filter(r -> !r.actual().equals(cls) && r.predicted().equals(cls)).count();

            double precision = (tp + fp == 0) ? 0.0 : (double) tp / (tp + fp);
            precisionMap.put(cls, precision);
        }

        double averagePrecision = precisionMap.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        precisionMap.put("Średnia makro", averagePrecision);
        return precisionMap;
    }

    public Map<String, Double> calculateRecall(List<ClassificationResult> results) {
        Set<String> classes = getAllClasses(results);
        Map<String, Double> recallMap = new HashMap<>();

        for (String cls : classes) {
            long tp = results.stream().filter(r -> r.actual().equals(cls) && r.predicted().equals(cls)).count();
            long fn = results.stream().filter(r -> r.actual().equals(cls) && !r.predicted().equals(cls)).count();

            double recall = (tp + fn == 0) ? 0.0 : (double) tp / (tp + fn);
            recallMap.put(cls, recall);
        }

        double averageRecall = recallMap.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        recallMap.put("Średnia makro", averageRecall);
        return recallMap;
    }

    public Map<String, Double> calculateF1(Map<String, Double> precisionMap, Map<String, Double> recallMap) {
        Map<String, Double> f1Map = new HashMap<>();

        Set<String> classes = precisionMap.keySet().stream()
                .filter(k -> !k.equals("Średnia makro"))
                .collect(Collectors.toSet());

        for (String cls : classes) {
            double p = precisionMap.getOrDefault(cls, 0.0);
            double r = recallMap.getOrDefault(cls, 0.0);
            double f1 = (p + r == 0) ? 0.0 : 2 * (p * r) / (p + r);
            f1Map.put(cls, f1);
        }

        double averageF1 = f1Map.values().stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);

        f1Map.put("Średnia makro", averageF1);

        return f1Map;
    }

    private Set<String> getAllClasses(List<ClassificationResult> results) {
        Set<String> classes = results.stream().map(ClassificationResult::actual).collect(Collectors.toSet());
        classes.addAll(results.stream().map(ClassificationResult::predicted).collect(Collectors.toSet()));
        return classes;
    }

    public void printReport(List<ClassificationResult> results) {
        Map<String, Double> accuracy = calculateAccuracy(results);
        Map<String, Double> precision = calculatePrecision(results);
        Map<String, Double> recall = calculateRecall(results);
        Map<String, Double> f1 = calculateF1(precision, recall);

        System.out.println("\n--- RAPORT MIAR JAKOŚCI ---");
        System.out.printf("Accuracy (Ogólny): %.2f%%\n", accuracy.get("Ogólny (Accuracy)") * 100);
        System.out.printf("Precision (Makro): %.2f%%\n", precision.get("Średnia makro") * 100);
        System.out.printf("Recall (Makro):    %.2f%%\n", recall.get("Średnia makro") * 100);
        System.out.printf("F1 (Makro):        %.2f%%\n", f1.get("Średnia makro") * 100);

        System.out.println("\nSzczegóły dla klas:");
        System.out.printf("%-15s | %-10s | %-10s | %-10s\n", "Klasa", "Precision", "Recall", "F1");
        System.out.println("-------------------------------------------------------------");

        List<String> classes = precision.keySet().stream()
                .filter(k -> !k.equals("Średnia makro"))
                .sorted()
                .toList();

        for (String cls : classes) {
            System.out.printf("%-15s | %-10.2f%% | %-10.2f%% | %-10.2f%%\n",
                    cls, precision.get(cls) * 100, recall.get(cls) * 100, f1.get(cls) * 100);
        }
    }

    public DetailedMetrics calculateDetailedMetrics(List<ClassificationResult> results) {
        Map<String, Double> accuracyMap = calculateAccuracy(results);
        Map<String, Double> precisionMap = calculatePrecision(results);
        Map<String, Double> recallMap = calculateRecall(results);
        Map<String, Double> f1Map = calculateF1(precisionMap, recallMap);

        Set<String> classNames = new TreeSet<>();
        classNames.addAll(filterClassNames(precisionMap.keySet()));
        classNames.addAll(filterClassNames(recallMap.keySet()));
        classNames.addAll(filterClassNames(f1Map.keySet()));

        List<ClassMetrics> perClassMetrics = classNames.stream()
                .map(className -> new ClassMetrics(
                        className,
                        precisionMap.getOrDefault(className, 0.0),
                        recallMap.getOrDefault(className, 0.0),
                        f1Map.getOrDefault(className, 0.0)
                ))
                .toList();

        double accuracy = extractAccuracyValue(accuracyMap);
        double macroF1 = extractMacroValue(f1Map);

        return new DetailedMetrics(
                accuracy,
                extractMacroValue(precisionMap),
                extractMacroValue(recallMap),
                macroF1,
                harmonicMean(accuracy, macroF1),
                perClassMetrics
        );
    }

    private Set<String> filterClassNames(Set<String> rawKeys) {
        return rawKeys.stream()
                .filter(this::isClassName)
                .collect(Collectors.toCollection(TreeSet::new));
    }

    private boolean isClassName(String key) {
        String normalized = key.toLowerCase(Locale.ROOT);
        return !normalized.contains("macro")
                && !normalized.contains("makro")
                && !normalized.contains("accuracy");
    }

    private double extractAccuracyValue(Map<String, Double> accuracyMap) {
        return accuracyMap.entrySet().stream()
                .filter(entry -> entry.getKey().toLowerCase(Locale.ROOT).contains("accuracy"))
                .map(Map.Entry::getValue)
                .findFirst()
                .orElse(0.0);
    }

    private double extractMacroValue(Map<String, Double> metricMap) {
        return metricMap.entrySet().stream()
                .filter(entry -> {
                    String key = entry.getKey().toLowerCase(Locale.ROOT);
                    return key.contains("macro") || key.contains("makro");
                })
                .map(Map.Entry::getValue)
                .findFirst()
                .orElse(0.0);
    }

    private double harmonicMean(double first, double second) {
        if (first <= 0.0 || second <= 0.0) {
            return 0.0;
        }
        return 2.0 * first * second / (first + second);
    }

    public record ClassificationResult(String actual, String predicted) {
    }
}
