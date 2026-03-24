package org.kosowskiinowak;

import org.kosowskiinowak.classifier.metric.ChebyshevMetric;
import org.kosowskiinowak.classifier.metric.EuclideanMetric;
import org.kosowskiinowak.classifier.metric.ManhattanMetric;
import org.kosowskiinowak.classifier.metric.Metric;
import org.kosowskiinowak.model.ExperimentParams;
import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.model.SingleArticle;
import org.kosowskiinowak.service.*;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class Main {
    public static void main(String[] args) {
        String dataSource = "src/main/resources/reuters21578";

        // 0. DOSTĘPNE PARAMETRY DO EKSPERYMENTÓW
        long seed = 42;
        List<Integer> listK = IntStream.rangeClosed(1, 40)
                .boxed()
                .toList();
        List<Double> listTrainRatios = List.of(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        List<Metric> listMetrics = List.of(
                new EuclideanMetric(new TextSimilarityService()),
                new ManhattanMetric(new TextSimilarityService()),
                new ChebyshevMetric(new TextSimilarityService())
        );
        List<ExperimentParams> experiments = listK.stream()
                .flatMap(k -> listTrainRatios.stream()
                        .flatMap(trainRatio -> listMetrics.stream()
                                .map(metric -> new ExperimentParams(k, trainRatio, metric))))
                .toList();
        System.out.println("Przygotowano " + experiments.size() + " konfiguracji eksperymentów.");



        // 1. WCZYTYWANIE ARTYKUŁÓW
        ArticleLoader articleLoader = new ArticleLoader();
        List<SingleArticle> rawArticles = articleLoader.loadArticles(dataSource);
        System.out.println("Załadowano surowych artykułów: " + rawArticles.size());

        // 2. EKSTRAKCJA CECH
        System.out.println("Rozpoczynanie ekstrakcji cech (współbieżnie)...");
        FeatureVectorExtractor extractor = new FeatureVectorExtractor();
        List<FeatureVector> rawVectors = rawArticles.parallelStream()
                .map(article -> {
                    try {
                        return extractor.extractAllFeaturesFromArticle(article);
                    } catch (IOException e) {
                        return null;
                    }
                })
                .filter(Objects::nonNull)
                .toList();
        System.out.println("Ekstrakcja zakończona (" + rawVectors.size() + " wektorów).");

        // 3. NORMALIZACJA
        NormalizationService normalizationService = new NormalizationService();
        List<FeatureVector> normalizedVectors = normalizationService.normalize(rawVectors);
        System.out.println("Normalizacja wektorów zakończona.");

        // 4. PRZYGOTOWANIE ZBIORU
        List<FeatureVector> universalDataset = new ArrayList<>(normalizedVectors);
        Collections.shuffle(universalDataset, new Random(seed));

        System.out.println("Rozpoczynanie serii eksperymentów (" + experiments.size() + ")...");
        long globalStartTime = System.currentTimeMillis();
        AtomicInteger progressCounter = new AtomicInteger(0);
        List<String> csvResults = Collections.synchronizedList(new ArrayList<>());
        csvResults.add("K;TrainRatio;Metric;Accuracy;Precision;Recall;F1;DurationMs;Status");

        // 5. URUCHOMIENIE EKSPERYMENTÓW WSPÓŁBIEŻNIE
        experiments.parallelStream().forEach(param -> {
            long startTime = System.currentTimeMillis();

            int k = param.k();
            double trainRatio = param.trainRatio();
            Metric metric = param.metric();
            String metricName = metric.getClass().getSimpleName();

            try {
                int trainCount = (int) (universalDataset.size() * trainRatio);
                List<FeatureVector> trainingSet = universalDataset.subList(0, trainCount);
                List<FeatureVector> testSet = universalDataset.subList(trainCount, universalDataset.size());

                KnnClassifier classifier = new KnnClassifier();

                // Walidacja k względem zbioru treningowego
                int maxK = classifier.maxAllowedK(trainingSet);
                if (k > maxK) {
                    csvResults.add(String.format(Locale.US, "%d;%.2f;%s;0;0;0;0;0;SKIPPED_K_TOO_LARGE_MAX_%d",
                            k, trainRatio, metricName, maxK));
                    return;
                }

                QualityMeasureService qualityService = new QualityMeasureService();
                List<QualityMeasureService.ClassificationResult> results = new ArrayList<>();

                for (FeatureVector testVector : testSet) {
                    String predictedLabel = classifier.classify(testVector, trainingSet, k, metric);
                    results.add(new QualityMeasureService.ClassificationResult(testVector.label(), predictedLabel));
                }

                Map<String, Double> accuracyMap = qualityService.calculateAccuracy(results);
                Map<String, Double> precisionMap = qualityService.calculatePrecision(results);
                Map<String, Double> recallMap = qualityService.calculateRecall(results);
                Map<String, Double> f1Map = qualityService.calculateF1(precisionMap, recallMap);

                double accuracy = accuracyMap.getOrDefault("Ogólny (Accuracy)", 0.0);
                double precision = precisionMap.getOrDefault("Średnia makro", 0.0);
                double recall = recallMap.getOrDefault("Średnia makro", 0.0);
                double f1 = f1Map.getOrDefault("Średnia makro", 0.0);

                long duration = System.currentTimeMillis() - startTime;

                String csvLine = String.format(Locale.US, "%d;%.2f;%s;%.4f;%.4f;%.4f;%.4f;%d;OK",
                        k, trainRatio, metricName, accuracy, precision, recall, f1, duration);

                csvResults.add(csvLine);

            } catch (Exception e) {
                System.err.println("Błąd w eksperymencie [k=" + k + ", ratio=" + trainRatio + "]: " + e.getMessage());
                csvResults.add(String.format(Locale.US, "%d;%.2f;%s;0;0;0;0;0;ERROR", k, trainRatio, metricName));
            } finally {
                int current = progressCounter.incrementAndGet();
                if (current % 50 == 0) {
                    System.out.printf("Ukończono %d / %d eksperymentów (%.1f%%)...\n",
                            current, experiments.size(), (double)current/experiments.size()*100);
                }
            }
        });

        long globalEndTime = System.currentTimeMillis();
        System.out.println("Wszystkie eksperymenty zakończone w czasie: " + (globalEndTime - globalStartTime) + " ms");

        // 6. ZAPIS WYNIKÓW DO PLIKU
        String outputFileName = "wyniki_eksperymentow.csv";
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName))) {
            String header = csvResults.get(0);
            List<String> dataRows = new ArrayList<>(csvResults.subList(1, csvResults.size()));
            Collections.sort(dataRows);

            writer.write(header);
            writer.newLine();
            for (String line : dataRows) {
                writer.write(line);
                writer.newLine();
            }
            System.out.println("Pełny raport zapisano do pliku: " + outputFileName);
        } catch (IOException e) {
            System.err.println("Błąd zapisu wyników: " + e.getMessage());
        }
    }
}