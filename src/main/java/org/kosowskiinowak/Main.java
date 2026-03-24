package org.kosowskiinowak;

import org.kosowskiinowak.classifier.metric.ChebyshevMetric;
import org.kosowskiinowak.classifier.metric.EuclideanMetric;
import org.kosowskiinowak.classifier.metric.ManhattanMetric;
import org.kosowskiinowak.classifier.metric.Metric;
import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.model.SingleArticle;
import org.kosowskiinowak.service.*;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "%4$s: %5$s%n");
    }

    private static final Logger LOGGER = Logger.getLogger(Main.class.getName());

    public static void main(String[] args) {
        String dataSource = "src/main/resources/reuters21578";

        // 0. DOSTĘPNE PARAMETRY DO EKSPERYMENTÓW
        long seed = 42;
        List<Double> listTrainRatios = List.of(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        List<Metric> listMetrics = List.of(
                new EuclideanMetric(new TextSimilarityService()),
                new ManhattanMetric(new TextSimilarityService()),
                new ChebyshevMetric(new TextSimilarityService())
        );

        // 1. WCZYTYWANIE ARTYKUŁÓW
        ArticleLoader articleLoader = new ArticleLoader();
        List<SingleArticle> rawArticles = articleLoader.loadArticles(dataSource);
        LOGGER.info("Loaded raw articles: " + rawArticles.size());

        // 2. EKSTRAKCJA CECH
        LOGGER.info("Starting feature extraction (concurrently)...");
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
        LOGGER.info("Extraction completed (" + rawVectors.size() + " vectors).");

        // 3. NORMALIZACJA
        NormalizationService normalizationService = new NormalizationService();
        List<FeatureVector> normalizedVectors = normalizationService.normalize(rawVectors);
        LOGGER.info("Vector normalization completed.");

        // 4. PRZYGOTOWANIE ZBIORU
        List<FeatureVector> universalDataset = new ArrayList<>(normalizedVectors);
        Collections.shuffle(universalDataset, new Random(seed));

        record BaseConfig(double trainRatio, Metric metric) {}
        List<BaseConfig> baseConfigs = listTrainRatios.stream()
                .flatMap(trainRatio -> listMetrics.stream()
                        .map(metric -> new BaseConfig(trainRatio, metric)))
                .toList();

        LOGGER.info("Starting base experiments suite (" + baseConfigs.size() + ") - k will be picked dynamically...");
        long globalStartTime = System.currentTimeMillis();
        AtomicInteger progressCounter = new AtomicInteger(0);
        List<String> csvResults = Collections.synchronizedList(new ArrayList<>());
        csvResults.add("K;TrainRatio;Metric;Class;Accuracy;Precision;Recall;F1;DurationMs;Status");

        // 5. URUCHOMIENIE EKSPERYMENTÓW WSPÓŁBIEŻNIE ze zmiennym K
        int patienceLimit = 5; // Ile kolejnych iteracji bez poprawy accuracy pozwala na zakończenie zwiększania K

        baseConfigs.parallelStream().forEach(config -> {
            double trainRatio = config.trainRatio();
            Metric metric = config.metric();
            String metricName = metric.getClass().getSimpleName();

            try {
                int trainCount = (int) (universalDataset.size() * trainRatio);
                List<FeatureVector> trainingSet = universalDataset.subList(0, trainCount);
                List<FeatureVector> testSet = universalDataset.subList(trainCount, universalDataset.size());

                KnnClassifier classifier = new KnnClassifier();
                QualityMeasureService qualityService = new QualityMeasureService();

                int maxPossibleK = Math.min(6, trainingSet.size()); // Szybki test dla k max 6
                double bestAccuracy = -1.0;
                int noImprovementCounter = 0;

                for (int k = 1; k <= maxPossibleK; k++) {
                    long startTime = System.currentTimeMillis();

                    List<QualityMeasureService.ClassificationResult> results = new ArrayList<>();

                    for (FeatureVector testVector : testSet) {
                        String predictedLabel = classifier.classify(testVector, trainingSet, k, metric);
                        results.add(new QualityMeasureService.ClassificationResult(testVector.label(), predictedLabel));
                    }

                    Map<String, Double> accuracyMap = qualityService.calculateAccuracy(results);
                    Map<String, Double> precisionMap = qualityService.calculatePrecision(results);
                    Map<String, Double> recallMap = qualityService.calculateRecall(results);
                    Map<String, Double> f1Map = qualityService.calculateF1(precisionMap, recallMap);

                    double overallAccuracy = accuracyMap.getOrDefault("Ogólny (Accuracy)", 0.0);
                    long duration = System.currentTimeMillis() - startTime;

                    for (String className : precisionMap.keySet()) {
                        if (className.equals("Średnia makro") || className.equals("Ogólny (Accuracy)")) {
                            continue; // Pomijamy sztuczne klasy zbiorcze
                        }

                        double classPrecision = precisionMap.getOrDefault(className, 0.0);
                        double classRecall = recallMap.getOrDefault(className, 0.0);
                        double classF1 = f1Map.getOrDefault(className, 0.0);

                        String csvLine = String.format(Locale.US, "%d;%.2f;%s;%s;%.4f;%.4f;%.4f;%.4f;%d;OK",
                                k, trainRatio, metricName, className, overallAccuracy, classPrecision, classRecall, classF1, duration);

                        csvResults.add(csvLine);
                    }

                    // Sprawdzanie nowej dokładności
                    if (overallAccuracy > bestAccuracy) {
                        bestAccuracy = overallAccuracy;
                        noImprovementCounter = 0; // zresetuj licznik jeśli jest poprawa
                    } else {
                        noImprovementCounter++;
                    }

                    // Early Stopping
                    if (noImprovementCounter >= patienceLimit) {
                        break;
                    }
                }
            } catch (Exception e) {
                LOGGER.log(Level.SEVERE, "Error in experiment [ratio=" + trainRatio + ", metric=" + metricName + "]: " + e.getMessage(), e);
                csvResults.add(String.format(Locale.US, "0;%.2f;%s;0;0;0;0;0;ERROR", trainRatio, metricName));
            } finally {
                int current = progressCounter.incrementAndGet();
                LOGGER.info(String.format(Locale.US, "Completed %d / %d configurations (%.1f%%)...",
                        current, baseConfigs.size(), (double) current / baseConfigs.size() * 100));
            }
        });

        long globalEndTime = System.currentTimeMillis();
        LOGGER.info("All experiments completed in: " + (globalEndTime - globalStartTime) + " ms");

        // 6. ZAPIS WYNIKÓW DO PLIKU
        String outputFileName = "results.csv";
        ResultsExportService exportService = new ResultsExportService();
        exportService.exportToCsv(csvResults, outputFileName);
    }
}