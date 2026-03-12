package org.kosowskiinowak;

import org.kosowskiinowak.classifier.metric.EuclideanMetric;
import org.kosowskiinowak.classifier.metric.Metric;
import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.model.SingleArticle;
import org.kosowskiinowak.service.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        String dataSource = "src/main/resources/reuters21578";
        int k = 5;
        double trainRatio = 0.6;
        long seed = 42;

        // 1. WCZYTYWANIE ARTYKUŁÓW
        ArticleLoader articleLoader = new ArticleLoader();
        List<SingleArticle> rawArticles = articleLoader.loadArticles(dataSource);
        System.out.println("Załadowano surowych artykułów: " + rawArticles.size());

        // 2. EKSTRAKCJA CECH
        FeatureVectorExtractor extractor = new FeatureVectorExtractor();
        List<FeatureVector> rawVectors = new ArrayList<>();

        System.out.print("Ekstrakcja cech... ");
        for (SingleArticle article : rawArticles) {
            try {
                rawVectors.add(extractor.extractAllFeaturesFromArticle(article));
            } catch (IOException e) {
                System.err.println("Błąd ekstrakcji dla artykułu: " + e.getMessage());
            }
        }
        System.out.println("Zakończona (" + rawVectors.size() + " wektorów).");

        // 3. NORMALIZACJA
        NormalizationService normalizationService = new NormalizationService();
        List<FeatureVector> normalizedVectors = normalizationService.normalize(rawVectors);
        System.out.println("Normalizacja wektorów zakończona.");

        // 4. DETERMINISTYCZNY PODZIAŁ
        Collections.shuffle(normalizedVectors, new Random(seed));

        int trainCount = (int) (normalizedVectors.size() * trainRatio);
        List<FeatureVector> trainingSet = normalizedVectors.subList(0, trainCount);
        List<FeatureVector> testSet = normalizedVectors.subList(trainCount, normalizedVectors.size());

        System.out.println("Podział: Uczący=" + trainingSet.size() + ", Testowy=" + testSet.size());

        // 5. KLASYFIKACJA k-NN
        KnnClassifier classifier = new KnnClassifier();

        int kMax = classifier.maxAllowedK(trainingSet);
        if (k < 1 || k > kMax) {
            System.err.println("Nieprawidłowe k! Musi być z zakresu [1, " + kMax + "].");
            return;
        }

        TextSimilarityService textService = new TextSimilarityService();
        Metric metric = new EuclideanMetric(textService);

        System.out.println("Rozpoczynanie klasyfikacji k-NN (k=" + k + ")...");

        int correctPredictions = 0;
        long startTime = System.currentTimeMillis();

        for (FeatureVector testVector : testSet) {
            String actualLabel = testVector.label();
            String predictedLabel = classifier.classify(testVector, trainingSet, k, metric);

            if (predictedLabel.equals(actualLabel)) {
                correctPredictions++;
            }
        }

        long endTime = System.currentTimeMillis();

        // 6. PREZENTACJA WYNIKÓW
        double accuracy = (double) correctPredictions / testSet.size();

        System.out.println("\n--- WYNIKI KLASYFIKACJI ---");
        System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);
        System.out.println("Czas wykonania: " + (endTime - startTime) + " ms");
        System.out.println("Poprawne trafienia: " + correctPredictions + " z " + testSet.size());
    }
}