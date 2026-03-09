package org.kosowskiinowak;

import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.model.SingleArticle;
import org.kosowskiinowak.service.ArticleLoader;
import org.kosowskiinowak.service.FeatureVectorExtractor;

import java.io.IOException;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        String dataSource = "src/main/resources/reuters21578";
        ArticleLoader articleLoader = new ArticleLoader();
        List<SingleArticle> rawArticles = articleLoader.loadArticles(dataSource);

        FeatureVectorExtractor extractor = new FeatureVectorExtractor();

        System.out.println("\n--- Wektory cech dla pierwszych 5 artykułów ---\n");

        int limit = Math.min(rawArticles.size(), 5);
        for (int i = 0; i < limit; i++) {
            try {
                SingleArticle article = rawArticles.get(i);
                FeatureVector vector = extractor.extractAllFeaturesFromArticle(article);

                displayVector(i + 1, vector);

            } catch (IOException e) {
                System.err.println("Błąd podczas ekstrakcji cech: " + e.getMessage());
            }
        }
    }

    private static void displayVector(int id, FeatureVector v) {
        System.out.println("Artykuł #" + id + " [" + v.label().toUpperCase() + "]");
        System.out.println("  [Tekstowe]");
        System.out.println("    Najdłuższe słowo:   " + v.longestWord());
        System.out.println("    Najczęstsze słowo:  " + v.mostFrequentWord());
        System.out.println("  [Liczbowe]");
        System.out.printf("    Śr. dł. słowa:      %.2f\n", v.averageWordLength());
        System.out.printf("    Bogactwo słown.:    %.2f\n", v.vocabularyRichness());
        System.out.printf("    Śr. dł. zdania:     %.2f\n", v.averageSentenceLength());
        System.out.printf("    Wielkie litery:     %.4f\n", v.uppercaseLetterRatio());
        System.out.printf("    Znaki finansowe:    %.4f\n", v.financialSignDensity());
        System.out.printf("    Indeks Flescha:     %.2f\n", v.fleschReadingEaseIndex());
        System.out.printf("    Samogł./Spółgł.:    %.2f\n", v.vowelToConsonantRatio());
        System.out.printf("    Suma liczb:         %.2f\n", v.sumOfAllNumericValues());
        System.out.println("--------------------------------------------------");
    }
}