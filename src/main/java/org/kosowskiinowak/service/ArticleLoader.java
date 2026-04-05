package org.kosowskiinowak.service;

import org.kosowskiinowak.model.SingleArticle;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.nio.charset.StandardCharsets;
import java.util.stream.Stream;

public class ArticleLoader {

    // Example tag: <REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="5544" NEWID="1">
    private static final Pattern REUTERS_PATTERN = Pattern.compile("<REUTERS.*?>(.*?)</REUTERS>", Pattern.DOTALL);

    private static final Pattern PLACES_PATTERN = Pattern.compile("<PLACES>(.*?)</PLACES>", Pattern.DOTALL);
    private static final Pattern D_TAG_PATTERN = Pattern.compile("<D>(.*?)</D>");
    private static final Pattern BODY_PATTERN = Pattern.compile("<BODY>(.*?)</BODY>", Pattern.DOTALL);
    private static final Set<String> ALLOWED_COUNTRIES = Set.of(
            "west-germany", "usa", "france", "uk", "canada", "japan"
    );


    public List<SingleArticle> loadArticles(String directoryPath) {
        List<SingleArticle> validArticles = new ArrayList<>();

        try (Stream<Path> pathStream = Files.list(Paths.get(directoryPath))) {
            List<Path> sgmFiles = pathStream
                    .filter(path -> path.toString().endsWith(".sgm"))
                    .sorted()
                    .toList();

            for (Path file : sgmFiles) {
                String fileContent = Files.readString(file, StandardCharsets.ISO_8859_1);
                validArticles.addAll(parseFileContent(fileContent));
            }

        } catch (IOException e) {
            System.err.println("Error reading files: " + e.getMessage());
        }

        System.out.println("Loaded " + validArticles.size() + " articles");
        return validArticles;
    }

    private List<SingleArticle> parseFileContent(String fileContent) {
        List<SingleArticle> articlesFromFile = new ArrayList<>();
        Matcher reutersMatcher = REUTERS_PATTERN.matcher(fileContent);

        while (reutersMatcher.find()) {
            String articleRawText = reutersMatcher.group(1);

            String countryLabel = extractValidCountry(articleRawText);

            if (countryLabel != null) {
                String body = extractTag(articleRawText, BODY_PATTERN);

                if (!body.isBlank()) {
                    articlesFromFile.add(new SingleArticle(countryLabel, body));
                }
            }
        }
        return articlesFromFile;
    }

    private String extractValidCountry(String articleRawText) {
        Matcher placesMatcher = PLACES_PATTERN.matcher(articleRawText);
        if (placesMatcher.find()) {
            String placesContent = placesMatcher.group(1);

            Matcher dTagMatcher = D_TAG_PATTERN.matcher(placesContent);
            List<String> foundCountries = new ArrayList<>();

            while (dTagMatcher.find()) {
                foundCountries.add(dTagMatcher.group(1));
            }

            if (foundCountries.size() == 1) {
                String singleCountry = foundCountries.getFirst();
                if (ALLOWED_COUNTRIES.contains(singleCountry)) {
                    return singleCountry;
                }
            }
        }
        return null;
    }

    private String extractTag(String text, Pattern pattern) {
        Matcher matcher = pattern.matcher(text);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }
        return "";
    }
}
