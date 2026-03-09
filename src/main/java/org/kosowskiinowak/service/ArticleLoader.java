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

public class ArticleLoader {

    // Example tag: <REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="5544" NEWID="1">
    // In regex: . means "any character", * means "zero or more times", ? means "lazy search" - stop at first pattern which matches the righthand side
    // () - means capture group - special place to store data, where we can access by matcher.group(index)
    // Pattern.DOTALL - . by default doesn't match newlines, but with DOTALL it matches everything, including newlines
    private static final Pattern REUTERS_PATTERN = Pattern.compile("<REUTERS.*?>(.*?)</REUTERS>", Pattern.DOTALL);
    // The regex engine works step by step and here are 5 of them
    // 1. Find <REUTERS, "eat" it
    // 2. Then explore the rest char by char, each time check if it is not a pattern from step 3 eat it. If it is, go to step 3.
    // 3. Eat >
    // 4. Then explore the rest char by char, each time check if it is not a pattern from step 5 save it and eat it. If it is, go to step 5.
    // 5. Eat </REUTERS>
    // Generally, the division into steps occurs when special regex characters are encountered
    // If you want to treat those special characters as normal characters, you can use \\ to escape them

    private static final Pattern PLACES_PATTERN = Pattern.compile("<PLACES>(.*?)</PLACES>", Pattern.DOTALL);
    private static final Pattern D_TAG_PATTERN = Pattern.compile("<D>(.*?)</D>");
    private static final Pattern TITLE_PATTERN = Pattern.compile("<TITLE>(.*?)</TITLE>", Pattern.DOTALL);
    private static final Pattern BODY_PATTERN = Pattern.compile("<BODY>(.*?)</BODY>", Pattern.DOTALL);
    private static final Set<String> ALLOWED_COUNTRIES = Set.of(
            "west-germany", "usa", "france", "uk", "canada", "japan"
    );


    public List<SingleArticle> loadArticles(String directoryPath) {
        List<SingleArticle> validArticles = new ArrayList<>();

        try {
            List<Path> sgmFiles = Files.list(Paths.get(directoryPath))
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

        // Once a match is found, Matcher remembers the position
        while (reutersMatcher.find()) {
            // group(0) - the whole match <REUTERS...>Article content...</REUTERS>
            String articleRawText = reutersMatcher.group(1);

            String countryLabel = extractValidCountry(articleRawText);

            if (countryLabel != null) {
                String title = extractTag(articleRawText, TITLE_PATTERN);
                String body = extractTag(articleRawText, BODY_PATTERN);

                String fullText = (title + " " + body).trim();

                fullText = fullText.replace("&#3;", "");

                if (!fullText.isBlank()) {
                    articlesFromFile.add(new SingleArticle(countryLabel, fullText));
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
                String singleCountry = foundCountries.get(0);
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
