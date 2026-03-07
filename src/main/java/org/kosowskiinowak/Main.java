package org.kosowskiinowak;

import org.kosowskiinowak.service.ArticleLoader;

public class Main {
    static void main(String[] args) {
        String dataSource = "src/main/resources/reuters21578";
        ArticleLoader articleLoader = new ArticleLoader();
        articleLoader.loadArticles(dataSource);
    }
}
