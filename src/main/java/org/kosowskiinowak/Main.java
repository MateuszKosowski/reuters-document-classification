package org.kosowskiinowak;

import org.kosowskiinowak.classifier.metric.ChebyshevMetric;
import org.kosowskiinowak.classifier.metric.EuclideanMetric;
import org.kosowskiinowak.classifier.metric.ManhattanMetric;
import org.kosowskiinowak.classifier.metric.Metric;
import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.model.SingleArticle;
import org.kosowskiinowak.service.ArticleLoader;
import org.kosowskiinowak.service.FeatureSubsetService;
import org.kosowskiinowak.service.FeatureVectorExtractor;
import org.kosowskiinowak.service.KnnClassifier;
import org.kosowskiinowak.service.NormalizationService;
import org.kosowskiinowak.service.QualityMeasureService;
import org.kosowskiinowak.service.ResultsExportService;
import org.kosowskiinowak.service.TextSimilarityService;
import org.kosowskiinowak.model.ClassMetrics;
import org.kosowskiinowak.model.CompletedExperiment;
import org.kosowskiinowak.model.DatasetSplit;
import org.kosowskiinowak.model.ExperimentOutcome;
import org.kosowskiinowak.model.ExperimentTask;
import org.kosowskiinowak.model.FeatureName;
import org.kosowskiinowak.model.MetricDefinition;
import org.kosowskiinowak.model.StageSearchResult;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.TreeSet;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class Main {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "%4$s: %5$s%n");
    }

    private static final Logger LOGGER = Logger.getLogger(Main.class.getName());

    private static final String DATA_SOURCE = "src/main/resources/reuters21578";
    private static final long SEED = 42L;

    private static final int MAX_K_TO_TEST = 30;

    private static final double STAGE_ONE_TRAIN_RATIO = 0.50;
    private static final List<Double> SPLIT_RATIOS = List.of(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90);

    private static final String OUTPUT_DIRECTORY = "experiment-results";
    private static final String STAGE_ONE_FILE = "01-k-metric-search.csv";
    private static final String STAGE_TWO_FILE = "02-split-search.csv";
    private static final String STAGE_THREE_FILE = "03-feature-search.csv";
    private static final String BEST_FINAL_FILE = "04-best-final-summary.csv";

    public static void main(String[] args) {
        long totalStartMs = System.currentTimeMillis();
        ExecutorService executor = null;
        try {
            List<FeatureVector> dataset = loadAndPrepareDataset();

            List<MetricDefinition> metricDefinitions = createMetricDefinitions();
            EnumSet<FeatureName> fullFeatureSet = EnumSet.allOf(FeatureName.class);

            FeatureSubsetService featureSubsetService = new FeatureSubsetService();
            ResultsExportService exportService = new ResultsExportService(featureSubsetService);
            NormalizationService normalizationService = new NormalizationService();
            KnnClassifier classifier = new KnnClassifier();
            QualityMeasureService qualityService = new QualityMeasureService();

            Path outputDirectory = exportService.prepareOutputDirectory(OUTPUT_DIRECTORY);
            int threadPoolSize = Math.max(1, Runtime.getRuntime().availableProcessors());
            LOGGER.info("Using fixed thread pool size = " + threadPoolSize);

            executor = Executors.newFixedThreadPool(threadPoolSize);
            Map<Double, DatasetSplit> preparedSplits = precomputeNormalizedSplits(dataset, normalizationService);

            List<String> stageOneRows = new ArrayList<>();
            stageOneRows.add(exportService.csvHeader());

            LOGGER.info(String.format(
                    Locale.US,
                    "Stage 1: fixed split %.0f/%.0f, fixed full feature set, scanning k and 3 metrics.",
                    STAGE_ONE_TRAIN_RATIO * 100,
                    (1.0 - STAGE_ONE_TRAIN_RATIO) * 100
            ));

            DatasetSplit fixedSplit = preparedSplits.get(STAGE_ONE_TRAIN_RATIO);
            LOGGER.info(String.format(
                    Locale.US,
                    "Stage 1 setup: train=%d, test=%d, kUpperBound=%d",
                    fixedSplit.trainingSet().size(),
                    fixedSplit.testSet().size(),
                    MAX_K_TO_TEST
            ));
            StageSearchResult stageOneResult = searchKAndMetric(
                    fixedSplit,
                    fullFeatureSet,
                    metricDefinitions,
                    classifier,
                    qualityService,
                    featureSubsetService,
                    executor,
                    stageOneRows,
                    exportService
            );

            if (stageOneResult.bestOutcome() == null) {
                LOGGER.warning("Stage 1 finished without a valid result.");
                return;
            }

            LOGGER.info(String.format(Locale.US, "Best after stage 1: k=%d, metric=%s",
                    stageOneResult.bestOutcome().k(), stageOneResult.bestOutcome().metricDefinition().name()));
            LOGGER.info(stageOneResult.stageNote());

            List<String> stageTwoRows = new ArrayList<>();
            stageTwoRows.add(exportService.csvHeader());

            LOGGER.info("Stage 2: fixed best k and metric from stage 1, scanning train/test splits.");
            LOGGER.info(String.format(
                    Locale.US,
                    "Stage 2 setup: k=%d, metric=%s, featureCount=%d.",
                    stageOneResult.bestOutcome().k(),
                    stageOneResult.bestOutcome().metricDefinition().name(),
                    stageOneResult.bestOutcome().activeFeatures().size()
            ));

            StageSearchResult stageTwoResult = searchSplits(
                    preparedSplits,
                    stageOneResult.bestOutcome().k(),
                    stageOneResult.bestOutcome().metricDefinition(),
                    fullFeatureSet,
                    classifier,
                    qualityService,
                    featureSubsetService,
                    executor,
                    stageTwoRows,
                    exportService
            );

            if (stageTwoResult.bestOutcome() == null) {
                LOGGER.warning("Stage 2 finished without a valid result.");
                return;
            }

            LOGGER.info(String.format(Locale.US, "Best after stage 2: split=%.0f/%.0f",
                    stageTwoResult.bestOutcome().trainRatio() * 100,
                    (1.0 - stageTwoResult.bestOutcome().trainRatio()) * 100));

            List<String> stageThreeRows = new ArrayList<>();
            stageThreeRows.add(exportService.csvHeader());

            LOGGER.info("Stage 3: fixed best k, metric and split, scanning feature subsets from 9 features down to 1.");
            LOGGER.info(String.format(
                    Locale.US,
                    "Stage 3 setup: split=%.0f/%.0f, k=%d, metric=%s.",
                    stageTwoResult.bestOutcome().trainRatio() * 100,
                    (1.0 - stageTwoResult.bestOutcome().trainRatio()) * 100,
                    stageTwoResult.bestOutcome().k(),
                    stageTwoResult.bestOutcome().metricDefinition().name()
            ));

            StageSearchResult stageThreeResult = searchFeatureSubsets(
                    preparedSplits.get(stageTwoResult.bestOutcome().trainRatio()),
                    stageTwoResult.bestOutcome().trainRatio(),
                    stageTwoResult.bestOutcome().k(),
                    stageTwoResult.bestOutcome().metricDefinition(),
                    classifier,
                    qualityService,
                    featureSubsetService,
                    executor,
                    stageThreeRows,
                    exportService
            );

            ExperimentOutcome bestFinalOutcome = stageTwoResult.bestOutcome();
            if (isBetter(stageThreeResult.bestOutcome(), bestFinalOutcome)) {
                bestFinalOutcome = stageThreeResult.bestOutcome();
            }

            List<String> bestFinalRows = new ArrayList<>();
            bestFinalRows.add(exportService.csvHeader());
            exportService.appendCsvRows(bestFinalRows, "BEST_FINAL", bestFinalOutcome);

            exportService.exportToCsv(stageOneRows, outputDirectory.resolve(STAGE_ONE_FILE).toString());
            exportService.exportToCsv(stageTwoRows, outputDirectory.resolve(STAGE_TWO_FILE).toString());
            exportService.exportToCsv(stageThreeRows, outputDirectory.resolve(STAGE_THREE_FILE).toString());
            exportService.exportToCsv(bestFinalRows, outputDirectory.resolve(BEST_FINAL_FILE).toString());

            LOGGER.info(String.format(Locale.US,
                    "Experiment completed. Best configuration: k=%d, metric=%s, split=%.0f/%.0f, featuresCount=%d -> score=%.4f",
                    bestFinalOutcome.k(),
                    bestFinalOutcome.metricDefinition().name(),
                    bestFinalOutcome.trainRatio() * 100,
                    (1.0 - bestFinalOutcome.trainRatio()) * 100,
                    bestFinalOutcome.activeFeatures().size(),
                    bestFinalOutcome.metrics().selectionScore()
            ));
            LOGGER.info(String.format(Locale.US, "Best configuration details: %s", bestFinalOutcome));
            printDetailedSummary(bestFinalOutcome, featureSubsetService);
            LOGGER.info("CSV files saved to: " + outputDirectory.toAbsolutePath());
        } finally {
            if (executor != null) {
                shutdownExecutor(executor);
            }
            logTotalExperimentTime(totalStartMs);
        }
    }

    private static List<FeatureVector> loadAndPrepareDataset() {
        long extractionStartMs = System.currentTimeMillis();
        ArticleLoader articleLoader = new ArticleLoader();
        List<SingleArticle> rawArticles = articleLoader.loadArticles(DATA_SOURCE);
        LOGGER.info("Loaded raw articles: " + rawArticles.size());
        LOGGER.info("Starting feature extraction with parallelStream over loaded articles.");

        FeatureVectorExtractor extractor = new FeatureVectorExtractor();
        List<FeatureVector> rawVectors = rawArticles.parallelStream()
                .map(article -> {
                    try {
                        return extractor.extractAllFeaturesFromArticle(article);
                    } catch (IOException exception) {
                        LOGGER.warning("Failed to extract features for one article: " + exception.getMessage());
                        return null;
                    }
                })
                .filter(Objects::nonNull)
                .collect(Collectors.toCollection(ArrayList::new));

        Collections.shuffle(rawVectors, new Random(SEED));
        LOGGER.info(String.format(
                Locale.US,
                "Prepared feature vectors: %d (feature extraction + shuffle took %d ms).",
                rawVectors.size(),
                System.currentTimeMillis() - extractionStartMs
        ));
        return rawVectors;
    }

    private static List<MetricDefinition> createMetricDefinitions() {
        TextSimilarityService textSimilarityService = new TextSimilarityService();

        return List.of(
                new MetricDefinition("Euclidean", new EuclideanMetric(textSimilarityService)),
                new MetricDefinition("Manhattan", new ManhattanMetric(textSimilarityService)),
                new MetricDefinition("Chebyshev", new ChebyshevMetric(textSimilarityService))
        );
    }

    private static StageSearchResult searchKAndMetric(DatasetSplit split,
                                                      EnumSet<FeatureName> activeFeatures,
                                                      List<MetricDefinition> metricDefinitions,
                                                      KnnClassifier classifier,
                                                      QualityMeasureService qualityService,
                                                      FeatureSubsetService featureSubsetService,
                                                      ExecutorService executor,
                                                      List<String> csvRows,
                                                      ResultsExportService exportService) {
        List<ExperimentTask> tasks = new ArrayList<>();
        for (int k = 1; k <= MAX_K_TO_TEST; k++) {
            for (MetricDefinition metricDefinition : metricDefinitions) {
                final int currentK = k;
                tasks.add(new ExperimentTask(
                        String.format(Locale.US, "k=%03d metric=%s", currentK, metricDefinition.name()),
                        () -> runSingleExperiment(
                                split,
                                STAGE_ONE_TRAIN_RATIO,
                                metricDefinition,
                                currentK,
                                activeFeatures,
                                classifier,
                                qualityService,
                                featureSubsetService
                        )
                ));
            }
        }

        LOGGER.info(String.format(
                Locale.US,
                "Stage 1 progress: submitting %d tasks (k=1..%d x %d metrics) in parallel.",
                tasks.size(),
                MAX_K_TO_TEST,
                metricDefinitions.size()
        ));

        final ExperimentOutcome[] bestOutcomeHolder = new ExperimentOutcome[1];

        consumeExperimentTasks(
                executor,
                tasks,
                "Stage 1",
                30,
                completedTask -> {
                    ExperimentOutcome outcome = completedTask.outcome();
                    exportService.appendCsvRows(csvRows, "STAGE_1_K_METRIC", outcome);
                    LOGGER.info(String.format(Locale.US,
                            "Stage 1 task finished [%s] -> score=%.4f, accuracy=%.4f, macroF1=%.4f",
                            completedTask.label(), outcome.metrics().selectionScore(), outcome.metrics().accuracy(), outcome.metrics().macroF1()
                    ));

                    if (isBetter(outcome, bestOutcomeHolder[0])) {
                        bestOutcomeHolder[0] = outcome;
                        LOGGER.info(String.format(Locale.US,
                                "Stage 1 NEW GLOBAL BEST: k=%d, metric=%s -> score=%.4f",
                                outcome.k(), outcome.metricDefinition().name(), outcome.metrics().selectionScore()
                        ));
                    }
                }
        );

        return new StageSearchResult(
                bestOutcomeHolder[0],
                String.format(
                        Locale.US,
                        "Stage 1 summary: testedK=%d, testedCombinations=%d, bestScore=%.4f, stopReason=FULL_SCAN_COMPLETED",
                        MAX_K_TO_TEST,
                        tasks.size(),
                        (bestOutcomeHolder[0] != null) ? bestOutcomeHolder[0].metrics().selectionScore() : 0.0
                )
        );
    }

    private static StageSearchResult searchSplits(Map<Double, DatasetSplit> preparedSplits,
                                                  int k,
                                                  MetricDefinition metricDefinition,
                                                  EnumSet<FeatureName> activeFeatures,
                                                  KnnClassifier classifier,
                                                  QualityMeasureService qualityService,
                                                  FeatureSubsetService featureSubsetService,
                                                  ExecutorService executor,
                                                  List<String> csvRows,
                                                  ResultsExportService exportService) {
        List<ExperimentTask> tasks = new ArrayList<>();
        for (double trainRatio : SPLIT_RATIOS) {
            DatasetSplit split = preparedSplits.get(trainRatio);

            tasks.add(new ExperimentTask(
                    String.format(
                            Locale.US,
                            "split=%.0f/%.0f train=%d test=%d k=%d",
                            trainRatio * 100,
                            (1.0 - trainRatio) * 100,
                            split.trainingSet().size(),
                            split.testSet().size(),
                            k
                    ),
                    () -> runSingleExperiment(
                            split,
                            trainRatio,
                            metricDefinition,
                            k,
                            activeFeatures,
                            classifier,
                            qualityService,
                            featureSubsetService
                    )
            ));
        }

        LOGGER.info(String.format(
                Locale.US,
                "Stage 2 progress: submitting %d split tasks.",
                tasks.size()
        ));
        final ExperimentOutcome[] bestOutcomeHolder = new ExperimentOutcome[1];

        consumeExperimentTasks(
                executor,
                tasks,
                "Stage 2",
                1,
                completedTask -> {
                    ExperimentOutcome outcome = completedTask.outcome();
                    exportService.appendCsvRows(csvRows, "STAGE_2_SPLIT", outcome);
                    LOGGER.info(String.format(Locale.US,
                            "Stage 2 split finished [%s] -> score=%.4f, accuracy=%.4f, macroF1=%.4f",
                            completedTask.label(), outcome.metrics().selectionScore(), outcome.metrics().accuracy(), outcome.metrics().macroF1()
                    ));

                    if (isBetter(outcome, bestOutcomeHolder[0])) {
                        bestOutcomeHolder[0] = outcome;
                        LOGGER.info(String.format(Locale.US,
                                "Stage 2 NEW BEST: split=%.0f/%.0f -> score=%.4f",
                                outcome.trainRatio() * 100, (1.0 - outcome.trainRatio()) * 100, outcome.metrics().selectionScore()
                        ));
                    }
                }
        );

        return new StageSearchResult(bestOutcomeHolder[0], "Stage 2 summary: tested all configured train/test splits.");
    }

    private static StageSearchResult searchFeatureSubsets(DatasetSplit split,
                                                          double trainRatio,
                                                          int k,
                                                          MetricDefinition metricDefinition,
                                                          KnnClassifier classifier,
                                                          QualityMeasureService qualityService,
                                                          FeatureSubsetService featureSubsetService,
                                                          ExecutorService executor,
                                                          List<String> csvRows,
                                                          ResultsExportService exportService) {
        List<EnumSet<FeatureName>> featureSubsets = featureSubsetService.generateFeatureSubsets();
        LOGGER.info(String.format(
                Locale.US,
                "Stage 3 progress: prepared %d feature subsets to evaluate.",
                featureSubsets.size()
        ));

        List<ExperimentTask> tasks = featureSubsets.stream()
                .map(activeFeatures -> new ExperimentTask(
                        String.format(
                                Locale.US,
                                "featureCount=%d features=%s",
                                activeFeatures.size(),
                                featureSubsetService.formatFeatureSet(activeFeatures)
                        ),
                        () -> runSingleExperiment(
                                split,
                                trainRatio,
                                metricDefinition,
                                k,
                                activeFeatures,
                                classifier,
                                qualityService,
                                featureSubsetService
                        )
                ))
                .toList();

        final ExperimentOutcome[] bestOutcomeHolder = new ExperimentOutcome[1];

        consumeExperimentTasks(
                executor,
                tasks,
                "Stage 3",
                25,
                completedTask -> {
                    ExperimentOutcome outcome = completedTask.outcome();
                    exportService.appendCsvRows(csvRows, "STAGE_3_FEATURES", outcome);

                    if (isBetter(outcome, bestOutcomeHolder[0])) {
                        bestOutcomeHolder[0] = outcome;
                        LOGGER.info(String.format(Locale.US,
                                "Stage 3 NEW BEST: featuresCount=%d, features=[%s] -> score=%.4f",
                                outcome.activeFeatures().size(), featureSubsetService.formatFeatureSet(outcome.activeFeatures()), outcome.metrics().selectionScore()
                        ));
                    }
                }
        );

        return new StageSearchResult(
                bestOutcomeHolder[0],
                String.format(Locale.US, "Stage 3 summary: tested %d feature subsets.", featureSubsets.size())
        );
    }

    private static DatasetSplit createNormalizedSplit(List<FeatureVector> dataset,
                                                      double trainRatio,
                                                      NormalizationService normalizationService) {
        int trainCount = (int) (dataset.size() * trainRatio);

        List<FeatureVector> rawTrainingSet = dataset.subList(0, trainCount);
        List<FeatureVector> rawTestSet = dataset.subList(trainCount, dataset.size());

        NormalizationService.NormalizationResult normalizationResult =
                normalizationService.normalizeTrainAndTest(rawTrainingSet, rawTestSet);

        return new DatasetSplit(normalizationResult.normalizedTrain, normalizationResult.normalizedTest);
    }

    private static Map<Double, DatasetSplit> precomputeNormalizedSplits(List<FeatureVector> dataset,
                                                                        NormalizationService normalizationService) {
        LOGGER.info("Precomputing normalized train/test splits.");
        List<Double> requiredRatios = new ArrayList<>(new TreeSet<>(SPLIT_RATIOS));
        if (!requiredRatios.contains(STAGE_ONE_TRAIN_RATIO)) {
            requiredRatios.add(STAGE_ONE_TRAIN_RATIO);
            requiredRatios.sort(Double::compareTo);
        }

        Map<Double, DatasetSplit> preparedSplits = requiredRatios.stream()
                .collect(Collectors.toMap(
                        ratio -> ratio,
                        ratio -> createNormalizedSplit(dataset, ratio, normalizationService)
                ));

        for (double ratio : requiredRatios) {
            DatasetSplit split = preparedSplits.get(ratio);
            LOGGER.info(String.format(
                    Locale.US,
                    "Prepared split %.0f/%.0f -> train=%d, test=%d.",
                    ratio * 100,
                    (1.0 - ratio) * 100,
                    split.trainingSet().size(),
                    split.testSet().size()
            ));
        }

        return preparedSplits;
    }

    private static ExperimentOutcome runSingleExperiment(DatasetSplit split,
                                                         double trainRatio,
                                                         MetricDefinition metricDefinition,
                                                         int k,
                                                         EnumSet<FeatureName> activeFeatures,
                                                         KnnClassifier classifier,
                                                         QualityMeasureService qualityService,
                                                         FeatureSubsetService featureSubsetService) {
        List<FeatureVector> trainingSet = featureSubsetService.isFullFeatureSet(activeFeatures)
                ? split.trainingSet()
                : featureSubsetService.maskDataset(split.trainingSet(), activeFeatures);
        List<FeatureVector> testSet = featureSubsetService.isFullFeatureSet(activeFeatures)
                ? split.testSet()
                : featureSubsetService.maskDataset(split.testSet(), activeFeatures);

        long startTime = System.currentTimeMillis();
        List<QualityMeasureService.ClassificationResult> results =
                classifyTestSetSequentially(testSet, trainingSet, k, metricDefinition.metric(), classifier);
        long durationMs = System.currentTimeMillis() - startTime;

        return new ExperimentOutcome(
                trainRatio,
                metricDefinition,
                k,
                EnumSet.copyOf(activeFeatures),
                qualityService.calculateDetailedMetrics(results),
                durationMs
        );
    }

    private static List<QualityMeasureService.ClassificationResult> classifyTestSetSequentially(
            List<FeatureVector> testSet,
            List<FeatureVector> trainingSet,
            int k,
            Metric metric,
            KnnClassifier classifier) {
        List<QualityMeasureService.ClassificationResult> results = new ArrayList<>(testSet.size());
        for (FeatureVector testVector : testSet) {
            String predictedLabel = classifier.classify(testVector, trainingSet, k, metric);
            results.add(new QualityMeasureService.ClassificationResult(testVector.label(), predictedLabel));
        }
        return results;
    }

    private static void consumeExperimentTasks(ExecutorService executor,
                                               List<ExperimentTask> tasks,
                                               String progressPrefix,
                                               int logEvery,
                                               Consumer<CompletedExperiment> onCompleted) {
        if (tasks.isEmpty()) {
            LOGGER.info(progressPrefix + " has no tasks to execute.");
            return;
        }

        try {
            ExecutorCompletionService<CompletedExperiment> completionService = new ExecutorCompletionService<>(executor);
            for (ExperimentTask task : tasks) {
                completionService.submit(() -> {
                    long startMs = System.currentTimeMillis();
                    ExperimentOutcome outcome = task.callable().call();
                    long wallTimeMs = System.currentTimeMillis() - startMs;
                    return new CompletedExperiment(task.label(), outcome, wallTimeMs);
                });
            }

            for (int completed = 1; completed <= tasks.size(); completed++) {
                Future<CompletedExperiment> future = completionService.take();
                CompletedExperiment completedExperiment = future.get();
                onCompleted.accept(completedExperiment);

                if (completed == 1 || completed == tasks.size() || completed % Math.max(1, logEvery) == 0) {
                    LOGGER.info(String.format(
                            Locale.US,
                            "%s progress: %d/%d completed (latest=%s, wallTime=%d ms).",
                            progressPrefix,
                            completed,
                            tasks.size(),
                            completedExperiment.label(),
                            completedExperiment.wallTimeMs()
                    ));
                }
            }
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Experiment execution interrupted.", exception);
        } catch (ExecutionException exception) {
            throw new IllegalStateException("Experiment execution failed.", exception.getCause());
        }
    }

    private static boolean isBetter(ExperimentOutcome candidate, ExperimentOutcome currentBest) {
        if (currentBest == null) return true;

        if (candidate.metrics().selectionScore() != currentBest.metrics().selectionScore()) {
            return candidate.metrics().selectionScore() > currentBest.metrics().selectionScore();
        }

        return candidate.activeFeatures().size() < currentBest.activeFeatures().size();
    }

    private static void printDetailedSummary(ExperimentOutcome outcome, FeatureSubsetService featureSubsetService) {
        LOGGER.info("=== FINAL BEST CONFIGURATION ===");
        LOGGER.info(String.format(Locale.US, "k = %d", outcome.k()));
        LOGGER.info("metric = " + outcome.metricDefinition().name());
        LOGGER.info(String.format(
                Locale.US,
                "train/test = %.0f%% / %.0f%%",
                outcome.trainRatio() * 100,
                (1.0 - outcome.trainRatio()) * 100
        ));
        LOGGER.info("features = " + featureSubsetService.formatFeatureSet(outcome.activeFeatures()));
        LOGGER.info(String.format(
                Locale.US,
                "selectionScore = %.6f (harmonic mean of accuracy and macro F1)",
                outcome.metrics().selectionScore()
        ));
        LOGGER.info(String.format(Locale.US, "accuracy = %.6f", outcome.metrics().accuracy()));
        LOGGER.info(String.format(
                Locale.US,
                "macro: precision = %.6f, recall = %.6f, f1 = %.6f",
                outcome.metrics().macroPrecision(),
                outcome.metrics().macroRecall(),
                outcome.metrics().macroF1()
        ));
        LOGGER.info("per-class results:");

        for (ClassMetrics classMetrics : outcome.metrics().perClassMetrics()) {
            LOGGER.info(String.format(
                    Locale.US,
                    "class=%s, precision=%.6f, recall=%.6f, f1=%.6f",
                    classMetrics.className(),
                    classMetrics.precision(),
                    classMetrics.recall(),
                    classMetrics.f1()
            ));
        }
    }

    private static void logTotalExperimentTime(long totalStartMs) {
        long totalDurationMs = System.currentTimeMillis() - totalStartMs;
        LOGGER.info(String.format(
                Locale.US,
                "Total experiment time = %d ms (%s)",
                totalDurationMs,
                formatDuration(totalDurationMs)
        ));
    }

    private static String formatDuration(long durationMs) {
        long hours = durationMs / 3_600_000;
        long minutes = (durationMs % 3_600_000) / 60_000;
        long seconds = (durationMs % 60_000) / 1_000;
        long milliseconds = durationMs % 1_000;

        return String.format(
                Locale.US,
                "%02d:%02d:%02d.%03d",
                hours,
                minutes,
                seconds,
                milliseconds
        );
    }

    private static void shutdownExecutor(ExecutorService executor) {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(1, TimeUnit.MINUTES)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException exception) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
