package org.kosowskiinowak;

import org.kosowskiinowak.classifier.metric.ChebyshevMetric;
import org.kosowskiinowak.classifier.metric.EuclideanMetric;
import org.kosowskiinowak.classifier.metric.ManhattanMetric;
import org.kosowskiinowak.classifier.metric.Metric;
import org.kosowskiinowak.model.FeatureVector;
import org.kosowskiinowak.model.SingleArticle;
import org.kosowskiinowak.service.ArticleLoader;
import org.kosowskiinowak.service.FeatureVectorExtractor;
import org.kosowskiinowak.service.KnnClassifier;
import org.kosowskiinowak.service.NormalizationService;
import org.kosowskiinowak.service.QualityMeasureService;
import org.kosowskiinowak.service.ResultsExportService;
import org.kosowskiinowak.service.TextSimilarityService;
import org.kosowskiinowak.model.ClassMetrics;
import org.kosowskiinowak.model.CompletedExperiment;
import org.kosowskiinowak.model.DatasetSplit;
import org.kosowskiinowak.model.DetailedMetrics;
import org.kosowskiinowak.model.ExperimentOutcome;
import org.kosowskiinowak.model.ExperimentTask;
import org.kosowskiinowak.model.FeatureName;
import org.kosowskiinowak.model.MetricDefinition;
import org.kosowskiinowak.model.StageSearchResult;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
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

            ResultsExportService exportService = new ResultsExportService();
            NormalizationService normalizationService = new NormalizationService();
            KnnClassifier classifier = new KnnClassifier();
            QualityMeasureService qualityService = new QualityMeasureService();

            Path outputDirectory = exportService.prepareOutputDirectory(OUTPUT_DIRECTORY);
            int threadPoolSize = Math.max(1, Runtime.getRuntime().availableProcessors());
            LOGGER.info("Using fixed thread pool size = " + threadPoolSize);

            executor = Executors.newFixedThreadPool(threadPoolSize);
            Map<Double, DatasetSplit> preparedSplits = precomputeNormalizedSplits(dataset, normalizationService);

            List<String> stageOneRows = new ArrayList<>();
            stageOneRows.add(csvHeader());

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
                    executor,
                    stageOneRows
            );

            if (stageOneResult.bestOutcome() == null) {
                LOGGER.warning("Stage 1 finished without a valid result.");
                return;
            }

            LOGGER.info(String.format(Locale.US, "Best after stage 1: k=%d, metric=%s",
                    stageOneResult.bestOutcome().k(), stageOneResult.bestOutcome().metricDefinition().name()));
            LOGGER.info(stageOneResult.stageNote());

            List<String> stageTwoRows = new ArrayList<>();
            stageTwoRows.add(csvHeader());

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
                    executor,
                    stageTwoRows
            );

            if (stageTwoResult.bestOutcome() == null) {
                LOGGER.warning("Stage 2 finished without a valid result.");
                return;
            }

            LOGGER.info(String.format(Locale.US, "Best after stage 2: split=%.0f/%.0f",
                    stageTwoResult.bestOutcome().trainRatio() * 100,
                    (1.0 - stageTwoResult.bestOutcome().trainRatio()) * 100));

            List<String> stageThreeRows = new ArrayList<>();
            stageThreeRows.add(csvHeader());

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
                    executor,
                    stageThreeRows
            );

            ExperimentOutcome bestFinalOutcome = stageTwoResult.bestOutcome();
            if (isBetter(stageThreeResult.bestOutcome(), bestFinalOutcome)) {
                bestFinalOutcome = stageThreeResult.bestOutcome();
            }

            List<String> bestFinalRows = new ArrayList<>();
            bestFinalRows.add(csvHeader());
            appendCsvRows(bestFinalRows, "BEST_FINAL", bestFinalOutcome);

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
            printDetailedSummary(bestFinalOutcome);
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
                                                      ExecutorService executor,
                                                      List<String> csvRows) {
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
                                qualityService
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
                    appendCsvRows(csvRows, "STAGE_1_K_METRIC", outcome);
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
                                                  ExecutorService executor,
                                                  List<String> csvRows) {
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
                            qualityService
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
                    appendCsvRows(csvRows, "STAGE_2_SPLIT", outcome);
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
                                                          ExecutorService executor,
                                                          List<String> csvRows) {
        List<EnumSet<FeatureName>> featureSubsets = generateFeatureSubsets();
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
                                formatFeatureSet(activeFeatures)
                        ),
                        () -> runSingleExperiment(
                                split,
                                trainRatio,
                                metricDefinition,
                                k,
                                activeFeatures,
                                classifier,
                                qualityService
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
                    appendCsvRows(csvRows, "STAGE_3_FEATURES", outcome);

                    if (isBetter(outcome, bestOutcomeHolder[0])) {
                        bestOutcomeHolder[0] = outcome;
                        LOGGER.info(String.format(Locale.US,
                                "Stage 3 NEW BEST: featuresCount=%d, features=[%s] -> score=%.4f",
                                outcome.activeFeatures().size(), formatFeatureSet(outcome.activeFeatures()), outcome.metrics().selectionScore()
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
                                                         QualityMeasureService qualityService) {
        List<FeatureVector> trainingSet = isFullFeatureSet(activeFeatures)
                ? split.trainingSet()
                : maskDataset(split.trainingSet(), activeFeatures);
        List<FeatureVector> testSet = isFullFeatureSet(activeFeatures)
                ? split.testSet()
                : maskDataset(split.testSet(), activeFeatures);

        long startTime = System.currentTimeMillis();
        List<QualityMeasureService.ClassificationResult> results =
                classifyTestSetSequentially(testSet, trainingSet, k, metricDefinition.metric(), classifier);
        long durationMs = System.currentTimeMillis() - startTime;

        return new ExperimentOutcome(
                trainRatio,
                metricDefinition,
                k,
                EnumSet.copyOf(activeFeatures),
                calculateMetrics(results, qualityService),
                durationMs
        );
    }

    private static DetailedMetrics calculateMetrics(List<QualityMeasureService.ClassificationResult> results,
                                                    QualityMeasureService qualityService) {
        Map<String, Double> accuracyMap = qualityService.calculateAccuracy(results);
        Map<String, Double> precisionMap = qualityService.calculatePrecision(results);
        Map<String, Double> recallMap = qualityService.calculateRecall(results);
        Map<String, Double> f1Map = qualityService.calculateF1(precisionMap, recallMap);

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

        return new DetailedMetrics(
                extractAccuracyValue(accuracyMap),
                extractMacroValue(precisionMap),
                extractMacroValue(recallMap),
                extractMacroValue(f1Map),
                harmonicMean(extractAccuracyValue(accuracyMap), extractMacroValue(f1Map)),
                perClassMetrics
        );
    }

    private static double harmonicMean(double first, double second) {
        if (first <= 0.0 || second <= 0.0) {
            return 0.0;
        }
        return 2.0 * first * second / (first + second);
    }

    private static boolean isFullFeatureSet(Set<FeatureName> activeFeatures) {
        return activeFeatures.size() == FeatureName.values().length;
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

    private static Set<String> filterClassNames(Set<String> rawKeys) {
        return rawKeys.stream()
                .filter(Main::isClassName)
                .collect(Collectors.toCollection(TreeSet::new));
    }

    private static boolean isClassName(String key) {
        String normalized = key.toLowerCase(Locale.ROOT);
        return !normalized.contains("macro")
                && !normalized.contains("makro")
                && !normalized.contains("accuracy");
    }

    private static double extractAccuracyValue(Map<String, Double> accuracyMap) {
        return accuracyMap.entrySet().stream()
                .filter(entry -> entry.getKey().toLowerCase(Locale.ROOT).contains("accuracy"))
                .map(Map.Entry::getValue)
                .findFirst()
                .orElse(0.0);
    }

    private static double extractMacroValue(Map<String, Double> metricMap) {
        return metricMap.entrySet().stream()
                .filter(entry -> {
                    String key = entry.getKey().toLowerCase(Locale.ROOT);
                    return key.contains("macro") || key.contains("makro");
                })
                .map(Map.Entry::getValue)
                .findFirst()
                .orElse(0.0);
    }

    private static List<FeatureVector> maskDataset(List<FeatureVector> dataset, Set<FeatureName> activeFeatures) {
        return dataset.stream()
                .map(vector -> maskVector(vector, activeFeatures))
                .toList();
    }

    private static FeatureVector maskVector(FeatureVector vector, Set<FeatureName> activeFeatures) {
        return new FeatureVector(
                vector.label(),
                activeFeatures.contains(FeatureName.LONGEST_WORD) ? safeText(vector.longestWord()) : "",
                activeFeatures.contains(FeatureName.MOST_FREQUENT_WORD) ? safeText(vector.mostFrequentWord()) : "",
                activeFeatures.contains(FeatureName.AVERAGE_WORD_LENGTH) ? vector.averageWordLength() : 0.0,
                activeFeatures.contains(FeatureName.VOCABULARY_RICHNESS) ? vector.vocabularyRichness() : 0.0,
                activeFeatures.contains(FeatureName.AVERAGE_SENTENCE_LENGTH) ? vector.averageSentenceLength() : 0.0,
                activeFeatures.contains(FeatureName.UPPERCASE_LETTER_RATIO) ? vector.uppercaseLetterRatio() : 0.0,
                activeFeatures.contains(FeatureName.FINANCIAL_SIGN_DENSITY) ? vector.financialSignDensity() : 0.0,
                activeFeatures.contains(FeatureName.FLESCH_READING_EASE_INDEX) ? vector.fleschReadingEaseIndex() : 0.0,
                activeFeatures.contains(FeatureName.VOWEL_TO_CONSONANT_RATIO) ? vector.vowelToConsonantRatio() : 0.0,
                activeFeatures.contains(FeatureName.SUM_OF_ALL_NUMERIC_VALUES) ? vector.sumOfAllNumericValues() : 0.0
        );
    }

    private static String safeText(String value) {
        return value == null ? "" : value;
    }

    private static List<EnumSet<FeatureName>> generateFeatureSubsets() {
        FeatureName[] features = FeatureName.values();
        List<EnumSet<FeatureName>> subsets = new ArrayList<>();

        int maxMask = (1 << features.length) - 1;
        for (int mask = 1; mask <= maxMask; mask++) {
            int featureCount = Integer.bitCount(mask);
            if (featureCount == features.length) {
                continue;
            }

            EnumSet<FeatureName> subset = EnumSet.noneOf(FeatureName.class);
            for (int bitIndex = 0; bitIndex < features.length; bitIndex++) {
                if ((mask & (1 << bitIndex)) != 0) {
                    subset.add(features[bitIndex]);
                }
            }

            subsets.add(subset);
        }

        subsets.sort(Comparator
                .<EnumSet<FeatureName>>comparingInt(Set::size)
                .reversed()
                .thenComparing(Main::formatFeatureSet));

        return subsets;
    }

    private static boolean isBetter(ExperimentOutcome candidate, ExperimentOutcome currentBest) {
        if (currentBest == null) return true;

        if (candidate.metrics().selectionScore() != currentBest.metrics().selectionScore()) {
            return candidate.metrics().selectionScore() > currentBest.metrics().selectionScore();
        }

        return candidate.activeFeatures().size() < currentBest.activeFeatures().size();
    }

    private static void appendCsvRows(List<String> csvRows, String stageName, ExperimentOutcome outcome) {
        csvRows.add(formatCsvRow(stageName, outcome, "MACRO", "MACRO",
                outcome.metrics().macroPrecision(),
                outcome.metrics().macroRecall(),
                outcome.metrics().macroF1()));

        for (ClassMetrics classMetrics : outcome.metrics().perClassMetrics()) {
            csvRows.add(formatCsvRow(stageName, outcome, "CLASS", classMetrics.className(),
                    classMetrics.precision(),
                    classMetrics.recall(),
                    classMetrics.f1()));
        }
    }

    private static String formatCsvRow(String stageName,
                                       ExperimentOutcome outcome,
                                       String rowType,
                                       String className,
                                       double precision,
                                       double recall,
                                       double f1) {
        return String.format(
                Locale.US,
                "%s;%.2f;%.2f;%s;%03d;%02d;%s;%s;%s;%.6f;%.6f;%.6f;%.6f;%.6f;%d",
                stageName,
                outcome.trainRatio(),
                1.0 - outcome.trainRatio(),
                outcome.metricDefinition().name(),
                outcome.k(),
                outcome.activeFeatures().size(),
                formatFeatureSet(outcome.activeFeatures()),
                rowType,
                className,
                outcome.metrics().selectionScore(),
                outcome.metrics().accuracy(),
                precision,
                recall,
                f1,
                outcome.durationMs()
        );
    }

    private static String csvHeader() {
        return "Stage;TrainRatio;TestRatio;Metric;K;FeatureCount;Features;RowType;Class;SelectionScore;Accuracy;Precision;Recall;F1;DurationMs";
    }

    private static String formatFeatureSet(Set<FeatureName> activeFeatures) {
        return activeFeatures.stream()
                .sorted(Comparator.comparingInt(Enum::ordinal))
                .map(FeatureName::csvLabel)
                .collect(Collectors.joining(","));
    }

    private static void printDetailedSummary(ExperimentOutcome outcome) {
        LOGGER.info("=== FINAL BEST CONFIGURATION ===");
        LOGGER.info(String.format(Locale.US, "k = %d", outcome.k()));
        LOGGER.info("metric = " + outcome.metricDefinition().name());
        LOGGER.info(String.format(
                Locale.US,
                "train/test = %.0f%% / %.0f%%",
                outcome.trainRatio() * 100,
                (1.0 - outcome.trainRatio()) * 100
        ));
        LOGGER.info("features = " + formatFeatureSet(outcome.activeFeatures()));
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
