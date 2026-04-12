package org.kosowskiinowak.service;

import org.kosowskiinowak.model.ClassMetrics;
import org.kosowskiinowak.model.ExperimentOutcome;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ResultsExportService {

    private static final Logger LOGGER = Logger.getLogger(ResultsExportService.class.getName());

    private final FeatureSubsetService featureSubsetService;

    public ResultsExportService(FeatureSubsetService featureSubsetService) {
        this.featureSubsetService = featureSubsetService;
    }

    public String csvHeader() {
        return "Stage;TrainRatio;TestRatio;Metric;K;FeatureCount;Features;RowType;Class;SelectionScore;Accuracy;Precision;Recall;F1;DurationMs";
    }

    public void appendCsvRows(List<String> csvRows, String stageName, ExperimentOutcome outcome) {
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

    public String formatCsvRow(String stageName,
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
                featureSubsetService.formatFeatureSet(outcome.activeFeatures()),
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

    public Path prepareOutputDirectory(String directoryName) {
        try {
            Path outputDirectory = Path.of(directoryName);
            Files.createDirectories(outputDirectory);
            return outputDirectory;
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to create output directory: " + directoryName, exception);
        }
    }

    public void exportToCsv(List<String> csvResults, String outputFileName) {
        if (csvResults == null || csvResults.isEmpty()) {
            LOGGER.warning("No results to export.");
            return;
        }

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
            LOGGER.info("Full report saved to file: " + outputFileName);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Error saving results: " + e.getMessage(), e);
        }
    }
}
