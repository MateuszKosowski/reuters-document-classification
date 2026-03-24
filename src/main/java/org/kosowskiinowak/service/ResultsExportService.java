package org.kosowskiinowak.service;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ResultsExportService {

    private static final Logger LOGGER = Logger.getLogger(ResultsExportService.class.getName());

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

