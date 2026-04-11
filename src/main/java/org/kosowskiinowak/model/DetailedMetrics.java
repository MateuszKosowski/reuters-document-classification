package org.kosowskiinowak.model;

import java.util.List;

public record DetailedMetrics(double accuracy,
                              double macroPrecision,
                              double macroRecall,
                              double macroF1,
                              double selectionScore,
                              List<ClassMetrics> perClassMetrics) {
}
