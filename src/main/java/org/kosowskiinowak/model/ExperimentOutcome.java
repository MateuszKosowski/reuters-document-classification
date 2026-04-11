package org.kosowskiinowak.model;

import java.util.EnumSet;

public record ExperimentOutcome(double trainRatio,
                                MetricDefinition metricDefinition,
                                int k,
                                EnumSet<FeatureName> activeFeatures,
                                DetailedMetrics metrics,
                                long durationMs) {
}
