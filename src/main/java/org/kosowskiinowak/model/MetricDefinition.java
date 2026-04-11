package org.kosowskiinowak.model;

import org.kosowskiinowak.classifier.metric.Metric;

public record MetricDefinition(String name, Metric metric) {
}
