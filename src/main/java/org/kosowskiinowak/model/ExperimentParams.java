package org.kosowskiinowak.model;

import org.kosowskiinowak.classifier.metric.Metric;

public record ExperimentParams(int k, double trainRatio, Metric metric) {}
