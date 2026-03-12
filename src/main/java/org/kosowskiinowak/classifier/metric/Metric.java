package org.kosowskiinowak.classifier.metric;

import org.kosowskiinowak.model.FeatureVector;

public interface Metric {
    double calculate(FeatureVector a, FeatureVector b);
}
