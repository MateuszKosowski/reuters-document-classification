package org.kosowskiinowak.model;

import java.util.List;

public record DatasetSplit(List<FeatureVector> trainingSet, List<FeatureVector> testSet) {
}
