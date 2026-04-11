package org.kosowskiinowak.model;

public record CompletedExperiment(String label, ExperimentOutcome outcome, long wallTimeMs) {
}
