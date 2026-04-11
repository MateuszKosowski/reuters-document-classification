package org.kosowskiinowak.model;

import java.util.concurrent.Callable;

public record ExperimentTask(String label, Callable<ExperimentOutcome> callable) {
}
