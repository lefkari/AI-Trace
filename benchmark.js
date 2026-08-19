/*
 AI TRACE — BENCHMARK ENGINE V4.1
 Ground-truth evaluation + calibration foundation
*/

const BENCHMARK_VERSION = "4.1";

const STORAGE_KEY = "aiTraceBenchmarkV41";

export class AITraceBenchmark {

  constructor() {
    this.results = this.load();
  }

  load() {
    try {
      return JSON.parse(
        localStorage.getItem(STORAGE_KEY)
      ) || [];
    } catch {
      return [];
    }
  }

  save() {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(this.results)
    );
  }

  add({
    id,
    groundTruth,
    source,
    words,
    tmr,
    modern,
    segmentScores,
    modelGap,
    segmentSpread
  }) {

    const record = {
      id,
      version: BENCHMARK_VERSION,
      timestamp: new Date().toISOString(),

      groundTruth,
      source,

      words,

      models: {
        tmr,
        modern
      },

      segments: segmentScores,

      modelGap,
      segmentSpread
    };

    this.results.push(record);

    this.save();

    return record;
  }

  clear() {
    this.results = [];
    this.save();
  }

  classifyRaw(record) {

    const { tmr, modern } =
      record.models;

    const avg =
      (tmr + modern) / 2;

    /*
      IMPORTANT:
      This is only a benchmark classification rule.
      It is NOT yet our final production calibration.
    */

    if (avg >= 70)
      return "AI";

    if (avg <= 30)
      return "HUMAN";

    return "UNCERTAIN";
  }

  calculateMetrics() {

    let TP = 0;
    let TN = 0;
    let FP = 0;
    let FN = 0;
    let uncertain = 0;

    const usable =
      this.results.filter(
        r =>
          r.groundTruth === "AI" ||
          r.groundTruth === "HUMAN"
      );

    for (const record of usable) {

      const prediction =
        this.classifyRaw(record);

      if (prediction === "UNCERTAIN") {
        uncertain++;
        continue;
      }

      if (
        record.groundTruth === "AI" &&
        prediction === "AI"
      ) TP++;

      if (
        record.groundTruth === "HUMAN" &&
        prediction === "HUMAN"
      ) TN++;

      if (
        record.groundTruth === "HUMAN" &&
        prediction === "AI"
      ) FP++;

      if (
        record.groundTruth === "AI" &&
        prediction === "HUMAN"
      ) FN++;
    }

    const decided =
      TP + TN + FP + FN;

    const accuracy =
      decided
        ? (TP + TN) / decided
        : 0;

    const precision =
      TP + FP
        ? TP / (TP + FP)
        : 0;

    const recall =
      TP + FN
        ? TP / (TP + FN)
        : 0;

    const specificity =
      TN + FP
        ? TN / (TN + FP)
        : 0;

    const falsePositiveRate =
      FP + TN
        ? FP / (FP + TN)
        : 0;

    const falseNegativeRate =
      FN + TP
        ? FN / (FN + TP)
        : 0;

    return {

      total: usable.length,

      TP,
      TN,
      FP,
      FN,

      uncertain,

      accuracy:
        Math.round(accuracy * 100),

      precision:
        Math.round(precision * 100),

      recall:
        Math.round(recall * 100),

      specificity:
        Math.round(specificity * 100),

      falsePositiveRate:
        Math.round(
          falsePositiveRate * 100
        ),

      falseNegativeRate:
        Math.round(
          falseNegativeRate * 100
        )
    };
  }

  calibrationTable() {

    const bins = [];

    for (
      let start = 0;
      start < 100;
      start += 10
    ) {

      const end = start + 10;

      const samples =
        this.results.filter(record => {

          if (
            record.groundTruth !== "AI" &&
            record.groundTruth !== "HUMAN"
          ) return false;

          const avg =
            (
              record.models.tmr +
              record.models.modern
            ) / 2;

          return (
            avg >= start &&
            avg < end
          );
        });

      if (!samples.length)
        continue;

      const actualAI =
        samples.filter(
          r =>
            r.groundTruth === "AI"
        ).length;

      bins.push({

        signalRange:
          `${start}-${end}%`,

        samples:
          samples.length,

        actualAIRate:
          Math.round(
            actualAI /
            samples.length *
            100
          )
      });
    }

    return bins;
  }

  report() {

    return {
      version:
        BENCHMARK_VERSION,

      metrics:
        this.calculateMetrics(),

      calibration:
        this.calibrationTable(),

      samples:
        this.results
    };
  }

  exportJSON() {

    const data =
      JSON.stringify(
        this.report(),
        null,
        2
      );

    const blob =
      new Blob(
        [data],
        {
          type:
            "application/json"
        }
      );

    const url =
      URL.createObjectURL(blob);

    const a =
      document.createElement("a");

    a.href = url;

    a.download =
      `AI-Trace-Benchmark-${Date.now()}.json`;

    a.click();

    URL.revokeObjectURL(url);
  }
}

window.AITraceBenchmark =
  AITraceBenchmark;
