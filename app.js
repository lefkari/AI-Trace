/*
  AI TRACE V4.4 — HUMAN EVIDENCE ENGINE
  One-file replacement build
*/
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;

const APP_VERSION = '4.4';
const TMR_MODEL = 'onnx-community/tmr-ai-text-detector-ONNX';
const MODERN_MODEL = 'onnx-community/modernbert-ai-detection-raid-mage-ONNX';
const BENCHMARK_STORAGE = 'aiTraceBenchmarkV44';
const LEGACY_KEYS = ['aiTraceBenchmarkV43','aiTraceBenchmarkV42','aiTraceBenchmarkV41'];
const SCAN_HISTORY_STORAGE = 'aiTraceScanHistoryV44';

let tmrClassifier = null;
let modernClassifier = null;

const $ = id => document.getElementById(id);
const text = $('text');

if (text) {
  text.oninput = () => {
    const words = text.value.trim() ? text.value.trim().split(/\s+/).filter(Boolean).length : 0;
    $('count').textContent = `${words} words`;
  };
}

if ($('clear')) {
  $('clear').onclick = () => {
    text.value = '';
    text.oninput?.();
    $('report')?.classList.add('hidden');
  };
}

if ($('demo')) {
  $('demo').onclick = () => {
    text.value = `Artificial intelligence is rapidly changing the way people work, communicate, and interact with technology. Over the past few years, AI systems have become capable of generating text, creating images, analyzing complex information, and assisting people with tasks that previously required significant amounts of human effort.

One of the most important advantages of artificial intelligence is its ability to process large amounts of information quickly. Organizations can use AI-powered tools to identify patterns, automate repetitive processes, and support better decision-making.

However, the growing use of artificial intelligence also creates important challenges. AI-generated information can sometimes be inaccurate, misleading, or difficult to distinguish from content created by humans.

As these systems become more capable, users will need reliable ways to understand where digital information comes from and how it was produced. The future will therefore require more than simply developing increasingly powerful artificial intelligence systems. Society will also need technologies that provide transparency, verification, and evidence about how digital content was created or modified.`;

    text.oninput?.();
  };
}

if ($('scan')) {
  $('scan').onclick = run;
}

function progress(percent, label) {
  $('progress')?.classList.remove('hidden');

  if ($('bar')) {
    $('bar').style.width = percent + '%';
  }

  if ($('progressText')) {
    $('progressText').textContent = label;
  }
}

function detectLanguage(value) {
  const latin = (value.match(/[A-Za-z]/g) || []).length;
  const total = (value.match(/\p{L}/gu) || []).length;

  if (!total) return 'Unknown';

  return latin / total > 0.8
    ? 'English'
    : 'Non-English';
}

function mean(values) {
  return values.length
    ? values.reduce((a, b) => a + b, 0) / values.length
    : 0;
}

function standardDeviation(values) {
  if (!values.length) return 0;

  const m = mean(values);

  return Math.sqrt(
    mean(
      values.map(
        value => (value - m) ** 2
      )
    )
  );
}

function clamp(value, min, max) {
  return Math.max(
    min,
    Math.min(
      max,
      value
    )
  );
}

function countRepeatedNgrams(words, n) {
  if (words.length < n) return 0;

  const map = new Map();

  for (let i = 0; i <= words.length - n; i++) {
    const gram = words
      .slice(i, i + n)
      .join(' ');

    map.set(
      gram,
      (map.get(gram) || 0) + 1
    );
  }

  let repeated = 0;

  for (const count of map.values()) {
    if (count > 1) {
      repeated += count - 1;
    }
  }

  return repeated;
}

function createProfile(value) {
  const words = value
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  const sentences = value
    .split(/[.!?]+/)
    .map(s => s.trim())
    .filter(Boolean);

  const paragraphs = value
    .split(/\n\s*\n/)
    .map(p => p.trim())
    .filter(Boolean);

  const cleanedWords = words
    .map(
      w =>
        w
          .toLowerCase()
          .replace(/[^\p{L}\p{N}']/gu, '')
    )
    .filter(Boolean);

  const sentenceLengths = sentences.map(
    s => s.split(/\s+/).filter(Boolean).length
  );

  const paragraphLengths = paragraphs.map(
    p => p.split(/\s+/).filter(Boolean).length
  );

  const averageSentenceLength = mean(sentenceLengths);
  const sentenceStd = standardDeviation(sentenceLengths);

  const punctuationCounts = {
    comma: (value.match(/,/g) || []).length,
    semicolon: (value.match(/;/g) || []).length,
    colon: (value.match(/:/g) || []).length,
    dash: (value.match(/[—–-]/g) || []).length,
    quote: (value.match(/["“”‘’']/g) || []).length,
    parentheses: (value.match(/[()]/g) || []).length
  };

  const punctuationTypes = Object.values(
    punctuationCounts
  ).filter(n => n > 0).length;

  const contractions = (
    value.match(/\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s)\b/gi) || []
  ).length;

  const firstPerson = (
    value.match(/\b(I|me|my|mine|we|us|our|ours)\b/gi) || []
  ).length;

  const subjectiveMarkers = (
    value.match(
      /\b(I think|I believe|I suppose|it seems to me|perhaps|maybe|I feel|in my view|I do not know)\b/gi
    ) || []
  ).length;

  const dialogueLines = value
    .split(/\n/)
    .filter(
      line =>
        /^[\s]*["“‘—-]/.test(line) ||
        /["”’][\s]*$/.test(line)
    ).length;

  const openers = sentences
    .map(
      s =>
        s
          .trim()
          .split(/\s+/)
          .slice(0, 2)
          .join(' ')
          .toLowerCase()
          .replace(/[^\p{L}\s]/gu, '')
    )
    .filter(Boolean);

  const openerDiversity =
    new Set(openers).size /
    Math.max(1, openers.length);

  const transitionCount = (
    value.match(
      /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|in conclusion|additionally|nevertheless|on the other hand|as a result)\b/gi
    ) || []
  ).length;

  return {
    words: words.length,
    sentences: sentences.length,
    paragraphs: paragraphs.length,

    averageSentenceLength,
    sentenceStd,
    sentenceVariance: sentenceStd ** 2,

    sentenceBurstiness:
      averageSentenceLength > 0
        ? sentenceStd / averageSentenceLength
        : 0,

    paragraphStd:
      standardDeviation(paragraphLengths),

    lexicalDiversity:
      new Set(cleanedWords).size /
      Math.max(1, cleanedWords.length),

    punctuationCounts,
    punctuationTypes,
    contractions,
    firstPerson,
    subjectiveMarkers,
    dialogueLines,
    openerDiversity,
    transitionCount,

    repeatedTrigrams:
      countRepeatedNgrams(
        cleanedWords,
        3
      )
  };
}

function humanEvidenceEngine(profile) {
  let score = 0;
  const reasons = [];

  if (profile.sentenceBurstiness >= 0.65) {
    score += 22;
    reasons.push('High sentence-length burstiness');
  } else if (profile.sentenceBurstiness >= 0.45) {
    score += 14;
    reasons.push('Moderate sentence-length variation');
  } else if (profile.sentenceBurstiness >= 0.30) {
    score += 7;
  }

  if (profile.punctuationTypes >= 5) {
    score += 15;
    reasons.push('Rich punctuation variety');
  } else if (profile.punctuationTypes >= 3) {
    score += 8;
  }

  const quotes =
    profile.punctuationCounts.quote;

  if (
    quotes >= 8 ||
    profile.dialogueLines >= 2
  ) {
    score += 14;
    reasons.push('Dialogue / quotation structure');
  } else if (quotes >= 3) {
    score += 7;
  }

  if (
    profile.firstPerson >= 3 ||
    profile.subjectiveMarkers >= 2
  ) {
    score += 12;
    reasons.push('Personal or subjective voice');
  } else if (
    profile.firstPerson > 0 ||
    profile.subjectiveMarkers > 0
  ) {
    score += 6;
  }

  if (profile.openerDiversity >= 0.85) {
    score += 12;
    reasons.push('High sentence-opener diversity');
  } else if (profile.openerDiversity >= 0.70) {
    score += 7;
  }

  if (
    profile.paragraphs >= 3 &&
    profile.paragraphStd >= 25
  ) {
    score += 10;
    reasons.push('Irregular paragraph rhythm');
  } else if (
    profile.paragraphs >= 2 &&
    profile.paragraphStd >= 12
  ) {
    score += 5;
  }

  if (profile.lexicalDiversity >= 0.62) {
    score += 8;
  } else if (profile.lexicalDiversity >= 0.50) {
    score += 4;
  }

  if (profile.contractions >= 3) {
    score += 5;
  }

  if (profile.transitionCount >= 4) {
    score -= 8;
  } else if (profile.transitionCount >= 2) {
    score -= 4;
  }

  if (profile.repeatedTrigrams >= 4) {
    score -= 8;
  } else if (profile.repeatedTrigrams >= 2) {
    score -= 4;
  }

  return {
    score:
      clamp(
        Math.round(score),
        0,
        100
      ),

    reasons
  };
}

function chunkText(
  value,
  maxCharacters = 1450
) {
  const sentences =
    value.match(
      /[^.!?]+[.!?]+|[^.!?]+$/g
    ) || [value];

  const chunks = [];
  let current = '';

  for (const sentence of sentences) {
    if (
      (current + sentence).length >
        maxCharacters &&
      current
    ) {
      chunks.push(
        current.trim()
      );

      current = sentence;
    } else {
      current += sentence;
    }
  }

  if (current.trim()) {
    chunks.push(
      current.trim()
    );
  }

  return chunks
    .filter(Boolean)
    .slice(0, 8);
}

async function loadTMR() {
  if (tmrClassifier) {
    return tmrClassifier;
  }

  $('modelState').textContent =
    'Loading TMR engine…';

  progress(
    10,
    'Loading Quick Scan model…'
  );

  tmrClassifier =
    await pipeline(
      'text-classification',
      TMR_MODEL,
      {
        dtype: 'q8'
      }
    );

  return tmrClassifier;
}

async function loadModernBERT() {
  if (modernClassifier) {
    return modernClassifier;
  }

  $('modelState').textContent =
    'Loading Deep Scan engine…';

  progress(
    55,
    'Loading second detector…'
  );

  modernClassifier =
    await pipeline(
      'text-classification',
      MODERN_MODEL,
      {
        dtype: 'q4f16'
      }
    );

  return modernClassifier;
}

function extractAIProbability(output) {
  const results =
    (
      Array.isArray(output)
        ? output
        : [output]
    ).flat();

  let ai = null;
  let human = null;

  for (const result of results) {
    const label =
      String(
        result.label || ''
      ).toLowerCase();

    const score =
      Number(result.score) || 0;

    if (
      label.includes('ai') ||
      label.includes('machine') ||
      label.includes('generated') ||
      label === 'label_1'
    ) {
      ai =
        Math.max(
          ai ?? 0,
          score
        );
    }

    if (
      label.includes('human') ||
      label === 'label_0'
    ) {
      human =
        Math.max(
          human ?? 0,
          score
        );
    }
  }

  if (ai !== null) {
    return ai;
  }

  if (human !== null) {
    return 1 - human;
  }

  if (results.length >= 2) {
    return Number(
      results[1]?.score ?? 0.5
    );
  }

  return 0.5;
}

async function classify(
  classifier,
  value
) {
  const output =
    await classifier(
      value,
      {
        top_k: null,
        truncation: true
      }
    );

  return Math.round(
    extractAIProbability(output) *
    100
  );
}

function calculateEvidenceQuality({
  modelGap,
  segmentDeviation,
  segmentRange,
  profile,
  language,
  activeModels
}) {
  const modelAgreement =
    clamp(
      100 -
      modelGap * 2.1,
      0,
      100
    );

  const segmentConsistency =
    clamp(
      100 -
      (
        segmentDeviation * 1.45 +
        segmentRange * 0.32
      ),
      0,
      100
    );

  const lengthQuality =
    profile.words >= 250
      ? 100
      : profile.words >= 150
        ? 85
        : profile.words >= 100
          ? 65
          : 40;

  const languageQuality =
    language === 'English'
      ? 100
      : 45;

  const modelQuality =
    activeModels === 2
      ? 100
      : 40;

  return Math.round(
    modelAgreement * 0.30 +
    segmentConsistency * 0.35 +
    lengthQuality * 0.12 +
    languageQuality * 0.13 +
    modelQuality * 0.10
  );
}

function buildConsensus({
  tmr,
  modern,
  segmentScores,
  profile,
  language,
  tmrWorked,
  modernWorked,
  humanEvidence
}) {
  const activeModels =
    Number(tmrWorked) +
    Number(modernWorked);

  const modelGap =
    (
      tmrWorked &&
      modernWorked
    )
      ? Math.abs(
          tmr -
          modern
        )
      : 0;

  const segmentDeviation =
    Math.round(
      standardDeviation(
        segmentScores
      )
    );

  const segmentRange =
    segmentScores.length
      ? Math.max(
          ...segmentScores
        ) -
        Math.min(
          ...segmentScores
        )
      : 0;

  let rawSignal;

  if (
    tmrWorked &&
    modernWorked
  ) {
    rawSignal =
      Math.round(
        tmr * 0.52 +
        modern * 0.48
      );
  } else if (tmrWorked) {
    rawSignal = tmr;
  } else if (modernWorked) {
    rawSignal = modern;
  } else {
    rawSignal = 50;
  }

  const evidenceQuality =
    calculateEvidenceQuality({
      modelGap,
      segmentDeviation,
      segmentRange,
      profile,
      language,
      activeModels
    });

  const counterWeight =
    0.12 +
    (
      (100 - evidenceQuality) /
      100
    ) *
    0.62;

  const humanPenalty =
    humanEvidence.score *
    counterWeight *
    (
      rawSignal /
      100
    );

  let calibratedScore =
    Math.round(
      rawSignal -
      humanPenalty
    );

  if (
    humanEvidence.score >= 60 &&
    (
      segmentDeviation >= 22 ||
      segmentRange >= 55
    )
  ) {
    calibratedScore =
      Math.min(
        calibratedScore,
        69
      );
  }

  calibratedScore =
    clamp(
      calibratedScore,
      0,
      100
    );

  const directionalConflict =
    (
      rawSignal >= 70 &&
      humanEvidence.score >= 55
    )
      ? Math.min(
          20,
          (
            humanEvidence.score -
            50
          ) *
          0.40
        )
      : 0;

  const uncertainty =
    clamp(
      Math.round(
        100 -
        evidenceQuality +
        directionalConflict
      ),
      5,
      95
    );

  const confidence =
    100 -
    uncertainty;

  const modelsConflict =
    modelGap >= 35;

  const unstableSegments =
    segmentDeviation >= 28 ||
    segmentRange >= 70;

  const humanOverrideGuard =
    humanEvidence.score >= 60 &&
    rawSignal >= 70;

  let verdict =
    'INCONCLUSIVE';

  if (
    activeModels < 2 ||
    language !== 'English' ||
    modelsConflict
  ) {
    verdict =
      'INCONCLUSIVE';

  } else if (
    calibratedScore >= 85 &&
    evidenceQuality >= 75 &&
    humanEvidence.score < 55 &&
    segmentDeviation < 22 &&
    segmentRange < 55
  ) {
    verdict =
      'Strong AI evidence';

  } else if (
    calibratedScore >= 72 &&
    evidenceQuality >= 60 &&
    humanEvidence.score < 60 &&
    segmentDeviation < 26 &&
    segmentRange < 65
  ) {
    verdict =
      'Likely AI';

  } else if (
    calibratedScore <= 18 &&
    humanEvidence.score >= 60 &&
    evidenceQuality >= 55
  ) {
    verdict =
      'Strong human evidence';

  } else if (
    calibratedScore <= 35 &&
    humanEvidence.score >= 50
  ) {
    verdict =
      'Likely human';
  }

  if (
    humanOverrideGuard &&
    (
      unstableSegments ||
      evidenceQuality < 65
    )
  ) {
    verdict =
      'INCONCLUSIVE';
  }

  return {
    rawSignal,
    calibratedScore,
    evidenceQuality,
    uncertainty,
    confidence,
    verdict,
    modelGap,
    segmentDeviation,
    segmentRange,
    activeModels,
    modelsConflict,
    unstableSegments,
    humanOverrideGuard,

    humanPenalty:
      Math.round(
        humanPenalty
      )
  };
}

function loadJSON(key) {
  try {
    const raw =
      localStorage.getItem(key);

    return raw
      ? JSON.parse(raw)
      : [];
  } catch {
    return [];
  }
}

function normalizeLegacyRecord(record) {
  return {
    ...record,

    appVersion:
      record.appVersion ||
      record.version ||
      'legacy',

    calibratedScore:
      record.calibratedScore ??
      record.score ??
      record.rawSignal ??
      50,

    evidenceQuality:
      record.evidenceQuality ??
      record.confidence ??
      50,

    humanEvidence:
      record.humanEvidence ??
      null
  };
}

function loadBenchmark() {
  let records =
    loadJSON(
      BENCHMARK_STORAGE
    );

  if (records.length) {
    return records;
  }

  for (const key of LEGACY_KEYS) {
    const legacy =
      loadJSON(key);

    if (legacy.length) {
      records =
        legacy.map(
          normalizeLegacyRecord
        );

      saveBenchmark(records);

      return records;
    }
  }

  return [];
}

function saveBenchmark(records) {
  try {
    localStorage.setItem(
      BENCHMARK_STORAGE,
      JSON.stringify(records)
    );
  } catch (error) {
    console.warn(
      'Benchmark save failed',
      error
    );
  }
}

function nextBenchmarkID(
  truth,
  records
) {
  const prefix = {
    AI: 'A',
    HUMAN: 'H',
    MIXED: 'M',
    UNCERTAIN: 'U'
  }[truth] || 'X';

  const count =
    records.filter(
      r =>
        r.groundTruth === truth
    ).length + 1;

  return (
    prefix +
    '-' +
    String(count).padStart(
      3,
      '0'
    )
  );
}

function addBenchmark({
  groundTruth,
  source,
  profile,
  language,
  tmr,
  modern,
  tmrWorked,
  modernWorked,
  segmentScores,
  humanEvidence,
  consensus
}) {
  const records =
    loadBenchmark();

  const id =
    nextBenchmarkID(
      groundTruth,
      records
    );

  records.push({
    id,

    appVersion:
      APP_VERSION,

    timestamp:
      new Date()
        .toISOString(),

    groundTruth,

    source,

    words:
      profile.words,

    language,

    models: {
      tmr,
      modern,
      tmrWorked,
      modernWorked
    },

    segments:
      segmentScores,

    humanEvidence:
      humanEvidence.score,

    humanEvidenceReasons:
      humanEvidence.reasons,

    rawSignal:
      consensus.rawSignal,

    calibratedScore:
      consensus.calibratedScore,

    evidenceQuality:
      consensus.evidenceQuality,

    verdict:
      consensus.verdict,

    confidence:
      consensus.confidence,

    uncertainty:
      consensus.uncertainty,

    modelGap:
      consensus.modelGap,

    segmentDeviation:
      consensus.segmentDeviation,

    segmentRange:
      consensus.segmentRange
  });

  saveBenchmark(records);

  return id;
}

function recordPrediction(record) {
  const verdict =
    record.verdict || '';

  if (
    verdict === 'Strong AI evidence' ||
    verdict === 'Likely AI'
  ) {
    return 'AI';
  }

  if (
    verdict === 'Strong human evidence' ||
    verdict === 'Likely human'
  ) {
    return 'HUMAN';
  }

  return 'ABSTAIN';
}

function pct(value) {
  return Math.round(
    value * 100
  );
}

function calculateBenchmarkMetrics() {
  const records =
    loadBenchmark()
      .filter(
        r =>
          r.groundTruth === 'AI' ||
          r.groundTruth === 'HUMAN'
      );

  let TP = 0;
  let TN = 0;
  let FP = 0;
  let FN = 0;

  let aiAbstain = 0;
  let humanAbstain = 0;

  for (const record of records) {
    const prediction =
      recordPrediction(record);

    if (
      prediction === 'ABSTAIN'
    ) {
      if (
        record.groundTruth === 'AI'
      ) {
        aiAbstain++;
      } else {
        humanAbstain++;
      }

      continue;
    }

    if (
      record.groundTruth === 'AI' &&
      prediction === 'AI'
    ) {
      TP++;
    }

    if (
      record.groundTruth === 'HUMAN' &&
      prediction === 'HUMAN'
    ) {
      TN++;
    }

    if (
      record.groundTruth === 'HUMAN' &&
      prediction === 'AI'
    ) {
      FP++;
    }

    if (
      record.groundTruth === 'AI' &&
      prediction === 'HUMAN'
    ) {
      FN++;
    }
  }

  const totalAI =
    records.filter(
      r =>
        r.groundTruth === 'AI'
    ).length;

  const totalHuman =
    records.filter(
      r =>
        r.groundTruth === 'HUMAN'
    ).length;

  const total =
    records.length;

  const decided =
    TP +
    TN +
    FP +
    FN;

  const abstained =
    aiAbstain +
    humanAbstain;

  return {
    total,
    totalAI,
    totalHuman,

    TP,
    TN,
    FP,
    FN,

    aiAbstain,
    humanAbstain,
    abstained,
    decided,

    coverage:
      pct(
        total
          ? decided / total
          : 0
      ),

    selectiveAccuracy:
      pct(
        decided
          ? (
              TP +
              TN
            ) /
            decided
          : 0
      ),

    precision:
      pct(
        TP + FP
          ? TP /
            (
              TP +
              FP
            )
          : 0
      ),

    recall:
      pct(
        totalAI
          ? TP /
            totalAI
          : 0
      ),

    specificity:
      pct(
        totalHuman
          ? TN /
            totalHuman
          : 0
      ),

    falsePositiveRate:
      pct(
        totalHuman
          ? FP /
            totalHuman
          : 0
      ),

    falseNegativeRate:
      pct(
        totalAI
          ? FN /
            totalAI
          : 0
      ),

    aiAbstentionRate:
      pct(
        totalAI
          ? aiAbstain /
            totalAI
          : 0
      ),

    humanAbstentionRate:
      pct(
        totalHuman
          ? humanAbstain /
            totalHuman
          : 0
      )
  };
}

function experimentalThresholdSearch() {
  const records =
    loadBenchmark()
      .filter(
        r =>
          r.groundTruth === 'AI' ||
          r.groundTruth === 'HUMAN'
      );

  const aiCount =
    records.filter(
      r =>
        r.groundTruth === 'AI'
    ).length;

  const humanCount =
    records.filter(
      r =>
        r.groundTruth === 'HUMAN'
    ).length;

  if (
    aiCount < 10 ||
    humanCount < 10
  ) {
    return {
      ready: false,

      message:
        'At least 10 HUMAN and 10 AI samples are required before threshold search begins.'
    };
  }

  let best = null;

  for (
    let aiThreshold = 70;
    aiThreshold <= 92;
    aiThreshold += 2
  ) {
    for (
      let humanThreshold = 15;
      humanThreshold <= 40;
      humanThreshold += 5
    ) {
      for (
        let minEvidence = 45;
        minEvidence <= 80;
        minEvidence += 5
      ) {
        let TP = 0;
        let TN = 0;
        let FP = 0;
        let FN = 0;
        let abstain = 0;

        for (const record of records) {
          const score =
            record.calibratedScore ??
            record.rawSignal ??
            50;

          const evidence =
            record.evidenceQuality ??
            record.confidence ??
            50;

          let prediction =
            'ABSTAIN';

          if (
            evidence >= minEvidence
          ) {
            if (
              score >= aiThreshold
            ) {
              prediction = 'AI';
            } else if (
              score <= humanThreshold
            ) {
              prediction =
                'HUMAN';
            }
          }

          if (
            prediction === 'ABSTAIN'
          ) {
            abstain++;

            continue;
          }

          if (
            record.groundTruth === 'AI' &&
            prediction === 'AI'
          ) {
            TP++;
          }

          if (
            record.groundTruth === 'HUMAN' &&
            prediction === 'HUMAN'
          ) {
            TN++;
          }

          if (
            record.groundTruth === 'HUMAN' &&
            prediction === 'AI'
          ) {
            FP++;
          }

          if (
            record.groundTruth === 'AI' &&
            prediction === 'HUMAN'
          ) {
            FN++;
          }
        }

        const total =
          records.length;

        const decided =
          TP +
          TN +
          FP +
          FN;

        const coverage =
          total
            ? decided / total
            : 0;

        const accuracy =
          decided
            ? (
                TP +
                TN
              ) /
              decided
            : 0;

        const objective =
          accuracy * 100 +
          coverage * 25 -
          FP * 18 -
          FN * 10 -
          abstain * 1.2;

        if (
          !best ||
          objective >
            best.objective
        ) {
          best = {
            ready: true,

            aiThreshold,
            humanThreshold,
            minEvidence,

            TP,
            TN,
            FP,
            FN,
            abstain,

            coverage:
              pct(coverage),

            selectiveAccuracy:
              pct(accuracy),

            objective:
              Math.round(
                objective * 10
              ) /
              10
          };
        }
      }
    }
  }

  return best;
}

function saveScanHistory(result) {
  try {
    const history =
      loadJSON(
        SCAN_HISTORY_STORAGE
      );

    history.push({
      timestamp:
        new Date()
          .toISOString(),

      ...result
    });

    localStorage.setItem(
      SCAN_HISTORY_STORAGE,
      JSON.stringify(
        history.slice(-100)
      )
    );
  } catch (error) {
    console.warn(
      'Scan history save failed',
      error
    );
  }
}

function askBenchmarkLabel(scanData) {
  const answer =
    prompt(
`AI TRACE BENCHMARK V4.4

Do you KNOW the true origin of this text?

Type:
AI        = definitely AI-generated
HUMAN     = definitely human-written
MIXED     = known mixture of human + AI
UNCERTAIN = origin is not known with certainty

Leave empty / Cancel to skip.`
    );

  if (!answer) {
    return;
  }

  const truth =
    answer
      .trim()
      .toUpperCase();

  const allowed = [
    'AI',
    'HUMAN',
    'MIXED',
    'UNCERTAIN'
  ];

  if (
    !allowed.includes(truth)
  ) {
    alert(
      'Benchmark not saved. Use AI, HUMAN, MIXED or UNCERTAIN.'
    );

    return;
  }

  const source =
    prompt(
      'Source / note for this sample:',

      truth === 'AI'
        ? 'Known AI-generated sample'
        : truth === 'HUMAN'
          ? 'Known human-written sample'
          : 'Known benchmark sample'
    ) || '';

  const id =
    addBenchmark({
      groundTruth:
        truth,

      source,

      ...scanData
    });

  const metrics =
    calculateBenchmarkMetrics();

  alert(
`Benchmark saved: ${id}

Known binary samples: ${metrics.total}
AI: ${metrics.totalAI}
HUMAN: ${metrics.totalHuman}

TP: ${metrics.TP}
TN: ${metrics.TN}
FP: ${metrics.FP}
FN: ${metrics.FN}

AI abstentions: ${metrics.aiAbstain}
Human abstentions: ${metrics.humanAbstain}

Coverage: ${metrics.coverage}%
Selective accuracy: ${metrics.selectiveAccuracy}%
Precision: ${metrics.precision}%

AI recall: ${metrics.recall}%
Human specificity: ${metrics.specificity}%

False Positive Rate: ${metrics.falsePositiveRate}%
False Negative Rate: ${metrics.falseNegativeRate}%

Development metrics only — not production accuracy claims.`
  );

  renderBenchmarkPanel();
}

async function run() {
  const value =
    text.value.trim();

  const wordCount =
    value
      ? value
          .split(/\s+/)
          .filter(Boolean)
          .length
      : 0;

  if (wordCount < 80) {
    alert(
      'Paste at least 80 words for a meaningful analysis.'
    );

    return;
  }

  $('scan').disabled =
    true;

  progress(
    3,
    'Building document profile…'
  );

  const profile =
    createProfile(value);

  const language =
    detectLanguage(value);

  const humanEvidence =
    humanEvidenceEngine(
      profile
    );

  const chunks =
    chunkText(value);

  let tmrDocument = 50;
  let modernDocument = 50;

  let tmrWorked = true;
  let modernWorked = true;

  const tmrSegments = [];

  try {
    const tmr =
      await loadTMR();

    progress(
      20,
      'TMR Quick Scan…'
    );

    tmrDocument =
      await classify(
        tmr,
        value
      );

    for (
      let i = 0;
      i < chunks.length;
      i++
    ) {
      progress(
        25 +
        Math.round(
          (
            i /
            Math.max(
              1,
              chunks.length
            )
          ) *
          25
        ),

        `TMR segment ${i + 1}/${chunks.length}`
      );

      tmrSegments.push(
        await classify(
          tmr,
          chunks[i]
        )
      );
    }
  } catch (error) {
    console.error(
      'TMR error:',
      error
    );

    tmrWorked = false;

    for (
      let i = 0;
      i < chunks.length;
      i++
    ) {
      tmrSegments.push(50);
    }
  }

  try {
    const modern =
      await loadModernBERT();

    progress(
      70,
      'ModernBERT Deep Scan…'
    );

    modernDocument =
      await classify(
        modern,
        value
      );
  } catch (error) {
    console.error(
      'ModernBERT error:',
      error
    );

    modernWorked = false;
  }

  progress(
    87,
    'Combining AI and human evidence…'
  );

  const consensus =
    buildConsensus({
      tmr:
        tmrDocument,

      modern:
        modernDocument,

      segmentScores:
        tmrSegments,

      profile,

      language,

      tmrWorked,

      modernWorked,

      humanEvidence
    });

  renderV44({
    consensus,
    profile,
    chunks,

    segmentScores:
      tmrSegments,

    language,
    tmrDocument,
    modernDocument,
    tmrWorked,
    modernWorked,
    humanEvidence
  });

  saveScanHistory({
    version:
      APP_VERSION,

    words:
      profile.words,

    language,

    tmr:
      tmrDocument,

    modern:
      modernDocument,

    humanEvidence:
      humanEvidence.score,

    rawSignal:
      consensus.rawSignal,

    calibratedScore:
      consensus.calibratedScore,

    evidenceQuality:
      consensus.evidenceQuality,

    verdict:
      consensus.verdict,

    confidence:
      consensus.confidence,

    uncertainty:
      consensus.uncertainty,

    modelGap:
      consensus.modelGap,

    segmentDeviation:
      consensus.segmentDeviation,

    segmentRange:
      consensus.segmentRange
  });

  progress(
    100,
    'Trace complete'
  );

  $('modelState').textContent =
    (
      tmrWorked &&
      modernWorked
    )
      ? 'V4.4 Human Evidence Engine ready ✓'
      : 'Limited evidence mode';

  setTimeout(
    () => {
      $('progress')
        ?.classList
        .add('hidden');
    },
    500
  );

  $('scan').disabled =
    false;

  setTimeout(
    () => {
      askBenchmarkLabel({
        profile,
        language,

        tmr:
          tmrDocument,

        modern:
          modernDocument,

        tmrWorked,
        modernWorked,

        segmentScores:
          tmrSegments,

        humanEvidence,

        consensus
      });
    },
    700
  );
}

function escapeHTML(value) {
  return value.replace(
    /[&<>"']/g,

    character =>
      ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
      })[character]
  );
}

function renderV44({
  consensus,
  profile,
  chunks,
  segmentScores,
  language,
  tmrDocument,
  modernDocument,
  tmrWorked,
  modernWorked,
  humanEvidence
}) {
  $('report')
    ?.classList
    .remove('hidden');

  if ($('score')) {
    $('score').textContent =
      consensus.calibratedScore +
      '%';
  }

  if ($('scaleFill')) {
    $('scaleFill').style.width =
      consensus.calibratedScore +
      '%';
  }

  if ($('verdict')) {
    $('verdict').textContent =
      consensus.verdict;
  }

  const confidenceLabel =
    consensus.confidence >= 75
      ? 'High'
      : consensus.confidence >= 50
        ? 'Medium'
        : 'Low';

  if ($('confidence')) {
    $('confidence').textContent =
      `Confidence: ${confidenceLabel} (${consensus.confidence}%)`;
  }

  if ($('explain')) {
    $('explain').textContent =
`Raw AI detector signal: ${consensus.rawSignal}%. Calibrated score: ${consensus.calibratedScore}%. Evidence quality: ${consensus.evidenceQuality}%. Human counter-evidence: ${humanEvidence.score}%. TMR: ${tmrDocument}%. ModernBERT: ${modernDocument}%. Model disagreement: ${consensus.modelGap} points. Segment range: ${consensus.segmentRange} points.`;
  }

  const aiDisplay =
    consensus.calibratedScore;

  const humanDisplay =
    clamp(
      Math.round(
        humanEvidence.score *
        0.70 +
        (
          100 -
          consensus.calibratedScore
        ) *
        0.30
      ),
      0,
      100
    );

  if ($('humanVal')) {
    $('humanVal').textContent =
      humanDisplay + '%';
  }

  if ($('aiVal')) {
    $('aiVal').textContent =
      aiDisplay + '%';
  }

  if ($('uncertainVal')) {
    $('uncertainVal').textContent =
      consensus.uncertainty +
      '%';
  }

  if ($('humanBar')) {
    $('humanBar').style.width =
      humanDisplay + '%';
  }

  if ($('aiBar')) {
    $('aiBar').style.width =
      aiDisplay + '%';
  }

  if ($('uncertainBar')) {
    $('uncertainBar').style.width =
      consensus.uncertainty +
      '%';
  }

  if ($('engineBadge')) {
    $('engineBadge').textContent =
      (
        tmrWorked &&
        modernWorked
      )
        ? 'V4.4 • HUMAN EVIDENCE ENGINE'
        : 'LIMITED EVIDENCE';
  }

  const humanReasons =
    humanEvidence.reasons.length
      ? humanEvidence.reasons
          .slice(0, 4)
          .join(' • ')
      : 'No strong human-style counter-signals detected';

  const evidence = [
    [
      'Raw detector signal',
      `${consensus.rawSignal}%`,
      'Raw'
    ],

    [
      'Calibrated score',
      `${consensus.calibratedScore}%`,
      'Adjusted'
    ],

    [
      'Evidence quality',
      `${consensus.evidenceQuality}%`,

      consensus.evidenceQuality >= 75
        ? 'Strong'
        : consensus.evidenceQuality >= 50
          ? 'Medium'
          : 'Weak'
    ],

    [
      'Human evidence',
      `${humanEvidence.score}% — ${humanReasons}`,

      humanEvidence.score >= 60
        ? 'Strong counter-evidence'
        : humanEvidence.score >= 40
          ? 'Moderate'
          : 'Low'
    ],

    [
      'TMR detector',
      tmrWorked
        ? `${tmrDocument}% AI signal`
        : 'Unavailable',
      'Model A'
    ],

    [
      'ModernBERT detector',
      modernWorked
        ? `${modernDocument}% AI signal`
        : 'Unavailable',
      'Model B'
    ],

    [
      'Model disagreement',
      `${consensus.modelGap} points`,

      consensus.modelGap >= 35
        ? 'High conflict'
        : 'Acceptable'
    ],

    [
      'Segment deviation',
      `${consensus.segmentDeviation}`,

      consensus.segmentDeviation >= 28
        ? 'Unstable'
        : 'Stable'
    ],

    [
      'Segment range',
      `${consensus.segmentRange} points`,

      consensus.segmentRange >= 70
        ? 'High variation'
        : 'Acceptable'
    ],

    [
      'Language fit',

      language === 'English'
        ? 'English detected — strongest supported path.'
        : 'Non-English detected — reliability is reduced.',

      'Context'
    ]
  ];

  if ($('evidence')) {
    $('evidence').innerHTML =
      evidence
        .map(
          item => `
<div class="ev">
  <div class="evTop">
    <span>${item[0]}</span>
    <span>${item[2]}</span>
  </div>
  <small>${item[1]}</small>
</div>`
        )
        .join('');
  }

  const metrics = {
    Words:
      profile.words,

    Sentences:
      profile.sentences,

    'Avg. words / sentence':
      profile
        .averageSentenceLength
        .toFixed(1),

    'Sentence burstiness':
      profile
        .sentenceBurstiness
        .toFixed(2),

    'Lexical diversity':
      Math.round(
        profile.lexicalDiversity *
        100
      ) + '%',

    'Human evidence':
      humanEvidence.score + '%',

    Language:
      language,

    'Models active':
      `${consensus.activeModels}/2`,

    'Raw signal':
      `${consensus.rawSignal}%`,

    'Calibrated score':
      `${consensus.calibratedScore}%`,

    'Evidence quality':
      `${consensus.evidenceQuality}%`,

    'Model disagreement':
      `${consensus.modelGap} pts`,

    'Segment deviation':
      consensus.segmentDeviation,

    'Segment range':
      `${consensus.segmentRange} pts`
  };

  if ($('metrics')) {
    $('metrics').innerHTML =
      Object.entries(metrics)
        .map(
          ([key, value]) => `
<div class="metric">
  <span>${key}</span>
  <b>${value}</b>
</div>`
        )
        .join('');
  }

  if ($('segments')) {
    $('segments').innerHTML =
      chunks
        .map(
          (chunk, index) => {
            const score =
              segmentScores[index] ?? 50;

            return `
<div class="segment">
  <div class="segmentHead">
    <b>Segment ${index + 1}</b>
    <span>${score}% TMR signal</span>
  </div>

  <div class="segmentMeter">
    <i style="width:${score}%"></i>
  </div>

  <p>
    ${escapeHTML(
      chunk.slice(0, 300)
    )}
    ${chunk.length > 300 ? '…' : ''}
  </p>
</div>`;
          }
        )
        .join('');
  }

  $('report')
    ?.scrollIntoView({
      behavior:
        'smooth',

      block:
        'start'
    });

  renderBenchmarkPanel();
}

function renderBenchmarkPanel() {
  if (!$('report')) {
    return;
  }

  let panel =
    document.getElementById(
      'benchmarkPanelV44'
    );

  if (!panel) {
    panel =
      document.createElement(
        'section'
      );

    panel.id =
      'benchmarkPanelV44';

    panel.className =
      'panel';

    panel.style.marginTop =
      '18px';

    $('report')
      .appendChild(panel);
  }

  const records =
    loadBenchmark();

  const binary =
    records.filter(
      r =>
        r.groundTruth === 'AI' ||
        r.groundTruth === 'HUMAN'
    );

  const mixed =
    records.filter(
      r =>
        r.groundTruth === 'MIXED'
    ).length;

  const unknown =
    records.filter(
      r =>
        r.groundTruth === 'UNCERTAIN'
    ).length;

  const metrics =
    calculateBenchmarkMetrics();

  const search =
    experimentalThresholdSearch();

  const recordsHTML =
    records
      .slice()
      .reverse()
      .slice(0, 20)
      .map(
        record => `
<div class="ev">
  <div class="evTop">
    <span>${record.id || 'Record'}</span>
    <span>${record.groundTruth || '?'}</span>
  </div>

  <small>
    Calibrated: ${record.calibratedScore ?? record.rawSignal ?? '?'}%
    · Evidence: ${record.evidenceQuality ?? '?'}%
    · Human evidence: ${record.humanEvidence ?? 'n/a'}%
    · TMR: ${record.models?.tmr ?? '?'}%
    · Modern: ${record.models?.modern ?? '?'}%
    · Verdict: ${record.verdict ?? 'legacy'}
  </small>
</div>`
      )
      .join('');

  const searchHTML =
    search.ready
      ? `
<div class="ev">
  <div class="evTop">
    <span>Experimental thresholds</span>
    <span>DEV ONLY</span>
  </div>

  <small>
    AI threshold: ${search.aiThreshold}% ·
    Human threshold: ${search.humanThreshold}% ·
    Minimum evidence: ${search.minEvidence}% ·
    Coverage: ${search.coverage}% ·
    Selective accuracy: ${search.selectiveAccuracy}% ·
    FP: ${search.FP} · FN: ${search.FN}.
    Do not treat these as production thresholds.
  </small>
</div>`
      : `
<div class="ev">
  <div class="evTop">
    <span>Experimental Calibration</span>
    <span>Not ready</span>
  </div>

  <small>${search.message}</small>
</div>`;

  panel.innerHTML = `
<span class="over">
  V4.4 BENCHMARK • DEVELOPMENT ONLY
</span>

<h2>
  Benchmark Results
</h2>

<p class="sub">
  Abstentions are tracked separately. Metrics below are development measurements from known samples stored on this device; they are not production accuracy claims.
</p>

<div class="metrics">
  <div class="metric"><span>Total records</span><b>${records.length}</b></div>
  <div class="metric"><span>Binary samples</span><b>${binary.length}</b></div>
  <div class="metric"><span>AI</span><b>${metrics.totalAI}</b></div>
  <div class="metric"><span>Human</span><b>${metrics.totalHuman}</b></div>
  <div class="metric"><span>Mixed</span><b>${mixed}</b></div>
  <div class="metric"><span>Uncertain ground truth</span><b>${unknown}</b></div>
</div>

<h3>Confusion Matrix + Abstention</h3>

<div class="metrics">
  <div class="metric"><span>True positive</span><b>${metrics.TP}</b></div>
  <div class="metric"><span>True negative</span><b>${metrics.TN}</b></div>
  <div class="metric"><span>False positive</span><b>${metrics.FP}</b></div>
  <div class="metric"><span>False negative</span><b>${metrics.FN}</b></div>
  <div class="metric"><span>AI abstentions</span><b>${metrics.aiAbstain}</b></div>
  <div class="metric"><span>Human abstentions</span><b>${metrics.humanAbstain}</b></div>
</div>

<h3>Performance</h3>

<div class="metrics">
  <div class="metric"><span>Coverage</span><b>${metrics.coverage}%</b></div>
  <div class="metric"><span>Selective accuracy</span><b>${metrics.selectiveAccuracy}%</b></div>
  <div class="metric"><span>Precision</span><b>${metrics.precision}%</b></div>
  <div class="metric"><span>AI recall</span><b>${metrics.recall}%</b></div>
  <div class="metric"><span>Human specificity</span><b>${metrics.specificity}%</b></div>
  <div class="metric"><span>False positive rate</span><b>${metrics.falsePositiveRate}%</b></div>
  <div class="metric"><span>False negative rate</span><b>${metrics.falseNegativeRate}%</b></div>
  <div class="metric"><span>AI abstention rate</span><b>${metrics.aiAbstentionRate}%</b></div>
  <div class="metric"><span>Human abstention rate</span><b>${metrics.humanAbstentionRate}%</b></div>
</div>

<h3>Experimental Calibration</h3>

<div class="evidence">
  ${searchHTML}
</div>

<h3>Benchmark Records</h3>

<div class="evidence">
  ${
    recordsHTML ||
    '<div class="ev"><small>No benchmark records yet.</small></div>'
  }
</div>
`;
}

window.AITraceBenchmark = {
  report() {
    return {
      version:
        APP_VERSION,

      metrics:
        calculateBenchmarkMetrics(),

      thresholdSearch:
        experimentalThresholdSearch(),

      samples:
        loadBenchmark()
    };
  },

  exportJSON() {
    const json =
      JSON.stringify(
        this.report(),
        null,
        2
      );

    const blob =
      new Blob(
        [json],
        {
          type:
            'application/json'
        }
      );

    const url =
      URL.createObjectURL(
        blob
      );

    const anchor =
      document.createElement(
        'a'
      );

    anchor.href =
      url;

    anchor.download =
      `AI-Trace-V44-Benchmark-${Date.now()}.json`;

    anchor.click();

    URL.revokeObjectURL(
      url
    );
  },

  clear() {
    const confirmation =
      confirm(
        'Delete all V4.4 benchmark records on this device?'
      );

    if (!confirmation) {
      return;
    }

    localStorage.removeItem(
      BENCHMARK_STORAGE
    );

    renderBenchmarkPanel();

    alert(
      'V4.4 benchmark data deleted.'
    );
  },

  history() {
    return loadJSON(
      SCAN_HISTORY_STORAGE
    );
  }
};

setTimeout(
  renderBenchmarkPanel,
  400
);
