/*
  AI TRACE V6.1 — EVIDENCE SUFFICIENCY + RELIABILITY LAB
  Full replacement app.js

  Features:
  - TMR detector
  - E5-small detector
  - Conditional ModernBERT judge on desktop
  - Human counter-evidence
  - Domain routing
  - Segment analysis
  - Outlier defense
  - Evidence Sufficiency Gate
  - Reliability-weighted consensus
  - Benchmark lab
  - False-positive / false-negative / abstention inspectors
  - JSON / CSV export
  - Local scan history
  - Zero paid API
*/

import {
  pipeline,
  env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;

const VERSION = '6.1';

const MODEL_TMR =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODEL_E5 =
  'onnx-community/e5-small-lora-ai-generated-detector-ONNX';

const MODEL_MODERN =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';

const BENCH_KEY =
  'aiTraceBenchmarkV61';

const HISTORY_KEY =
  'aiTraceHistoryV61';

const LEGACY_BENCH_KEYS = [
  'aiTraceBenchmarkV6',
  'aiTraceBenchmarkV54',
  'aiTraceBenchmarkV53',
  'aiTraceBenchmarkV52',
  'aiTraceBenchmarkV51',
  'aiTraceBenchmarkV44'
];

let tmrModel = null;
let e5Model = null;
let modernModel = null;

const $ = id =>
  document.getElementById(id);

const textEl =
  $('text');

/* =========================================================
   DEVICE
========================================================= */

function isMobileDevice() {
  return (
    window.matchMedia(
      '(max-width: 768px)'
    ).matches ||
    /Android|iPhone|iPad|iPod/i.test(
      navigator.userAgent
    )
  );
}

/* =========================================================
   BASIC HELPERS
========================================================= */

function clamp(
  value,
  min = 0,
  max = 100
) {
  return Math.max(
    min,
    Math.min(
      max,
      value
    )
  );
}

function average(values) {
  const usable =
    values.filter(
      Number.isFinite
    );

  if (!usable.length) {
    return 0;
  }

  return usable.reduce(
    (sum, value) =>
      sum + value,
    0
  ) / usable.length;
}

function weightedAverage(
  values,
  weights
) {
  let numerator = 0;
  let denominator = 0;

  for (
    let i = 0;
    i < values.length;
    i++
  ) {
    const value =
      values[i];

    const weight =
      weights[i];

    if (
      Number.isFinite(value) &&
      Number.isFinite(weight) &&
      weight > 0
    ) {
      numerator +=
        value * weight;

      denominator +=
        weight;
    }
  }

  return denominator
    ? numerator / denominator
    : 50;
}

function standardDeviation(
  values
) {
  const usable =
    values.filter(
      Number.isFinite
    );

  if (!usable.length) {
    return 0;
  }

  const mean =
    average(usable);

  return Math.sqrt(
    average(
      usable.map(
        value =>
          (value - mean) ** 2
      )
    )
  );
}

function median(values) {
  const usable =
    values
      .filter(
        Number.isFinite
      )
      .slice()
      .sort(
        (a, b) =>
          a - b
      );

  if (!usable.length) {
    return 50;
  }

  const middle =
    Math.floor(
      usable.length / 2
    );

  if (
    usable.length % 2
  ) {
    return usable[middle];
  }

  return (
    usable[middle - 1] +
    usable[middle]
  ) / 2;
}

function wordCount(value) {
  const clean =
    String(
      value || ''
    ).trim();

  if (!clean) {
    return 0;
  }

  return clean
    .split(/\s+/)
    .filter(Boolean)
    .length;
}

function escapeHTML(value) {
  return String(value).replace(
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

function detectLanguage(value) {
  const latin =
    (
      value.match(
        /[A-Za-z]/g
      ) || []
    ).length;

  const total =
    (
      value.match(
        /\p{L}/gu
      ) || []
    ).length;

  if (!total) {
    return 'Unknown';
  }

  return (
    latin / total > 0.82
  )
    ? 'English'
    : 'Non-English';
}

function countMatches(
  value,
  regex
) {
  return (
    String(value).match(regex) ||
    []
  ).length;
}

function nowISO() {
  return new Date()
    .toISOString();
}

function loadJSON(
  key,
  fallback = []
) {
  try {
    return JSON.parse(
      localStorage.getItem(
        key
      ) ||
      JSON.stringify(
        fallback
      )
    );
  } catch {
    return fallback;
  }
}

function saveJSON(
  key,
  data
) {
  try {
    localStorage.setItem(
      key,
      JSON.stringify(
        data
      )
    );

    return true;

  } catch (error) {
    console.warn(
      `Storage failed: ${key}`,
      error
    );

    return false;
  }
}

/* =========================================================
   UI
========================================================= */

function updateCount() {
  if ($('count')) {
    $('count').textContent =
      `${wordCount(
        textEl?.value || ''
      )} words`;
  }
}

function setProgress(
  percent,
  label
) {
  $('progress')
    ?.classList
    .remove('hidden');

  if ($('bar')) {
    $('bar').style.width =
      `${clamp(percent)}%`;
  }

  if ($('progressText')) {
    $('progressText').textContent =
      label;
  }
}

function hideProgress() {
  setTimeout(
    () =>
      $('progress')
        ?.classList
        .add('hidden'),
    450
  );
}

function setState(label) {
  if ($('modelState')) {
    $('modelState').textContent =
      label;
  }
}

function loadDemo() {
  if (!textEl) {
    return;
  }

  textEl.value = `Artificial intelligence is transforming modern society by changing how people communicate, work, learn, and make decisions. Recent advances in machine learning have allowed software systems to generate text, analyze images, summarize documents, write computer code, and assist with complex research tasks.

One major advantage of artificial intelligence is its ability to process large amounts of information quickly. Businesses can automate repetitive workflows, researchers can examine large datasets, and individuals can use intelligent tools to improve productivity. These systems can identify patterns that might be difficult for humans to notice manually.

At the same time, artificial intelligence introduces important challenges. Generated content may contain incorrect information, fabricated details, or biased conclusions. As AI-generated text becomes more natural, it can also become increasingly difficult to determine whether a document was written by a person or produced by a machine.

For this reason, future digital platforms may require stronger authenticity and provenance systems. Rather than relying on a single detection score, trustworthy tools should examine multiple signals, communicate uncertainty, and avoid presenting probabilistic evidence as absolute proof.`;

  updateCount();
}

textEl?.addEventListener(
  'input',
  updateCount
);

$('clear')?.addEventListener(
  'click',
  () => {
    if (textEl) {
      textEl.value = '';
    }

    updateCount();

    $('report')
      ?.classList
      .add('hidden');
  }
);

$('demo')?.addEventListener(
  'click',
  loadDemo
);

$('scan')?.addEventListener(
  'click',
  runSmartScan
);

/* =========================================================
   DOCUMENT PROFILE
========================================================= */

function createProfile(value) {
  const words =
    value
      .trim()
      .split(/\s+/)
      .filter(Boolean);

  const sentences =
    value
      .split(/[.!?]+/)
      .map(
        sentence =>
          sentence.trim()
      )
      .filter(Boolean);

  const paragraphs =
    value
      .split(/\n\s*\n/)
      .map(
        paragraph =>
          paragraph.trim()
      )
      .filter(Boolean);

  const lines =
    value
      .split(/\n/)
      .map(
        line =>
          line.trim()
      )
      .filter(Boolean);

  const sentenceLengths =
    sentences.map(
      sentence =>
        wordCount(sentence)
    );

  const paragraphLengths =
    paragraphs.map(
      paragraph =>
        wordCount(paragraph)
    );

  const cleanedWords =
    words
      .map(
        word =>
          word
            .toLowerCase()
            .replace(
              /[^\p{L}\p{N}']/gu,
              ''
            )
      )
      .filter(Boolean);

  const avgSentence =
    average(
      sentenceLengths
    );

  const sentenceSD =
    standardDeviation(
      sentenceLengths
    );

  const punctuationPatterns = [
    /,/g,
    /;/g,
    /:/g,
    /[—–-]/g,
    /["“”‘’']/g,
    /[()]/g
  ];

  const punctuationTypes =
    punctuationPatterns
      .filter(
        regex =>
          (
            value.match(regex) ||
            []
          ).length > 0
      )
      .length;

  const quoteCount =
    countMatches(
      value,
      /["“”‘’]/g
    );

  const semicolonCount =
    countMatches(
      value,
      /;/g
    );

  const dialogueLines =
    lines.filter(
      line =>
        /^[“"'—-]/.test(line) ||
        /[”"']$/.test(line)
    ).length;

  const firstPerson =
    countMatches(
      value,
      /\b(I|me|my|mine|we|us|our|ours)\b/gi
    );

  const contractions =
    countMatches(
      value,
      /\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s)\b/gi
    );

  const subjectiveMarkers =
    countMatches(
      value,
      /\b(I think|I believe|I suppose|I feel|in my view|perhaps|maybe|it seems to me|I do not know)\b/gi
    );

  const transitions =
    countMatches(
      value,
      /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|in conclusion|additionally|nevertheless|as a result|on the other hand)\b/gi
    );

  const titleReferences =
    countMatches(
      value,
      /\b(Mr|Mrs|Miss|Ms|Sir|Lady|Lord|Dr)\.?\s+[A-Z][a-z]+/g
    );

  const narrativeMarkers =
    countMatches(
      value,
      /\b(she|he|her|his|hers|him|father|mother|daughter|son|sister|brother|friend|marriage|family|house|young|years|woman|man)\b/gi
    );

  const literaryVocabulary =
    countMatches(
      value,
      /\b(affection|disposition|esteem|governess|intimacy|temper|authority|misfortune|existence|indulgent|remembrance|caresses|enjoyments|judgment|vex|distress|attached|acquaintance|gentleman|lady)\b/gi
    );

  return {
    words:
      words.length,

    sentences:
      sentences.length,

    paragraphs:
      paragraphs.length,

    lineBreaks:
      Math.max(
        0,
        lines.length - 1
      ),

    averageLineLength:
      lines.length
        ? value.length /
          lines.length
        : value.length,

    averageSentenceLength:
      avgSentence,

    sentenceDeviation:
      sentenceSD,

    sentenceBurstiness:
      avgSentence
        ? sentenceSD /
          avgSentence
        : 0,

    paragraphDeviation:
      standardDeviation(
        paragraphLengths
      ),

    lexicalDiversity:
      new Set(
        cleanedWords
      ).size /
      Math.max(
        1,
        cleanedWords.length
      ),

    punctuationTypes,

    quoteCount,

    semicolonCount,

    dialogueLines,

    firstPerson,

    contractions,

    subjectiveMarkers,

    transitions,

    titleReferences,

    narrativeMarkers,

    literaryVocabulary
  };
}
/* =========================================================
   DOMAIN ESTIMATION
========================================================= */

function estimateDomain(
  value,
  profile
) {
  const content =
    value.toLowerCase();

  const signals = {
    abstracts:
      countMatches(
        content,
        /\b(method|methods|results|conclusion|study|participants|dataset|experiment|analysis|significant|hypothesis|abstract|research|findings)\b/g
      ),

    recipes:
      countMatches(
        content,
        /\b(cup|tablespoon|teaspoon|ingredients|preheat|oven|bake|stir|chop|minutes|serve|recipe)\b/g
      ),

    reviews:
      countMatches(
        content,
        /\b(review|rating|stars|recommend|purchase|product|quality|price|experience|bought)\b/g
      ),

    reddit:
      countMatches(
        content,
        /\b(aita|tldr|subreddit|upvote|downvote|edit:|throwaway|imo|lol|op)\b/g
      ),

    wiki:
      countMatches(
        content,
        /\b(was born|is a|refers to|located in|population|history of|known for|founded|species|established)\b/g
      ),

    news:
      countMatches(
        content,
        /\b(reuters|reported|according to|officials|government|minister|president|said on|announced|agency|spokesperson)\b/g
      ),

    books:
      (
        profile.quoteCount >= 4
          ? 3
          : 0
      ) +
      (
        profile.dialogueLines >= 1
          ? 3
          : 0
      ) +
      (
        profile.titleReferences >= 2
          ? 3
          : 0
      ) +
      (
        profile.narrativeMarkers >= 8
          ? 3
          : 0
      ) +
      (
        profile.literaryVocabulary >= 3
          ? 4
          : 0
      ),

    poetry:
      (
        profile.lineBreaks >= 6 &&
        profile.averageLineLength < 60
      )
        ? 6
        : 0
  };

  const sorted =
    Object.entries(signals)
      .sort(
        (a, b) =>
          b[1] - a[1]
      );

  const [
    domain,
    score
  ] = sorted[0];

  if (
    !score ||
    score < 2
  ) {
    return {
      domain:
        'general',

      confidence:
        'low',

      score:
        score || 0,

      signals
    };
  }

  return {
    domain,

    confidence:
      score >= 8
        ? 'high'
        : score >= 4
          ? 'medium'
          : 'low',

    score,

    signals
  };
}


/* =========================================================
   HUMAN COUNTER-EVIDENCE
========================================================= */

function calculateHumanEvidence(
  profile,
  domain
) {
  let score = 0;

  const reasons = [];

  if (
    profile.sentenceBurstiness >= 0.65
  ) {
    score += 20;

    reasons.push(
      'High sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >= 0.45
  ) {
    score += 12;

    reasons.push(
      'Moderate sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >= 0.30
  ) {
    score += 5;
  }

  if (
    profile.punctuationTypes >= 5
  ) {
    score += 12;

    reasons.push(
      'Rich punctuation variety'
    );

  } else if (
    profile.punctuationTypes >= 3
  ) {
    score += 6;
  }

  if (
    profile.quoteCount >= 8 ||
    profile.dialogueLines >= 2
  ) {
    score += 15;

    reasons.push(
      'Dialogue or quotation structure'
    );

  } else if (
    profile.quoteCount >= 3
  ) {
    score += 7;
  }

  if (
    profile.firstPerson >= 4 ||
    profile.subjectiveMarkers >= 2
  ) {
    score += 10;

    reasons.push(
      'Personal or subjective voice'
    );

  } else if (
    profile.firstPerson > 0 ||
    profile.subjectiveMarkers > 0
  ) {
    score += 4;
  }

  if (
    profile.contractions >= 4
  ) {
    score += 6;

    reasons.push(
      'Natural contraction usage'
    );
  }

  if (
    profile.paragraphDeviation >= 25 &&
    profile.paragraphs >= 3
  ) {
    score += 8;

    reasons.push(
      'Irregular paragraph rhythm'
    );

  } else if (
    profile.paragraphDeviation >= 12 &&
    profile.paragraphs >= 2
  ) {
    score += 4;
  }

  if (
    profile.lexicalDiversity >= 0.62
  ) {
    score += 7;

    reasons.push(
      'High lexical diversity'
    );

  } else if (
    profile.lexicalDiversity >= 0.50
  ) {
    score += 3;
  }

  if (
    profile.semicolonCount >= 2
  ) {
    score += 5;

    reasons.push(
      'Complex punctuation rhythm'
    );
  }

  if (
    profile.titleReferences >= 2
  ) {
    score += 7;

    reasons.push(
      'Character-reference structure'
    );
  }

  if (
    profile.literaryVocabulary >= 3
  ) {
    score += 8;

    reasons.push(
      'Literary vocabulary pattern'
    );
  }

  if (
    profile.transitions >= 4
  ) {
    score -= 7;
  }

  if (
    domain === 'books'
  ) {
    score += 10;

    reasons.push(
      'Literary-domain caution'
    );
  }

  if (
    domain === 'poetry'
  ) {
    score += 12;

    reasons.push(
      'Poetry-domain caution'
    );
  }

  return {
    score:
      clamp(
        Math.round(score)
      ),

    reasons
  };
}


/* =========================================================
   CHUNKING
========================================================= */

function chunkText(
  value,
  maxChars = 1300
) {
  const sentences =
    value.match(
      /[^.!?]+[.!?]+|[^.!?]+$/g
    ) || [value];

  const chunks = [];

  let current = '';

  for (
    const sentence
    of sentences
  ) {
    if (
      (
        current +
        sentence
      ).length > maxChars &&
      current
    ) {
      chunks.push(
        current.trim()
      );

      current =
        sentence;

    } else {
      current +=
        sentence;
    }
  }

  if (
    current.trim()
  ) {
    chunks.push(
      current.trim()
    );
  }

  return chunks
    .filter(Boolean)
    .slice(
      0,
      8
    );
}


/* =========================================================
   MODEL LOADERS
========================================================= */

async function loadTMR() {
  if (tmrModel) {
    return tmrModel;
  }

  setState(
    'Loading TMR detector…'
  );

  setProgress(
    8,
    'Loading Model A…'
  );

  tmrModel =
    await pipeline(
      'text-classification',
      MODEL_TMR,
      {
        dtype:
          'q4f16'
      }
    );

  return tmrModel;
}


async function loadE5() {
  if (e5Model) {
    return e5Model;
  }

  setState(
    'Loading E5 detector…'
  );

  setProgress(
    18,
    'Loading Model B…'
  );

  e5Model =
    await pipeline(
      'text-classification',
      MODEL_E5,
      {
        dtype:
          'q4f16'
      }
    );

  return e5Model;
}


async function loadModern() {
  if (modernModel) {
    return modernModel;
  }

  setState(
    'Loading ModernBERT judge…'
  );

  setProgress(
    72,
    'Loading Model C…'
  );

  modernModel =
    await pipeline(
      'text-classification',
      MODEL_MODERN,
      {
        dtype:
          'q4f16'
      }
    );

  return modernModel;
}


/* =========================================================
   MODEL OUTPUT NORMALIZATION
========================================================= */

function aiProbability(
  output
) {
  const results =
    (
      Array.isArray(output)
        ? output
        : [output]
    ).flat();

  let ai = null;
  let human = null;

  for (
    const item
    of results
  ) {
    const label =
      String(
        item?.label || ''
      )
        .toLowerCase()
        .trim();

    const score =
      Number(
        item?.score
      );

    if (
      !Number.isFinite(score)
    ) {
      continue;
    }

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

  if (
    ai !== null
  ) {
    return clamp(
      ai,
      0,
      1
    );
  }

  if (
    human !== null
  ) {
    return clamp(
      1 - human,
      0,
      1
    );
  }

  if (
    results.length === 1 &&
    Number.isFinite(
      Number(
        results[0]?.score
      )
    )
  ) {
    return clamp(
      Number(
        results[0].score
      ),
      0,
      1
    );
  }

  if (
    results.length >= 2
  ) {
    const second =
      Number(
        results[1]?.score
      );

    if (
      Number.isFinite(second)
    ) {
      return clamp(
        second,
        0,
        1
      );
    }
  }

  return 0.5;
}


async function classify(
  model,
  value
) {
  const result =
    await model(
      value,
      {
        top_k:
          null,

        truncation:
          true
      }
    );

  return Math.round(
    aiProbability(
      result
    ) * 100
  );
}


/* =========================================================
   THIRD-MODEL ROUTING
========================================================= */

function shouldUseThirdModel({
  quickScores,
  segmentScores,
  domain,
  humanScore,
  words
}) {
  if (
    !Number.isFinite(
      quickScores.tmr
    ) ||
    !Number.isFinite(
      quickScores.e5
    )
  ) {
    return false;
  }

  const modelGap =
    Math.abs(
      quickScores.tmr -
      quickScores.e5
    );

  const raw =
    median([
      quickScores.tmr,
      quickScores.e5
    ]);

  const segmentRange =
    segmentScores.length
      ? Math.max(
          ...segmentScores
        ) -
        Math.min(
          ...segmentScores
        )
      : 0;

  const segmentSD =
    standardDeviation(
      segmentScores
    );

  return (
    words < 180 ||

    modelGap >= 18 ||

    (
      raw >= 35 &&
      raw <= 85
    ) ||

    segmentRange >= 45 ||

    segmentSD >= 20 ||

    humanScore >= 45 ||

    domain === 'books' ||

    domain === 'poetry'
  );
}


/* =========================================================
   MODEL OUTLIER DEFENSE
========================================================= */

function detectModelOutlier(
  scores
) {
  const entries =
    Object.entries({
      tmr:
        scores.tmr,

      e5:
        scores.e5,

      modern:
        scores.modern
    })
      .filter(
        ([, value]) =>
          Number.isFinite(value)
      );

  if (
    entries.length < 3
  ) {
    return {
      detected:
        false,

      detector:
        null,

      pairGap:
        null,

      outlierDistance:
        null,

      reason:
        'Three active models are required for outlier detection.'
    };
  }

  const candidates = [];

  for (
    let i = 0;
    i < entries.length;
    i++
  ) {
    const [
      name,
      value
    ] = entries[i];

    /*
      IMPORTANT:
      The previous broken version used an invalid
      empty function parameter here.

      This is the corrected syntax.
    */
    const others =
      entries.filter(
        (
          item,
          index
        ) =>
          index !== i
      );

    const pairGap =
      Math.abs(
        others[0][1] -
        others[1][1]
      );

    const pairMean =
      average(
        others.map(
          item =>
            item[1]
        )
      );

    const outlierDistance =
      Math.abs(
        value -
        pairMean
      );

    candidates.push({
      detector:
        name,

      value,

      pairGap,

      outlierDistance,

      pair:
        others.map(
          item =>
            item[0]
        ),

      pairValues:
        others.map(
          item =>
            item[1]
        )
    });
  }

  candidates.sort(
    (a, b) =>
      b.outlierDistance -
      a.outlierDistance
  );

  const best =
    candidates[0];

  const detected =
    best.pairGap <= 18 &&
    best.outlierDistance >= 32;

  return {
    detected,

    detector:
      detected
        ? best.detector
        : null,

    value:
      best.value,

    pair:
      best.pair,

    pairValues:
      best.pairValues,

    pairGap:
      Math.round(
        best.pairGap
      ),

    outlierDistance:
      Math.round(
        best.outlierDistance
      ),

    reason:
      detected
        ? `${best.detector} disagrees strongly while the other two detectors agree.`
        : 'No clear single-model outlier detected.'
  };
}


/* =========================================================
   RELIABILITY WEIGHTS
========================================================= */

function modelReliabilityWeights({
  scores,
  outlier,
  domain,
  language
}) {
  const weights = {
    tmr:
      Number.isFinite(
        scores.tmr
      )
        ? 1
        : 0,

    e5:
      Number.isFinite(
        scores.e5
      )
        ? 1
        : 0,

    modern:
      Number.isFinite(
        scores.modern
      )
        ? 1
        : 0
  };

  if (
    outlier.detected &&
    outlier.detector &&
    weights[
      outlier.detector
    ] > 0
  ) {
    weights[
      outlier.detector
    ] *= 0.22;
  }

  if (
    domain === 'books' ||
    domain === 'poetry'
  ) {
    if (
      weights.tmr
    ) {
      weights.tmr *= 0.82;
    }

    if (
      weights.e5
    ) {
      weights.e5 *= 0.82;
    }
  }

  if (
    language !== 'English'
  ) {
    for (
      const key
      of Object.keys(
        weights
      )
    ) {
      weights[key] *= 0.65;
    }
  }

  return weights;
}


/* =========================================================
   EVIDENCE SUFFICIENCY GATE
========================================================= */

function calculateEvidenceSufficiency({
  scores,
  profile,
  language,
  domain,
  human,
  segmentScores,
  outlier,
  thirdUsed
}) {
  const activeScores =
    [
      scores.tmr,
      scores.e5,
      scores.modern
    ]
      .filter(
        Number.isFinite
      );

  const activeModels =
    activeScores.length;

  const modelSpread =
    activeModels >= 2
      ? Math.max(
          ...activeScores
        ) -
        Math.min(
          ...activeScores
        )
      : 100;

  const segmentRange =
    segmentScores.length
      ? Math.max(
          ...segmentScores
        ) -
        Math.min(
          ...segmentScores
        )
      : 100;

  const segmentSD =
    segmentScores.length
      ? standardDeviation(
          segmentScores
        )
      : 50;

  let score = 100;

  const reasons = [];

  if (
    activeModels < 2
  ) {
    score -= 55;

    reasons.push(
      'Fewer than two detectors produced usable results'
    );
  }

  if (
    activeModels === 2
  ) {
    score -= 8;
  }

  if (
    profile.words < 100
  ) {
    score -= 25;

    reasons.push(
      'Text is short'
    );

  } else if (
    profile.words < 150
  ) {
    score -= 15;

    reasons.push(
      'Limited text length'
    );

  } else if (
    profile.words < 250
  ) {
    score -= 6;
  }

  if (
    language !== 'English'
  ) {
    score -= 35;

    reasons.push(
      'Current detector path is optimized primarily for English'
    );
  }

  if (
    modelSpread >= 50
  ) {
    score -= 30;

    reasons.push(
      'Very high detector disagreement'
    );

  } else if (
    modelSpread >= 32
  ) {
    score -= 20;

    reasons.push(
      'High detector disagreement'
    );

  } else if (
    modelSpread >= 20
  ) {
    score -= 9;
  }

  if (
    segmentRange >= 75
  ) {
    score -= 24;

    reasons.push(
      'Very unstable segment signals'
    );

  } else if (
    segmentRange >= 55
  ) {
    score -= 14;

    reasons.push(
      'Large variation across text segments'
    );

  } else if (
    segmentRange >= 35
  ) {
    score -= 6;
  }

  if (
    segmentSD >= 28
  ) {
    score -= 14;

    reasons.push(
      'High segment deviation'
    );

  } else if (
    segmentSD >= 20
  ) {
    score -= 7;
  }

  if (
    domain === 'books' ||
    domain === 'poetry'
  ) {
    score -= 12;

    reasons.push(
      'Literary text requires additional caution'
    );
  }

  if (
    human.score >= 60
  ) {
    score -= 8;

    reasons.push(
      'Strong human-style counter-evidence conflicts with detector evidence'
    );
  }

  if (
    outlier.detected
  ) {
    score -= 6;

    reasons.push(
      `Possible ${outlier.detector} detector outlier`
    );
  }

  if (
    !thirdUsed &&
    !isMobileDevice() &&
    activeModels === 2 &&
    modelSpread >= 18
  ) {
    score -= 7;

    reasons.push(
      'Third-model verification was not available'
    );
  }

  score =
    clamp(
      Math.round(score)
    );

  let level =
    'INSUFFICIENT';

  if (
    score >= 75
  ) {
    level =
      'STRONG';

  } else if (
    score >= 58
  ) {
    level =
      'MODERATE';

  } else if (
    score >= 42
  ) {
    level =
      'LIMITED';
  }

  return {
    score,

    level,

    sufficientForStrongVerdict:
      score >= 75,

    sufficientForDirectionalVerdict:
      score >= 58,

    activeModels,

    modelSpread:
      Math.round(
        modelSpread
      ),

    segmentRange:
      Math.round(
        segmentRange
      ),

    segmentSD:
      Math.round(
        segmentSD
      ),

    reasons
  };
}


/* =========================================================
   CONSENSUS ENGINE
========================================================= */

function buildConsensus({
  scores,
  segmentScores,
  profile,
  language,
  domain,
  human,
  thirdUsed
}) {
  const outlier =
    detectModelOutlier(
      scores
    );

  const weights =
    modelReliabilityWeights({
      scores,
      outlier,
      domain,
      language
    });

  const values = [
    scores.tmr,
    scores.e5,
    scores.modern
  ];

  const weightValues = [
    weights.tmr,
    weights.e5,
    weights.modern
  ];

  const activeScores =
    values.filter(
      Number.isFinite
    );

  const raw =
    Math.round(
      weightedAverage(
        values,
        weightValues
      )
    );

  const diagnosticMedian =
    Math.round(
      median(
        activeScores
      )
    );

  const evidenceSufficiency =
    calculateEvidenceSufficiency({
      scores,
      profile,
      language,
      domain,
      human,
      segmentScores,
      outlier,
      thirdUsed
    });

  const modelSpread =
    evidenceSufficiency
      .modelSpread;

  const segmentRange =
    evidenceSufficiency
      .segmentRange;

  const segmentSD =
    evidenceSufficiency
      .segmentSD;

  const instability =
    1 -
    evidenceSufficiency.score /
    100;

  const humanPenalty =
    human.score *
    (
      0.08 +
      instability * 0.48
    ) *
    (
      raw / 100
    );

  let calibrated =
    clamp(
      Math.round(
        raw -
        humanPenalty
      )
    );

  /*
    Literary protection:
    avoid extreme AI claims when human-style
    literary evidence is substantial.
  */
  if (
    (
      domain === 'books' ||
      domain === 'poetry'
    ) &&
    human.score >= 45
  ) {
    calibrated =
      Math.min(
        calibrated,
        68
      );
  }

  if (
    outlier.detected &&
    human.score >= 50 &&
    calibrated >= 75
  ) {
    calibrated =
      Math.min(
        calibrated,
        72
      );
  }

  let quality =
    evidenceSufficiency.score;

  if (
    outlier.detected
  ) {
    quality =
      Math.max(
        0,
        quality - 4
      );
  }

  quality =
    clamp(
      Math.round(
        quality
      )
    );

  let uncertainty =
    100 - quality;

  if (
    raw >= 70 &&
    human.score >= 55
  ) {
    uncertainty +=
      Math.round(
        (
          human.score - 50
        ) * 0.35
      );
  }

  if (
    calibrated >= 40 &&
    calibrated <= 65
  ) {
    uncertainty += 5;
  }

  uncertainty =
    clamp(
      Math.round(
        uncertainty
      ),
      5,
      95
    );

  const confidence =
    100 -
    uncertainty;

  const highConflict =
    modelSpread >= 32 ||
    segmentRange >= 70 ||
    segmentSD >= 28;

  let verdict =
    'INCONCLUSIVE';

  if (
    language !== 'English'
  ) {
    verdict =
      'INCONCLUSIVE';

  } else if (
    evidenceSufficiency
      .sufficientForStrongVerdict &&
    calibrated >= 86 &&
    !highConflict &&
    human.score < 48
  ) {
    verdict =
      'Strong AI evidence';

  } else if (
    evidenceSufficiency
      .sufficientForDirectionalVerdict &&
    calibrated >= 73 &&
    modelSpread < 30 &&
    segmentRange < 65 &&
    human.score < 55
  ) {
    verdict =
      'Likely AI';

  } else if (
    evidenceSufficiency
      .sufficientForStrongVerdict &&
    calibrated <= 18 &&
    human.score >= 55 &&
    !highConflict
  ) {
    verdict =
      'Strong human evidence';

  } else if (
    evidenceSufficiency
      .sufficientForDirectionalVerdict &&
    calibrated <= 36 &&
    human.score >= 42 &&
    modelSpread < 32
  ) {
    verdict =
      'Likely human';
  }

  /*
    Hard abstention guards.
  */
  if (
    evidenceSufficiency.score < 58
  ) {
    verdict =
      'INCONCLUSIVE';
  }

  if (
    highConflict &&
    (
      verdict ===
        'Strong AI evidence' ||
      verdict ===
        'Strong human evidence'
    )
  ) {
    verdict =
      'INCONCLUSIVE';
  }

  if (
    (
      domain === 'books' ||
      domain === 'poetry'
    ) &&
    human.score >= 50 &&
    verdict.includes('AI')
  ) {
    verdict =
      'INCONCLUSIVE';
  }

  if (
    outlier.detected &&
    modelSpread >= 45 &&
    verdict.includes('AI')
  ) {
    verdict =
      'INCONCLUSIVE';
  }

  return {
    raw,

    diagnosticMedian,

    calibrated,

    quality,

    uncertainty,

    confidence,

    verdict,

    modelSpread,

    modelSD:
      Math.round(
        standardDeviation(
          activeScores
        )
      ),

    segmentRange,

    segmentSD,

    activeModels:
      activeScores.length,

    thirdUsed,

    humanPenalty:
      Math.round(
        humanPenalty
      ),

    outlier,

    weights,

    evidenceSufficiency,

    highConflict
  };
}
/* =========================================================
   MAIN SMART SCAN
========================================================= */

async function runSmartScan() {
  const value =
    textEl?.value
      ?.trim() ||
    '';

  const words =
    wordCount(
      value
    );

  if (
    words < 80
  ) {
    alert(
      'Paste at least 80 words. 150+ words is recommended.'
    );

    return;
  }

  if (
    !$('scan')
  ) {
    return;
  }

  $('scan').disabled =
    true;

  try {
    setProgress(
      3,
      'Profiling document…'
    );

    setState(
      'V6.1 Smart Scan running…'
    );

    const language =
      detectLanguage(
        value
      );

    const profile =
      createProfile(
        value
      );

    const domainInfo =
      estimateDomain(
        value,
        profile
      );

    const human =
      calculateHumanEvidence(
        profile,
        domainInfo.domain
      );

    const chunks =
      chunkText(
        value
      );

    const scores = {
      tmr:
        NaN,

      e5:
        NaN,

      modern:
        NaN
    };

    let segmentScores =
      [];

    let thirdUsed =
      false;

    const mobile =
      isMobileDevice();


    /* =====================================================
       MODEL A — TMR
    ===================================================== */

    try {
      const modelA =
        await loadTMR();

      setProgress(
        22,
        'Running Model A — TMR…'
      );

      scores.tmr =
        await classify(
          modelA,
          value
        );

      for (
        let i = 0;
        i < chunks.length;
        i++
      ) {
        setProgress(
          28 +
          Math.round(
            (
              i /
              Math.max(
                1,
                chunks.length
              )
            ) * 27
          ),

          `Trace Map ${i + 1}/${chunks.length}…`
        );

        const segmentScore =
          await classify(
            modelA,
            chunks[i]
          );

        segmentScores.push(
          segmentScore
        );
      }

    } catch (error) {
      console.error(
        'TMR failed:',
        error
      );
    }


    /* =====================================================
       MODEL B — E5
    ===================================================== */

    try {
      const modelB =
        await loadE5();

      setProgress(
        60,
        'Running Model B — E5-small…'
      );

      scores.e5 =
        await classify(
          modelB,
          value
        );

    } catch (error) {
      console.error(
        'E5 failed:',
        error
      );
    }


    /* =====================================================
       MODEL C ROUTING
    ===================================================== */

    if (
      !mobile &&
      Number.isFinite(
        scores.tmr
      ) &&
      Number.isFinite(
        scores.e5
      )
    ) {
      thirdUsed =
        shouldUseThirdModel({
          quickScores:
            scores,

          segmentScores,

          domain:
            domainInfo.domain,

          humanScore:
            human.score,

          words
        });
    }


    /* =====================================================
       MODEL C — MODERNBERT
    ===================================================== */

    if (
      thirdUsed
    ) {
      try {
        const modelC =
          await loadModern();

        setProgress(
          76,
          'Running Model C — ModernBERT…'
        );

        scores.modern =
          await classify(
            modelC,
            value
          );

      } catch (error) {
        console.error(
          'ModernBERT failed:',
          error
        );

        thirdUsed =
          false;
      }
    }


    /*
      Safe fallback for segment rendering.
    */

    if (
      !segmentScores.length
    ) {
      segmentScores =
        chunks.map(
          () => 50
        );
    }


    setProgress(
      88,
      'Checking evidence sufficiency…'
    );

    await new Promise(
      resolve =>
        requestAnimationFrame(
          resolve
        )
    );


    setProgress(
      92,
      'Calibrating evidence…'
    );


    const consensus =
      buildConsensus({
        scores,

        segmentScores,

        profile,

        language,

        domain:
          domainInfo.domain,

        human,

        thirdUsed
      });


    const scan = {
      version:
        VERSION,

      timestamp:
        nowISO(),

      words,

      language,

      domain:
        domainInfo.domain,

      domainConfidence:
        domainInfo.confidence,

      domainScore:
        domainInfo.score,

      profile,

      scores,

      segmentScores,

      human,

      consensus
    };


    setProgress(
      96,
      'Building evidence report…'
    );


    render(
      scan
    );


    saveHistory(
      scan
    );


    setProgress(
      100,
      'Trace complete'
    );


    if (
      consensus.activeModels < 2
    ) {
      setState(
        'V6.1 Limited Evidence Mode'
      );

    } else if (
      consensus.outlier
        ?.detected
    ) {
      setState(
        `V6.1 Outlier Defense ✓ (${consensus.outlier.detector})`
      );

    } else if (
      thirdUsed
    ) {
      setState(
        'V6.1 3-Model Consensus ✓'
      );

    } else if (
      mobile
    ) {
      setState(
        'V6.1 Mobile Safe Consensus ✓'
      );

    } else {
      setState(
        'V6.1 Smart Consensus ✓'
      );
    }


    setTimeout(
      () => {
        try {
          benchmarkPrompt(
            scan
          );

        } catch (error) {
          console.warn(
            'Benchmark prompt error:',
            error
          );
        }
      },
      700
    );

  } catch (fatalError) {
    console.error(
      'AI TRACE V6.1 fatal error:',
      fatalError
    );

    setState(
      'Scan error'
    );

    alert(
`AI Trace could not complete the scan.

${fatalError?.message || 'Unknown error'}

Open the browser console for details.`
    );

  } finally {
    if ($('scan')) {
      $('scan').disabled =
        false;
    }

    hideProgress();
  }
}


/* =========================================================
   RENDER MAIN REPORT
========================================================= */

function render(scan) {
  const {
    consensus,
    scores,
    human,
    language,
    domain,
    domainConfidence,
    segmentScores,
    profile
  } = scan;


  $('report')
    ?.classList
    .remove('hidden');


  const resolvedDecision =
    consensus.verdict !==
      'INCONCLUSIVE';


  /*
    V6.1:
    A detector signal is NOT presented as a probability
    when evidence is insufficient.
  */

  if ($('score')) {
    $('score').textContent =
      resolvedDecision
        ? `${consensus.calibrated}%`
        : '—';
  }


  if ($('scaleFill')) {
    $('scaleFill').style.width =
      resolvedDecision
        ? `${consensus.calibrated}%`
        : '0%';
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
      `Evidence confidence: ${confidenceLabel} (${consensus.confidence}%)`;
  }


  if ($('explain')) {
    let explanation;

    if (
      consensus.verdict ===
      'INCONCLUSIVE'
    ) {
      explanation =
        `AI Trace does not have enough independent evidence for a reliable AI/Human verdict. Diagnostic detector signal: ${consensus.calibrated}%. Evidence sufficiency: ${consensus.evidenceSufficiency.score}% (${consensus.evidenceSufficiency.level}). Active models: ${consensus.activeModels}/3.`;

    } else {
      explanation =
        `The result combines multiple detector signals, evidence sufficiency, human counter-evidence, domain context and segment stability. Diagnostic AI signal: ${consensus.calibrated}%. Evidence quality: ${consensus.quality}%.`;
    }

    if (
      consensus.outlier
        ?.detected
    ) {
      explanation +=
        ` ${String(
          consensus.outlier.detector
        ).toUpperCase()} was identified as a possible detector outlier.`;
    }

    $('explain').textContent =
      explanation;
  }


  /* =======================================================
     TRACE DNA
  ======================================================= */

  const humanDisplay =
    clamp(
      Math.round(
        human.score * 0.72 +
        (
          100 -
          consensus.calibrated
        ) * 0.28
      )
    );


  if ($('humanVal')) {
    $('humanVal').textContent =
      `${humanDisplay}%`;
  }


  if ($('aiVal')) {
    $('aiVal').textContent =
      resolvedDecision
        ? `${consensus.calibrated}%`
        : 'N/A';
  }


  if ($('uncertainVal')) {
    $('uncertainVal').textContent =
      `${consensus.uncertainty}%`;
  }


  if ($('humanBar')) {
    $('humanBar').style.width =
      `${humanDisplay}%`;
  }


  if ($('aiBar')) {
    $('aiBar').style.width =
      resolvedDecision
        ? `${consensus.calibrated}%`
        : '0%';
  }


  if ($('uncertainBar')) {
    $('uncertainBar').style.width =
      `${consensus.uncertainty}%`;
  }


  if ($('engineBadge')) {
    if (
      consensus.activeModels < 2
    ) {
      $('engineBadge').textContent =
        'V6.1 • LIMITED EVIDENCE';

    } else if (
      consensus.outlier
        ?.detected
    ) {
      $('engineBadge').textContent =
        'V6.1 • OUTLIER DEFENSE';

    } else if (
      consensus.thirdUsed
    ) {
      $('engineBadge').textContent =
        'V6.1 • 3-MODEL CONSENSUS';

    } else if (
      isMobileDevice()
    ) {
      $('engineBadge').textContent =
        'V6.1 • MOBILE SAFE';

    } else {
      $('engineBadge').textContent =
        'V6.1 • SMART CONSENSUS';
    }
  }


  const humanReasons =
    human.reasons.length
      ? human.reasons
          .slice(0, 5)
          .join(' • ')
      : 'No strong human-style counter-signals';


  const modernText =
    Number.isFinite(
      scores.modern
    )
      ? `${scores.modern}% AI signal`
      : isMobileDevice()
        ? 'Disabled on mobile for memory stability'
        : 'Not used / unavailable';


  const outlierText =
    consensus.outlier
      ?.detected
      ? `${String(
          consensus.outlier.detector
        ).toUpperCase()} (${consensus.outlier.value}%) flagged as possible outlier.`
      : 'No clear detector outlier';


  const sufficiency =
    consensus
      .evidenceSufficiency;


  const evidence = [
    [
      'Final decision',
      consensus.verdict,
      'Outcome'
    ],

    [
      'Diagnostic AI signal',
      `${consensus.calibrated}%`,
      resolvedDecision
        ? 'Resolved'
        : 'Not a probability'
    ],

    [
      'Raw weighted detector signal',
      `${consensus.raw}%`,
      'Diagnostic'
    ],

    [
      'Detector median',
      `${consensus.diagnosticMedian}%`,
      'Diagnostic'
    ],

    [
      'Evidence sufficiency',
      `${sufficiency.score}% — ${sufficiency.level}`,
      sufficiency.sufficientForStrongVerdict
        ? 'Strong'
        : sufficiency.sufficientForDirectionalVerdict
          ? 'Usable'
          : 'Insufficient'
    ],

    [
      'Independent models',
      `${consensus.activeModels}/3 active`,
      consensus.activeModels >= 2
        ? 'Sufficient count'
        : 'Insufficient count'
    ],

    [
      'Evidence quality',
      `${consensus.quality}%`,
      consensus.quality >= 75
        ? 'Strong'
        : consensus.quality >= 58
          ? 'Moderate'
          : 'Weak'
    ],

    [
      'TMR detector',
      Number.isFinite(
        scores.tmr
      )
        ? `${scores.tmr}% AI signal`
        : 'Unavailable',
      'Model A'
    ],

    [
      'E5-small detector',
      Number.isFinite(
        scores.e5
      )
        ? `${scores.e5}% AI signal`
        : 'Unavailable',
      'Model B'
    ],

    [
      'ModernBERT judge',
      modernText,
      'Model C'
    ],

    [
      'Human counter-evidence',
      `${human.score}% — ${humanReasons}`,
      human.score >= 55
        ? 'Strong'
        : human.score >= 35
          ? 'Moderate'
          : 'Low'
    ],

    [
      'Model spread',
      `${consensus.modelSpread} points`,
      consensus.modelSpread >= 32
        ? 'High conflict'
        : 'Acceptable'
    ],

    [
      'Segment range',
      `${consensus.segmentRange} points`,
      consensus.segmentRange >= 70
        ? 'High variation'
        : 'Acceptable'
    ],

    [
      'Segment deviation',
      `${consensus.segmentSD}`,
      consensus.segmentSD >= 28
        ? 'High'
        : 'Acceptable'
    ],

    [
      'Outlier analysis',
      outlierText,
      consensus.outlier
        ?.detected
        ? 'Possible outlier'
        : 'Clear'
    ],

    [
      'Domain context',
      `${domain} (${domainConfidence} confidence)`,
      (
        domain === 'books' ||
        domain === 'poetry'
      )
        ? 'Protected domain'
        : 'Context'
    ],

    [
      'Language fit',
      language === 'English'
        ? 'English — strongest supported path'
        : 'Non-English — reduced reliability',
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

    <span>
      ${escapeHTML(
        item[0]
      )}
    </span>

    <span>
      ${escapeHTML(
        item[2]
      )}
    </span>

  </div>

  <small>
    ${escapeHTML(
      item[1]
    )}
  </small>

</div>
`
        )
        .join('');
  }


  /* =======================================================
     METRICS
  ======================================================= */

  const metrics = {
    Words:
      profile.words,

    Sentences:
      profile.sentences,

    Domain:
      domain,

    'Domain confidence':
      domainConfidence,

    Language:
      language,

    'Models active':
      `${consensus.activeModels}/3`,

    'Sentence burstiness':
      profile
        .sentenceBurstiness
        .toFixed(2),

    'Lexical diversity':
      `${Math.round(
        profile.lexicalDiversity *
        100
      )}%`,

    'Human evidence':
      `${human.score}%`,

    'Diagnostic AI signal':
      `${consensus.calibrated}%`,

    'Raw detector signal':
      `${consensus.raw}%`,

    'Evidence sufficiency':
      `${sufficiency.score}%`,

    'Evidence level':
      sufficiency.level,

    'Model spread':
      `${consensus.modelSpread} pts`,

    'Segment deviation':
      consensus.segmentSD,

    'Segment range':
      `${consensus.segmentRange} pts`,

    'Evidence quality':
      `${consensus.quality}%`,

    'Possible outlier':
      consensus.outlier
        ?.detected
        ? consensus.outlier.detector
        : 'None',

    Decision:
      consensus.verdict
  };


  if ($('metrics')) {
    $('metrics').innerHTML =
      Object.entries(
        metrics
      )
        .map(
          ([key, value]) => `
<div class="metric">

  <span>
    ${escapeHTML(key)}
  </span>

  <b>
    ${escapeHTML(
      String(value)
    )}
  </b>

</div>
`
        )
        .join('');
  }


  /* =======================================================
     TRACE MAP
  ======================================================= */

  const chunks =
    chunkText(
      textEl?.value
        ?.trim() ||
      ''
    );


  if ($('segments')) {
    $('segments').innerHTML =
      chunks
        .map(
          (
            chunk,
            index
          ) => {
            const score =
              segmentScores[index] ??
              50;

            return `
<div class="segment">

  <div class="segmentHead">

    <b>
      Segment ${index + 1}
    </b>

    <span>
      ${score}% TMR diagnostic
    </span>

  </div>

  <div class="segmentMeter">

    <i style="width:${clamp(score)}%">
    </i>

  </div>

  <p>
    ${escapeHTML(
      chunk.slice(
        0,
        320
      )
    )}${
      chunk.length > 320
        ? '…'
        : ''
    }
  </p>

</div>
`;
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


/* =========================================================
   BENCHMARK STORAGE + MIGRATION
========================================================= */

function normalizeBenchmarkRecord(
  record
) {
  if (!record) {
    return null;
  }

  const truth =
    String(
      record.truth ||
      record.groundTruth ||
      ''
    )
      .trim()
      .toUpperCase();

  return {
    ...record,

    truth,

    version:
      record.version ||
      record.appVersion ||
      'legacy',

    domain:
      record.domain ||
      'general',

    scores:
      record.scores || {
        tmr:
          Number(
            record.models
              ?.tmr ??
            record.tmr
          ),

        e5:
          Number(
            record.models
              ?.e5 ??
            record.e5
          ),

        modern:
          Number(
            record.models
              ?.modern ??
            record.modern
          )
      },

    consensus:
      record.consensus || {
        calibrated:
          record.calibratedScore ??
          record.score ??
          record.rawSignal ??
          50,

        quality:
          record.evidenceQuality ??
          record.confidence ??
          50,

        verdict:
          record.verdict ||
          'INCONCLUSIVE'
      }
  };
}


function loadBench() {
  const current =
    loadJSON(
      BENCH_KEY,
      []
    );

  if (
    Array.isArray(current) &&
    current.length
  ) {
    return current
      .map(
        normalizeBenchmarkRecord
      )
      .filter(Boolean);
  }

  for (
    const legacyKey
    of LEGACY_BENCH_KEYS
  ) {
    const legacy =
      loadJSON(
        legacyKey,
        []
      );

    if (
      Array.isArray(legacy) &&
      legacy.length
    ) {
      const normalized =
        legacy
          .map(
            normalizeBenchmarkRecord
          )
          .filter(Boolean);

      saveJSON(
        BENCH_KEY,
        normalized
      );

      return normalized;
    }
  }

  return [];
}


function saveBench(
  records
) {
  saveJSON(
    BENCH_KEY,
    records
  );
}


/* =========================================================
   BENCHMARK PREDICTION
========================================================= */

function benchmarkPrediction(
  record
) {
  const verdict =
    record?.consensus
      ?.verdict ||
    record?.verdict ||
    '';

  if (
    verdict ===
      'Strong AI evidence' ||
    verdict ===
      'Likely AI'
  ) {
    return 'AI';
  }

  if (
    verdict ===
      'Strong human evidence' ||
    verdict ===
      'Likely human'
  ) {
    return 'HUMAN';
  }

  return 'ABSTAIN';
}


/* =========================================================
   BENCHMARK METRICS
========================================================= */

function benchmarkMetrics(
  records =
    loadBench()
) {
  const rows =
    records.filter(
      record =>
        record.truth === 'AI' ||
        record.truth === 'HUMAN'
    );

  let TP = 0;
  let TN = 0;
  let FP = 0;
  let FN = 0;

  let aiAbstain = 0;
  let humanAbstain = 0;

  for (
    const record
    of rows
  ) {
    const predicted =
      benchmarkPrediction(
        record
      );

    if (
      predicted ===
      'ABSTAIN'
    ) {
      if (
        record.truth ===
        'AI'
      ) {
        aiAbstain++;
      } else {
        humanAbstain++;
      }

      continue;
    }

    if (
      record.truth === 'AI' &&
      predicted === 'AI'
    ) {
      TP++;
    }

    if (
      record.truth === 'HUMAN' &&
      predicted === 'HUMAN'
    ) {
      TN++;
    }

    if (
      record.truth === 'HUMAN' &&
      predicted === 'AI'
    ) {
      FP++;
    }

    if (
      record.truth === 'AI' &&
      predicted === 'HUMAN'
    ) {
      FN++;
    }
  }

  const totalAI =
    rows.filter(
      record =>
        record.truth === 'AI'
    ).length;

  const totalHuman =
    rows.filter(
      record =>
        record.truth === 'HUMAN'
    ).length;

  const decided =
    TP +
    TN +
    FP +
    FN;

  const total =
    rows.length;

  const abstained =
    aiAbstain +
    humanAbstain;

  const pct =
    (
      numerator,
      denominator
    ) =>
      denominator
        ? Math.round(
            numerator /
            denominator *
            100
          )
        : 0;

  return {
    total,
    totalAI,
    totalHuman,

    TP,
    TN,
    FP,
    FN,

    decided,
    abstained,

    aiAbstain,
    humanAbstain,

    coverage:
      pct(
        decided,
        total
      ),

    selectiveAccuracy:
      pct(
        TP + TN,
        decided
      ),

    precision:
      pct(
        TP,
        TP + FP
      ),

    recall:
      pct(
        TP,
        totalAI
      ),

    specificity:
      pct(
        TN,
        totalHuman
      ),

    fpr:
      pct(
        FP,
        totalHuman
      ),

    fnr:
      pct(
        FN,
        totalAI
      ),

    aiAbstainRate:
      pct(
        aiAbstain,
        totalAI
      ),

    humanAbstainRate:
      pct(
        humanAbstain,
        totalHuman
      )
  };
}


/* =========================================================
   BENCHMARK PROMPT
========================================================= */

function benchmarkPrompt(
  scan
) {
  const answer =
    prompt(
`AI TRACE V6.1 BENCHMARK

Only label samples whose TRUE origin you KNOW.

AI      = definitely AI-generated
HUMAN   = definitely human-written
MIXED   = known human + AI mixture
UNKNOWN = origin not known

Cancel / leave empty to skip.`
    );

  if (!answer) {
    return;
  }

  const truth =
    answer
      .trim()
      .toUpperCase();

  if (
    ![
      'AI',
      'HUMAN',
      'MIXED',
      'UNKNOWN'
    ].includes(
      truth
    )
  ) {
    alert(
      'Benchmark skipped. Use AI, HUMAN, MIXED or UNKNOWN.'
    );

    return;
  }

  const source =
    prompt(
      'Source / note:',
      ''
    ) || '';

  const records =
    loadBench();

  const prefixMap = {
    AI:
      'A',

    HUMAN:
      'H',

    MIXED:
      'M',

    UNKNOWN:
      'U'
  };

  const count =
    records.filter(
      record =>
        record.truth === truth
    ).length + 1;

  const id =
    `${prefixMap[truth]}-${String(
      count
    ).padStart(
      3,
      '0'
    )}`;

  /*
    Ground truth is stored only AFTER
    the prediction has already happened.
  */

  records.push({
    id,

    truth,

    source,

    savedAt:
      nowISO(),

    predictionFrozen:
      true,

    ...scan
  });

  saveBench(
    records
  );

  renderBenchmarkPanel();

  const metrics =
    benchmarkMetrics(
      records
    );

  alert(
`Benchmark saved: ${id}

AI: ${metrics.totalAI}
HUMAN: ${metrics.totalHuman}

Coverage: ${metrics.coverage}%
Selective accuracy: ${metrics.selectiveAccuracy}%

False-positive rate: ${metrics.fpr}%
False-negative rate: ${metrics.fnr}%

AI abstention rate: ${metrics.aiAbstainRate}%
Human abstention rate: ${metrics.humanAbstainRate}%

Development data only.`
  );
}


/* =========================================================
   BENCHMARK PANEL
========================================================= */

function renderBenchmarkPanel() {
  const report =
    $('report');

  if (!report) {
    return;
  }

  let panel =
    $('benchmarkPanelV61');

  if (!panel) {
    panel =
      document.createElement(
        'section'
      );

    panel.id =
      'benchmarkPanelV61';

    panel.className =
      'panel devPanel';

    panel.style.marginTop =
      '18px';

    report.appendChild(
      panel
    );
  }

  const records =
    loadBench();

  const metrics =
    benchmarkMetrics(
      records
    );

  const binary =
    records.filter(
      record =>
        record.truth === 'AI' ||
        record.truth === 'HUMAN'
    );

  const mixed =
    records.filter(
      record =>
        record.truth === 'MIXED'
    );

  const falsePositives =
    binary.filter(
      record =>
        record.truth === 'HUMAN' &&
        benchmarkPrediction(
          record
        ) === 'AI'
    );

  const falseNegatives =
    binary.filter(
      record =>
        record.truth === 'AI' &&
        benchmarkPrediction(
          record
        ) === 'HUMAN'
    );

  const abstentions =
    binary.filter(
      record =>
        benchmarkPrediction(
          record
        ) === 'ABSTAIN'
    );

  const recordCard =
    record => `
<div class="ev">

  <div class="evTop">

    <span>
      ${escapeHTML(
        record.id ||
        'Record'
      )}
    </span>

    <span>
      ${escapeHTML(
        record.truth ||
        '?'
      )}
    </span>

  </div>

  <small>
    ${escapeHTML(
      record.domain ||
      'general'
    )}
    · Prediction ${escapeHTML(
      benchmarkPrediction(
        record
      )
    )}
    · Signal ${escapeHTML(
      String(
        record.consensus
          ?.calibrated ??
        '?'
      )
    )}%
    · Quality ${escapeHTML(
      String(
        record.consensus
          ?.quality ??
        '?'
      )
    )}%
  </small>

</div>
`;

  const fpHTML =
    falsePositives
      .slice()
      .reverse()
      .slice(
        0,
        8
      )
      .map(
        recordCard
      )
      .join('');

  const fnHTML =
    falseNegatives
      .slice()
      .reverse()
      .slice(
        0,
        8
      )
      .map(
        recordCard
      )
      .join('');

  const abstentionHTML =
    abstentions
      .slice()
      .reverse()
      .slice(
        0,
        8
      )
      .map(
        recordCard
      )
      .join('');

  const recentHTML =
    records
      .slice()
      .reverse()
      .slice(
        0,
        12
      )
      .map(
        recordCard
      )
      .join('');

  panel.innerHTML = `

<span class="over">
  V6.1 BENCHMARK LAB • DEVELOPMENT ONLY
</span>

<h2>
  Evidence Reliability Lab
</h2>

<p class="sub">
  Known samples are used to evaluate behavior. Abstentions are tracked separately and detector signals are not treated as proof of authorship.
</p>


<h3>
  Dataset
</h3>

<div class="metrics">

  <div class="metric">
    <span>Binary samples</span>
    <b>${metrics.total}</b>
  </div>

  <div class="metric">
    <span>AI</span>
    <b>${metrics.totalAI}</b>
  </div>

  <div class="metric">
    <span>HUMAN</span>
    <b>${metrics.totalHuman}</b>
  </div>

  <div class="metric">
    <span>MIXED</span>
    <b>${mixed.length}</b>
  </div>

</div>


<h3>
  Confusion matrix
</h3>

<div class="metrics">

  <div class="metric">
    <span>True positive</span>
    <b>${metrics.TP}</b>
  </div>

  <div class="metric">
    <span>True negative</span>
    <b>${metrics.TN}</b>
  </div>

  <div class="metric">
    <span>False positive</span>
    <b>${metrics.FP}</b>
  </div>

  <div class="metric">
    <span>False negative</span>
    <b>${metrics.FN}</b>
  </div>

  <div class="metric">
    <span>AI abstentions</span>
    <b>${metrics.aiAbstain}</b>
  </div>

  <div class="metric">
    <span>Human abstentions</span>
    <b>${metrics.humanAbstain}</b>
  </div>

</div>


<h3>
  Performance
</h3>

<div class="metrics">

  <div class="metric">
    <span>Coverage</span>
    <b>${metrics.coverage}%</b>
  </div>

  <div class="metric">
    <span>Selective accuracy</span>
    <b>${metrics.selectiveAccuracy}%</b>
  </div>

  <div class="metric">
    <span>Precision</span>
    <b>${metrics.precision}%</b>
  </div>

  <div class="metric">
    <span>AI recall</span>
    <b>${metrics.recall}%</b>
  </div>

  <div class="metric">
    <span>Human specificity</span>
    <b>${metrics.specificity}%</b>
  </div>

  <div class="metric">
    <span>False-positive rate</span>
    <b>${metrics.fpr}%</b>
  </div>

  <div class="metric">
    <span>False-negative rate</span>
    <b>${metrics.fnr}%</b>
  </div>

  <div class="metric">
    <span>AI abstention rate</span>
    <b>${metrics.aiAbstainRate}%</b>
  </div>

  <div class="metric">
    <span>Human abstention rate</span>
    <b>${metrics.humanAbstainRate}%</b>
  </div>

</div>


<h3>
  False-positive inspector
</h3>

<div class="evidence">

  ${
    fpHTML ||
    `
    <div class="ev">
      <small>
        No known HUMAN sample is currently classified as AI.
      </small>
    </div>
    `
  }

</div>


<h3>
  False-negative inspector
</h3>

<div class="evidence">

  ${
    fnHTML ||
    `
    <div class="ev">
      <small>
        No known AI sample is currently classified as HUMAN.
      </small>
    </div>
    `
  }

</div>


<h3>
  Abstention inspector
</h3>

<div class="evidence">

  ${
    abstentionHTML ||
    `
    <div class="ev">
      <small>
        No binary abstentions recorded.
      </small>
    </div>
    `
  }

</div>


<h3>
  Recent benchmark records
</h3>

<div class="evidence">

  ${
    recentHTML ||
    `
    <div class="ev">
      <small>
        No benchmark samples yet.
      </small>
    </div>
    `
  }

</div>
`;
}


/* =========================================================
   HISTORY
========================================================= */

function saveHistory(
  scan
) {
  const history =
    loadJSON(
      HISTORY_KEY,
      []
    );

  history.push({
    timestamp:
      scan.timestamp,

    version:
      VERSION,

    words:
      scan.words,

    language:
      scan.language,

    domain:
      scan.domain,

    scores:
      scan.scores,

    humanScore:
      scan.human.score,

    consensus:
      scan.consensus
  });

  saveJSON(
    HISTORY_KEY,
    history.slice(
      -100
    )
  );
}


/* =========================================================
   EXPORT JSON
========================================================= */

function exportBenchmarkJSON() {
  const data = {
    version:
      VERSION,

    exportedAt:
      nowISO(),

    metrics:
      benchmarkMetrics(),

    records:
      loadBench()
  };

  const blob =
    new Blob(
      [
        JSON.stringify(
          data,
          null,
          2
        )
      ],
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
    `AI-Trace-V61-Benchmark-${Date.now()}.json`;

  document.body
    .appendChild(
      anchor
    );

  anchor.click();

  anchor.remove();

  URL.revokeObjectURL(
    url
  );
}


/* =========================================================
   EXPORT CSV
========================================================= */

function exportBenchmarkCSV() {
  const records =
    loadBench();

  const headers = [
    'id',
    'truth',
    'source',
    'version',
    'timestamp',
    'domain',
    'language',
    'words',
    'tmr',
    'e5',
    'modern',
    'humanEvidence',
    'raw',
    'calibrated',
    'quality',
    'confidence',
    'uncertainty',
    'verdict',
    'activeModels',
    'modelSpread',
    'segmentRange'
  ];

  const csvEscape =
    value => {
      const stringValue =
        String(
          value ??
          ''
        );

      if (
        /[",\n]/.test(
          stringValue
        )
      ) {
        return `"${stringValue.replace(
          /"/g,
          '""'
        )}"`;
      }

      return stringValue;
    };

  const rows =
    records.map(
      record => [
        record.id,
        record.truth,
        record.source,
        record.version,
        record.timestamp ||
          record.savedAt,
        record.domain,
        record.language,
        record.words,
        record.scores?.tmr,
        record.scores?.e5,
        record.scores?.modern,
        record.human?.score,
        record.consensus?.raw,
        record.consensus?.calibrated,
        record.consensus?.quality,
        record.consensus?.confidence,
        record.consensus?.uncertainty,
        record.consensus?.verdict,
        record.consensus?.activeModels,
        record.consensus?.modelSpread,
        record.consensus?.segmentRange
      ]
        .map(
          csvEscape
        )
        .join(',')
    );

  const csv =
    [
      headers.join(','),
      ...rows
    ].join(
      '\n'
    );

  const blob =
    new Blob(
      [csv],
      {
        type:
          'text/csv;charset=utf-8'
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
    `AI-Trace-V61-Benchmark-${Date.now()}.csv`;

  document.body
    .appendChild(
      anchor
    );

  anchor.click();

  anchor.remove();

  URL.revokeObjectURL(
    url
  );
}


/* =========================================================
   DEVELOPER API
========================================================= */

window.AITraceV61 = {
  benchmark() {
    return {
      version:
        VERSION,

      metrics:
        benchmarkMetrics(),

      records:
        loadBench()
    };
  },

  history() {
    return loadJSON(
      HISTORY_KEY,
      []
    );
  },

  exportJSON() {
    exportBenchmarkJSON();
  },

  exportCSV() {
    exportBenchmarkCSV();
  },

  clearBenchmark() {
    const confirmation =
      confirm(
        'Delete all V6.1 benchmark records from this device?'
      );

    if (
      !confirmation
    ) {
      return;
    }

    localStorage.removeItem(
      BENCH_KEY
    );

    renderBenchmarkPanel();

    alert(
      'V6.1 benchmark data deleted.'
    );
  },

  clearHistory() {
    const confirmation =
      confirm(
        'Delete local V6.1 scan history?'
      );

    if (
      !confirmation
    ) {
      return;
    }

    localStorage.removeItem(
      HISTORY_KEY
    );

    alert(
      'V6.1 scan history deleted.'
    );
  }
};


/* =========================================================
   INIT
========================================================= */

updateCount();

setTimeout(
  renderBenchmarkPanel,
  350
);

console.info(
  `AI TRACE V${VERSION} loaded`
);
