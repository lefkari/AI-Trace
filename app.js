/*
  ============================================================
  AI TRACE V5.2 — CONFLICT-AWARE AUTHENTICITY ENGINE
  ============================================================

  Main upgrades from V5.1:

  - Conflict-aware verdict system
  - No misleading headline AI percentage when evidence conflicts
  - Better literary / book-domain detection
  - Stronger false-positive protection
  - 3-model desktop consensus
  - 2-model mobile-safe consensus
  - Human counter-evidence engine
  - Segment instability analysis
  - Evidence-quality scoring
  - Abstention / unresolved state
  - V5.1 benchmark migration
  - Zero paid API architecture

  Important:
  AI detection is probabilistic evidence.
  It is not proof of authorship.
*/

import {
  pipeline,
  env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;


/* ============================================================
   VERSION + MODELS
============================================================ */

const VERSION = '5.2';

const MODEL_TMR =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODEL_E5 =
  'onnx-community/e5-small-lora-ai-generated-detector-ONNX';

const MODEL_MODERN =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';


/* ============================================================
   STORAGE
============================================================ */

const BENCH_KEY =
  'aiTraceBenchmarkV52';

const LEGACY_BENCH_KEYS = [
  'aiTraceBenchmarkV51',
  'aiTraceBenchmarkV5'
];

const HISTORY_KEY =
  'aiTraceHistoryV52';


/* ============================================================
   MODEL INSTANCES
============================================================ */

let tmr = null;
let e5 = null;
let modern = null;


/* ============================================================
   DOM
============================================================ */

const $ = id =>
  document.getElementById(id);

const textEl =
  $('text');


/* ============================================================
   DEVICE
============================================================ */

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


/* ============================================================
   UI EVENTS
============================================================ */

if (textEl) {

  textEl.addEventListener(
    'input',
    updateCount
  );
}


if ($('clear')) {

  $('clear').addEventListener(
    'click',
    () => {

      textEl.value = '';

      updateCount();

      $('report')
        ?.classList
        .add('hidden');
    }
  );
}


if ($('demo')) {

  $('demo').addEventListener(
    'click',
    loadDemo
  );
}


if ($('scan')) {

  $('scan').addEventListener(
    'click',
    runSmartScan
  );
}


/* ============================================================
   UI HELPERS
============================================================ */

function updateCount() {

  const words =
    wordCount(
      textEl?.value || ''
    );

  if ($('count')) {

    $('count').textContent =
      `${words} words`;
  }
}


function loadDemo() {

  if (!textEl) {
    return;
  }

  textEl.value = `Artificial intelligence is rapidly changing the way people work, communicate, and interact with technology. Over the past few years, AI systems have become capable of generating text, creating images, analyzing complex information, and assisting people with tasks that previously required significant amounts of human effort.

One of the most important advantages of artificial intelligence is its ability to process large amounts of information quickly. Organizations can use AI-powered tools to identify patterns, automate repetitive processes, and support better decision-making. For example, businesses can analyze customer behavior, doctors can receive assistance when examining medical information, and researchers can process datasets that would be extremely difficult to evaluate manually.

However, the growing use of artificial intelligence also creates important challenges. AI-generated information can sometimes be inaccurate, misleading, or difficult to distinguish from content created by humans. Synthetic images, artificial voices, and automatically generated articles are becoming increasingly realistic.

The future will therefore require more than simply developing increasingly powerful artificial intelligence systems. Society will also need technologies that provide transparency, verification, and evidence about how digital content was created or modified.`;

  updateCount();
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
    () => {

      $('progress')
        ?.classList
        .add('hidden');

    },
    450
  );
}


function setState(label) {

  if ($('modelState')) {

    $('modelState').textContent =
      label;
  }
}


/* ============================================================
   GENERIC HELPERS
============================================================ */

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

  return (
    usable.reduce(
      (sum, value) =>
        sum + value,
      0
    ) /
    usable.length
  );
}


function standardDeviation(values) {

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
      .filter(Number.isFinite)
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
    usable.length % 2 === 1
  ) {

    return usable[middle];
  }

  return (
    usable[middle - 1] +
    usable[middle]
  ) / 2;
}


function wordCount(value) {

  const trimmed =
    String(value || '')
      .trim();

  if (!trimmed) {
    return 0;
  }

  return trimmed
    .split(/\s+/)
    .filter(Boolean)
    .length;
}


function escapeHTML(value) {

  return String(value).replace(
    /[&<>"']/g,
    character => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    })[character]
  );
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
    latin / total >
    0.82
  )
    ? 'English'
    : 'Non-English';
}


/* ============================================================
   DOCUMENT PROFILE
============================================================ */

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

  const cleanWords =
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

  const commaCount =
    countMatches(
      value,
      /,/g
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

  const transitions =
    countMatches(
      value,
      /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|in conclusion|additionally|nevertheless|as a result)\b/gi
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
      new Set(cleanWords).size /
      Math.max(
        1,
        cleanWords.length
      ),

    punctuationTypes,

    quoteCount,

    semicolonCount,

    commaCount,

    dialogueLines,

    firstPerson,

    contractions,

    transitions,

    titleReferences,

    narrativeMarkers,

    literaryVocabulary
  };
}


/* ============================================================
   DOMAIN ESTIMATION V5.2
============================================================ */

function estimateDomain(
  value,
  profile
) {

  const content =
    value.toLowerCase();

  const scores = {

    abstracts:
      countMatches(
        content,
        /\b(method|methods|results|conclusion|study|participants|dataset|experiment|analysis|significant|hypothesis|abstract|research|findings)\b/g
      ) * 2,

    recipes:
      countMatches(
        content,
        /\b(cup|cups|tablespoon|teaspoon|ingredients|preheat|oven|bake|stir|chop|minutes|serve|flour|sugar)\b/g
      ) * 2,

    reviews:
      countMatches(
        content,
        /\b(review|rating|stars|recommend|purchase|product|quality|price|experience|bought|customer)\b/g
      ) * 2,

    reddit:
      countMatches(
        content,
        /\b(aita|tldr|subreddit|upvote|downvote|edit:|throwaway|imo|lol|op)\b/g
      ) * 3,

    wiki:
      countMatches(
        content,
        /\b(was born|refers to|located in|population|history of|known for|founded|species|geography|is a type of)\b/g
      ) * 2,

    news:
      countMatches(
        content,
        /\b(reuters|reported|according to|officials|government|minister|president|announced|agency|spokesperson|statement)\b/g
      ) * 2,

    poetry:
      0,

    books:
      0
  };


  /*
    Poetry
  */

  if (
    profile.lineBreaks >= 5 &&
    profile.averageLineLength < 70
  ) {

    scores.poetry += 8;
  }


  /*
    Literary prose / books.

    Important for false-positive protection.
  */

  scores.books +=
    profile.titleReferences * 3;

  scores.books +=
    Math.min(
      12,
      profile.literaryVocabulary * 2
    );

  scores.books +=
    Math.min(
      8,
      Math.floor(
        profile.narrativeMarkers / 3
      )
    );

  if (
    profile.averageSentenceLength >= 25
  ) {

    scores.books += 3;
  }

  if (
    profile.semicolonCount >= 2
  ) {

    scores.books += 3;
  }

  if (
    profile.quoteCount >= 4
  ) {

    scores.books += 3;
  }

  if (
    profile.dialogueLines >= 2
  ) {

    scores.books += 5;
  }

  if (
    profile.sentenceBurstiness >= 0.45
  ) {

    scores.books += 2;
  }


  const sorted =
    Object.entries(scores)
      .sort(
        (a, b) =>
          b[1] - a[1]
      );

  const [
    domain,
    score
  ] =
    sorted[0];


  if (
    !score ||
    score < 4
  ) {

    return {

      domain:
        'general',

      confidence:
        'low',

      score:
        score || 0
    };
  }


  return {

    domain,

    confidence:
      score >= 12
        ? 'high'
        : score >= 7
          ? 'medium'
          : 'low',

    score
  };
}


/* ============================================================
   HUMAN COUNTER-EVIDENCE V5.2
============================================================ */

function humanEvidence(
  profile,
  domain
) {

  let score = 0;

  const reasons = [];


  if (
    profile.sentenceBurstiness >=
    0.65
  ) {

    score += 22;

    reasons.push(
      'high sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >=
    0.45
  ) {

    score += 14;

    reasons.push(
      'moderate sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >=
    0.30
  ) {

    score += 6;
  }


  if (
    profile.punctuationTypes >= 5
  ) {

    score += 12;

    reasons.push(
      'rich punctuation variety'
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

    score += 14;

    reasons.push(
      'dialogue / quotation structure'
    );

  } else if (
    profile.quoteCount >= 3
  ) {

    score += 6;
  }


  if (
    profile.firstPerson >= 4
  ) {

    score += 8;

    reasons.push(
      'personal voice'
    );

  } else if (
    profile.firstPerson > 0
  ) {

    score += 4;
  }


  if (
    profile.contractions >= 4
  ) {

    score += 5;
  }


  if (
    profile.paragraphDeviation >= 25 &&
    profile.paragraphs >= 3
  ) {

    score += 8;

    reasons.push(
      'irregular paragraph rhythm'
    );
  }


  if (
    profile.lexicalDiversity >= 0.62
  ) {

    score += 6;
  }


  /*
    Literary evidence
  */

  if (
    profile.titleReferences >= 2
  ) {

    score += 8;

    reasons.push(
      'literary character references'
    );
  }


  if (
    profile.literaryVocabulary >= 3
  ) {

    score += 10;

    reasons.push(
      'literary prose characteristics'
    );
  }


  if (
    profile.semicolonCount >= 3 &&
    profile.averageSentenceLength >= 24
  ) {

    score += 8;

    reasons.push(
      'complex literary sentence structure'
    );
  }


  /*
    Excessively formulaic transitions can be
    weak machine-style evidence.
  */

  if (
    profile.transitions >= 4
  ) {

    score -= 6;
  }


  if (
    domain === 'books'
  ) {

    score += 14;

    reasons.push(
      'literary-domain protection'
    );
  }


  if (
    domain === 'poetry'
  ) {

    score += 12;

    reasons.push(
      'poetry-domain protection'
    );
  }


  return {

    score:
      clamp(
        Math.round(score)
      ),

    reasons:
      [
        ...new Set(reasons)
      ]
  };
}


/* ============================================================
   CHUNKING
============================================================ */

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
      ).length >
        maxChars &&
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


/* ============================================================
   MODEL LOADERS
============================================================ */

async function loadTMR() {

  if (tmr) {
    return tmr;
  }

  setState(
    'Loading TMR…'
  );

  setProgress(
    8,
    'Loading Model A…'
  );

  tmr =
    await pipeline(
      'text-classification',
      MODEL_TMR,
      {
        dtype:
          'q4f16'
      }
    );

  return tmr;
}


async function loadE5() {

  if (e5) {
    return e5;
  }

  setState(
    'Loading E5-small…'
  );

  setProgress(
    18,
    'Loading Model B…'
  );

  e5 =
    await pipeline(
      'text-classification',
      MODEL_E5,
      {
        dtype:
          'q4f16'
      }
    );

  return e5;
}


async function loadModern() {

  if (modern) {
    return modern;
  }

  setState(
    'Loading ModernBERT judge…'
  );

  setProgress(
    72,
    'Deep verification: loading Model C…'
  );

  modern =
    await pipeline(
      'text-classification',
      MODEL_MODERN,
      {
        dtype:
          'q4f16'
      }
    );

  return modern;
}


/* ============================================================
   MODEL OUTPUT NORMALIZATION
============================================================ */

function aiProbability(output) {

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
        item.label || ''
      ).toLowerCase();

    const score =
      Number(
        item.score
      ) || 0;


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

    return ai;
  }


  if (
    human !== null
  ) {

    return (
      1 -
      human
    );
  }


  if (
    results.length >= 2
  ) {

    return Number(
      results[1]?.score ??
      0.5
    );
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
    aiProbability(result) *
    100
  );
}


/* ============================================================
   THIRD MODEL ROUTING
============================================================ */

function shouldUseThirdModel({
  quickScores,
  segmentScores,
  domain,
  humanScore,
  words
}) {

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
      raw >= 30 &&
      raw <= 88
    ) ||

    segmentRange >= 40 ||

    segmentSD >= 18 ||

    humanScore >= 40 ||

    domain === 'books' ||

    domain === 'poetry'
  );
}


/* ============================================================
   CONFLICT CLASSIFIER V5.2
============================================================ */

function analyzeConflict({
  activeScores,
  modelSpread,
  segmentRange,
  segmentSD,
  quality,
  domain,
  humanScore
}) {

  const reasons = [];

  let severity = 0;


  if (
    modelSpread >= 60
  ) {

    severity += 3;

    reasons.push(
      'extreme model disagreement'
    );

  } else if (
    modelSpread >= 40
  ) {

    severity += 2;

    reasons.push(
      'high model disagreement'
    );

  } else if (
    modelSpread >= 28
  ) {

    severity += 1;

    reasons.push(
      'moderate model disagreement'
    );
  }


  if (
    segmentRange >= 70
  ) {

    severity += 3;

    reasons.push(
      'extreme segment variation'
    );

  } else if (
    segmentRange >= 50
  ) {

    severity += 2;

    reasons.push(
      'high segment variation'
    );

  } else if (
    segmentRange >= 35
  ) {

    severity += 1;

    reasons.push(
      'moderate segment variation'
    );
  }


  if (
    segmentSD >= 28
  ) {

    severity += 2;

    reasons.push(
      'unstable segment scores'
    );
  }


  if (
    quality < 35
  ) {

    severity += 2;

    reasons.push(
      'low evidence quality'
    );

  } else if (
    quality < 50
  ) {

    severity += 1;
  }


  /*
    Direct directional conflict:
    at least one detector says AI and
    another detector says Human.
  */

  const hasHighAI =
    activeScores.some(
      score =>
        score >= 75
    );

  const hasHighHuman =
    activeScores.some(
      score =>
        score <= 30
    );


  if (
    hasHighAI &&
    hasHighHuman
  ) {

    severity += 3;

    reasons.push(
      'detectors point in opposite directions'
    );
  }


  if (
    (
      domain === 'books' ||
      domain === 'poetry'
    ) &&
    humanScore >= 40
  ) {

    severity += 1;

    reasons.push(
      'literary-domain false-positive risk'
    );
  }


  return {

    severity,

    unresolved:
      severity >= 4,

    reasons:
      [
        ...new Set(reasons)
      ]
  };
}


/* ============================================================
   CONSENSUS ENGINE V5.2
============================================================ */

function buildConsensus({
  scores,
  segmentScores,
  profile,
  language,
  domain,
  human,
  thirdUsed
}) {

  const active =
    [
      scores.tmr,
      scores.e5,
      scores.modern
    ]
      .filter(
        Number.isFinite
      );


  const raw =
    Math.round(
      median(active)
    );


  const modelSpread =
    active.length > 1

      ? Math.max(
          ...active
        ) -
        Math.min(
          ...active
        )

      : 100;


  const modelSD =
    Math.round(
      standardDeviation(active)
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


  const segmentSD =
    Math.round(
      standardDeviation(
        segmentScores
      )
    );


  const domainRisk =
    domain === 'books'
      ? 20
      : domain === 'poetry'
        ? 22
        : domain === 'general'
          ? 7
          : 4;


  /*
    Evidence quality
  */

  let quality = 100;


  quality -=
    Math.min(
      42,
      modelSpread * 0.72
    );


  quality -=
    Math.min(
      32,

      segmentSD * 0.65 +

      segmentRange * 0.18
    );


  quality -=
    profile.words < 120
      ? 18
      : profile.words < 180
        ? 12
        : profile.words < 250
          ? 5
          : 0;


  quality -=
    language === 'English'
      ? 0
      : 35;


  quality -=
    domainRisk;


  /*
    Two-model results receive a small
    evidence-quality reduction but are
    still valid on mobile.
  */

  if (
    active.length < 3
  ) {

    quality -= 5;
  }


  quality =
    clamp(
      Math.round(quality)
    );


  const instability =
    1 -
    quality / 100;


  /*
    Human evidence modifies confidence,
    but it never directly "proves"
    human authorship.
  */

  const humanPenalty =
    human.score *
    (
      0.08 +
      instability * 0.42
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
    Literary safeguard
  */

  if (
    (
      domain === 'books' ||
      domain === 'poetry'
    ) &&
    human.score >= 40
  ) {

    calibrated =
      Math.min(
        calibrated,
        68
      );
  }


  const conflict =
    analyzeConflict({

      activeScores:
        active,

      modelSpread,

      segmentRange,

      segmentSD,

      quality,

      domain,

      humanScore:
        human.score
    });


  /*
    Important V5.2 behavior:

    If evidence conflicts badly,
    headline percentage is suppressed.
  */

  const unresolved =
    conflict.unresolved ||
    language !== 'English' ||
    active.length < 2;


  let uncertainty =
    clamp(
      Math.round(
        100 -
        quality +
        conflict.severity * 4
      ),
      5,
      98
    );


  if (
    unresolved
  ) {

    uncertainty =
      Math.max(
        uncertainty,
        70
      );
  }


  const confidence =
    100 -
    uncertainty;


  let verdict =
    'INCONCLUSIVE';


  if (
    unresolved
  ) {

    verdict =
      conflict.unresolved
        ? 'CONFLICTING EVIDENCE'
        : 'INCONCLUSIVE';

  } else if (

    calibrated >= 87 &&

    quality >= 72 &&

    modelSpread <= 22 &&

    segmentRange <= 45 &&

    human.score < 45

  ) {

    verdict =
      'Strong AI evidence';

  } else if (

    calibrated >= 74 &&

    quality >= 58 &&

    modelSpread <= 26 &&

    segmentRange <= 55 &&

    human.score < 52

  ) {

    verdict =
      'Likely AI';

  } else if (

    calibrated <= 18 &&

    quality >= 58 &&

    human.score >= 48

  ) {

    verdict =
      'Strong human evidence';

  } else if (

    calibrated <= 35 &&

    quality >= 48 &&

    human.score >= 40 &&

    modelSpread <= 28

  ) {

    verdict =
      'Likely human';

  } else {

    verdict =
      'INCONCLUSIVE';
  }


  const suppressHeadline =
    verdict ===
      'CONFLICTING EVIDENCE' ||

    (
      verdict ===
        'INCONCLUSIVE' &&
      quality < 40
    );


  return {

    raw,

    calibrated,

    quality,

    uncertainty,

    confidence,

    verdict,

    unresolved,

    suppressHeadline,

    conflictSeverity:
      conflict.severity,

    conflictReasons:
      conflict.reasons,

    modelSpread:
      Math.round(
        modelSpread
      ),

    modelSD,

    segmentRange:
      Math.round(
        segmentRange
      ),

    segmentSD,

    activeModels:
      active.length,

    thirdUsed,

    humanPenalty:
      Math.round(
        humanPenalty
      )
  };
}


/* ============================================================
   MAIN SCAN
============================================================ */

async function runSmartScan() {

  const value =
    textEl
      ?.value
      .trim() || '';

  const words =
    wordCount(value);


  if (
    words < 80
  ) {

    alert(
      'Paste at least 80 words. 150+ words is recommended.'
    );

    return;
  }


  if ($('scan')) {

    $('scan').disabled =
      true;
  }


  try {

    setProgress(
      3,
      'Profiling document…'
    );


    const language =
      detectLanguage(value);


    const profile =
      createProfile(value);


    const domainInfo =
      estimateDomain(
        value,
        profile
      );


    const human =
      humanEvidence(
        profile,
        domainInfo.domain
      );


    const chunks =
      chunkText(value);


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


    /*
      MODEL A — TMR
    */

    try {

      const modelA =
        await loadTMR();


      setProgress(
        22,
        'Running Model A…'
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
            ) *
            28
          ),

          `Trace Map ${i + 1}/${chunks.length}…`
        );


        segmentScores.push(
          await classify(
            modelA,
            chunks[i]
          )
        );
      }

    } catch (error) {

      console.error(
        'TMR failed:',
        error
      );
    }


    /*
      MODEL B — E5
    */

    try {

      const modelB =
        await loadE5();


      setProgress(
        61,
        'Running Model B…'
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


    /*
      Third model routing.
      Desktop only for V5.2.
    */

    const mobile =
      isMobileDevice();


    if (
      Number.isFinite(
        scores.tmr
      ) &&
      Number.isFinite(
        scores.e5
      )
    ) {

      thirdUsed =
        !mobile &&

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


    /*
      MODEL C — ModernBERT
    */

    if (
      thirdUsed
    ) {

      try {

        const modelC =
          await loadModern();


        setProgress(
          78,
          'Running Model C adjudication…'
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


    if (
      !segmentScores.length
    ) {

      segmentScores =
        chunks.map(
          () => 50
        );
    }


    setProgress(
      90,
      'Resolving evidence…'
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
        new Date()
          .toISOString(),

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


    try {

      render(scan);

    } catch (error) {

      console.error(
        'Render error:',
        error
      );

      alert(
        'Analysis completed, but the interface encountered an error. Open the browser console for details.'
      );
    }


    try {

      saveHistory(scan);

    } catch (error) {

      console.warn(
        'History save failed:',
        error
      );
    }


    setProgress(
      100,
      consensus.unresolved
        ? 'Evidence conflict detected'
        : 'Trace complete'
    );


    if (
      consensus.verdict ===
      'CONFLICTING EVIDENCE'
    ) {

      setState(
        'V5.2 Conflict Defense activated ✓'
      );

    } else if (
      mobile
    ) {

      setState(
        'V5.2 Mobile Safe • 2-model engine ✓'
      );

    } else if (
      thirdUsed
    ) {

      setState(
        'V5.2 Deep 3-model consensus ✓'
      );

    } else {

      setState(
        'V5.2 Smart consensus ✓'
      );
    }


    setTimeout(
      () => {

        try {

          benchmarkPrompt(scan);

        } catch (error) {

          console.warn(
            'Benchmark prompt failed:',
            error
          );
        }

      },
      700
    );

  } catch (error) {

    console.error(
      'Fatal scan error:',
      error
    );


    setState(
      'Scan error'
    );


    alert(
      'AI Trace encountered an error during analysis. Refresh the page and try again.'
    );

  } finally {

    if ($('scan')) {

      $('scan').disabled =
        false;
    }

    hideProgress();
  }
}


/* ============================================================
   RENDER V5.2
============================================================ */

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


  /*
    Critical V5.2 change:
    suppress misleading headline probability
    during severe evidence conflict.
  */

  if (
    consensus.suppressHeadline
  ) {

    if ($('score')) {

      $('score').textContent =
        '—';
    }

    if ($('scaleFill')) {

      $('scaleFill').style.width =
        '0%';
    }

  } else {

    if ($('score')) {

      $('score').textContent =
        `${consensus.calibrated}%`;
    }

    if ($('scaleFill')) {

      $('scaleFill').style.width =
        `${consensus.calibrated}%`;
    }
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

    if (
      consensus.verdict ===
      'CONFLICTING EVIDENCE'
    ) {

      const reasonText =
        consensus.conflictReasons.length
          ? consensus.conflictReasons.join(
              ', '
            )
          : 'inconsistent detector evidence';


      $('explain').textContent =
        `AI Trace cannot responsibly display a single AI probability for this document. The detectors conflict materially (${reasonText}). Raw and calibrated signals remain available below only as diagnostics.`;

    } else if (
      consensus.verdict ===
      'INCONCLUSIVE'
    ) {

      $('explain').textContent =
        `The evidence is insufficient for a reliable AI/Human classification. Diagnostic AI signal: ${consensus.calibrated}%. Evidence quality: ${consensus.quality}%.`;

    } else {

      $('explain').textContent =
        `Raw ensemble signal ${consensus.raw}%; calibrated to ${consensus.calibrated}% after model agreement, segment stability, domain context and human counter-evidence.`;
    }
  }


  /*
    TRACE DNA
  */

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


  if (
    consensus.suppressHeadline
  ) {

    if ($('humanVal')) {

      $('humanVal').textContent =
        `${humanDisplay}%`;
    }

    if ($('aiVal')) {

      $('aiVal').textContent =
        'N/A';
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
        '0%';
    }

    if ($('uncertainBar')) {

      $('uncertainBar').style.width =
        `${consensus.uncertainty}%`;
    }

  } else {

    if ($('humanVal')) {

      $('humanVal').textContent =
        `${humanDisplay}%`;
    }

    if ($('aiVal')) {

      $('aiVal').textContent =
        `${consensus.calibrated}%`;
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
        `${consensus.calibrated}%`;
    }

    if ($('uncertainBar')) {

      $('uncertainBar').style.width =
        `${consensus.uncertainty}%`;
    }
  }


  /*
    ENGINE BADGE
  */

  if ($('engineBadge')) {

    $('engineBadge').textContent =

      consensus.verdict ===
        'CONFLICTING EVIDENCE'

        ? 'V5.2 • CONFLICT DEFENSE'

        : consensus.thirdUsed

          ? 'V5.2 • 3-MODEL CONSENSUS'

          : isMobileDevice()

            ? 'V5.2 • MOBILE SAFE'

            : 'V5.2 • SMART CONSENSUS';
  }


  const humanReasons =
    human.reasons.length

      ? human.reasons
          .slice(
            0,
            5
          )
          .join(
            ' • '
          )

      : 'No strong human-style counter-signals';


  const modelCText =
    Number.isFinite(
      scores.modern
    )

      ? `${scores.modern}% AI signal`

      : isMobileDevice()

        ? 'Disabled on mobile for stability'

        : 'Not required by Smart Scan';


  const conflictText =
    consensus.conflictReasons.length

      ? consensus.conflictReasons.join(
          ' • '
        )

      : 'No severe conflict detected';


  const evidence = [

    [
      'Decision status',

      consensus.verdict,

      consensus.unresolved
        ? 'Abstention'
        : 'Resolved'
    ],

    [
      'Diagnostic calibrated signal',

      `${consensus.calibrated}%`,

      'Do not interpret alone'
    ],

    [
      'Raw ensemble signal',

      `${consensus.raw}%`,

      'Diagnostic'
    ],

    [
      'Evidence quality',

      `${consensus.quality}%`,

      consensus.quality >= 75
        ? 'Strong'
        : consensus.quality >= 50
          ? 'Medium'
          : 'Weak'
    ],

    [
      'Conflict analysis',

      conflictText,

      consensus.conflictSeverity >= 4
        ? 'Critical'
        : consensus.conflictSeverity >= 2
          ? 'Elevated'
          : 'Normal'
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

      modelCText,

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

      consensus.modelSpread >= 60
        ? 'Extreme conflict'
        : consensus.modelSpread >= 40
          ? 'High conflict'
          : consensus.modelSpread >= 28
            ? 'Moderate'
            : 'Acceptable'
    ],

    [
      'Segment range',

      `${consensus.segmentRange} points`,

      consensus.segmentRange >= 70
        ? 'Extreme variation'
        : consensus.segmentRange >= 50
          ? 'High variation'
          : 'Acceptable'
    ],

    [
      'Domain context',

      `${domain} (${domainConfidence} confidence)`,

      (
        domain === 'books' ||
        domain === 'poetry'
      )
        ? 'Protected domain'
        : 'Routing'
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

    <span>${escapeHTML(item[0])}</span>

    <span>${escapeHTML(item[2])}</span>

  </div>

  <small>${escapeHTML(item[1])}</small>

</div>

`
        )
        .join('');
  }


  /*
    METRICS
  */

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

    'Mobile safe mode':
      isMobileDevice()
        ? 'Yes'
        : 'No',

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

    'Model spread':
      `${consensus.modelSpread} pts`,

    'Segment deviation':
      consensus.segmentSD,

    'Segment range':
      `${consensus.segmentRange} pts`,

    'Evidence quality':
      `${consensus.quality}%`,

    'Conflict severity':
      consensus.conflictSeverity,

    'Decision':
      consensus.verdict
  };


  if ($('metrics')) {

    $('metrics').innerHTML =
      Object.entries(metrics)
        .map(
          ([key, value]) => `

<div class="metric">

  <span>${escapeHTML(key)}</span>

  <b>${escapeHTML(String(value))}</b>

</div>

`
        )
        .join('');
  }


  /*
    TRACE MAP
  */

  const chunks =
    chunkText(
      textEl.value.trim()
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

    <b>Segment ${index + 1}</b>

    <span>${score}% TMR diagnostic</span>

  </div>

  <div class="segmentMeter">

    <i style="width:${score}%"></i>

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


/* ============================================================
   BENCHMARK STORAGE + V5.1 MIGRATION
============================================================ */

function loadJSON(key) {

  try {

    return JSON.parse(
      localStorage.getItem(
        key
      ) || '[]'
    );

  } catch {

    return [];
  }
}


function loadBench() {

  const current =
    loadJSON(
      BENCH_KEY
    );


  if (
    current.length
  ) {

    return current;
  }


  /*
    Preserve V5.1 benchmark records.
  */

  for (
    const key
    of LEGACY_BENCH_KEYS
  ) {

    const legacy =
      loadJSON(key);


    if (
      legacy.length
    ) {

      const migrated =
        legacy.map(
          record => ({
            ...record,

            migratedFrom:
              record.version ||
              'V5.1'
          })
        );


      try {

        localStorage.setItem(
          BENCH_KEY,
          JSON.stringify(
            migrated
          )
        );

      } catch {}


      return migrated;
    }
  }


  return [];
}


function saveBench(records) {

  try {

    localStorage.setItem(
      BENCH_KEY,
      JSON.stringify(
        records
      )
    );

  } catch (error) {

    console.warn(
      'Benchmark save failed:',
      error
    );
  }
}


/* ============================================================
   BENCHMARK PROMPT
============================================================ */

function benchmarkPrompt(scan) {

  const answer =
    prompt(
`AI TRACE V5.2 BENCHMARK

Only label samples whose true origin you KNOW.

AI      = definitely AI-generated
HUMAN   = definitely human-written
MIXED   = known human + AI mixture
UNKNOWN = origin is not known

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
    ].includes(truth)
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


  const sameTruthCount =
    records.filter(
      record =>
        record.truth ===
        truth
    ).length;


  records.push({

    id:
      `${truth[0]}-${String(
        sameTruthCount + 1
      ).padStart(
        3,
        '0'
      )}`,

    truth,

    source,

    ...scan
  });


  saveBench(records);

  renderBenchmarkPanel();


  const results =
    benchmarkMetrics(records);


  alert(
`Saved.

Binary samples: ${results.total}
Coverage: ${results.coverage}%
Selective accuracy: ${results.selectiveAccuracy}%
False-positive rate: ${results.fpr}%
False-negative rate: ${results.fnr}%
Human abstention rate: ${results.humanAbstainRate}%

Conflict / inconclusive results count as abstentions.

Development data only — not a production accuracy claim.`
  );
}


/* ============================================================
   BENCHMARK PREDICTION
============================================================ */

function prediction(record) {

  const verdict =
    record.consensus
      ?.verdict ||
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


  /*
    CONFLICTING EVIDENCE and
    INCONCLUSIVE are abstentions.
  */

  return 'ABSTAIN';
}


/* ============================================================
   BENCHMARK METRICS
============================================================ */

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
      prediction(record);


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
        record.truth ===
        'AI'
    ).length;


  const totalHuman =
    rows.filter(
      record =>
        record.truth ===
        'HUMAN'
    ).length;


  const decided =
    TP +
    TN +
    FP +
    FN;


  const total =
    rows.length;


  const percentage =
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

    aiAbstain,

    humanAbstain,

    decided,

    abstained:
      aiAbstain +
      humanAbstain,

    coverage:
      percentage(
        decided,
        total
      ),

    selectiveAccuracy:
      percentage(
        TP + TN,
        decided
      ),

    precision:
      percentage(
        TP,
        TP + FP
      ),

    recall:
      percentage(
        TP,
        totalAI
      ),

    specificity:
      percentage(
        TN,
        totalHuman
      ),

    fpr:
      percentage(
        FP,
        totalHuman
      ),

    fnr:
      percentage(
        FN,
        totalAI
      ),

    humanAbstainRate:
      percentage(
        humanAbstain,
        totalHuman
      ),

    aiAbstainRate:
      percentage(
        aiAbstain,
        totalAI
      )
  };
}


/* ============================================================
   BENCHMARK PANEL
============================================================ */

function renderBenchmarkPanel() {

  const report =
    $('report');


  if (!report) {
    return;
  }


  let panel =
    $('benchmarkPanelV5');


  if (!panel) {

    panel =
      document.createElement(
        'section'
      );

    panel.id =
      'benchmarkPanelV5';

    panel.className =
      'panel devPanel';

    report.appendChild(
      panel
    );
  }


  const records =
    loadBench();


  const results =
    benchmarkMetrics(
      records
    );


  const recent =
    records
      .slice()
      .reverse()
      .slice(
        0,
        12
      )
      .map(
        record => {

          const consensus =
            record.consensus ||
            {};

          const headline =
            consensus.suppressHeadline
              ? 'UNRESOLVED'
              : Number.isFinite(
                  consensus.calibrated
                )
                ? `${consensus.calibrated}%`
                : '?';


          return `

<div class="ev">

  <div class="evTop">

    <span>${escapeHTML(
      record.id || 'Record'
    )}</span>

    <span>${escapeHTML(
      record.truth || '?'
    )}</span>

  </div>

  <small>

    ${escapeHTML(
      record.domain ||
      'general'
    )}

    · result ${escapeHTML(
      headline
    )}

    · quality ${escapeHTML(
      String(
        consensus.quality ??
        '?'
      )
    )}%

    · ${escapeHTML(
      consensus.verdict ||
      'legacy'
    )}

  </small>

</div>

`;

        }
      )
      .join('');


  panel.innerHTML = `

<span class="over">
  V5.2 BENCHMARK • DEVELOPMENT ONLY
</span>

<h2>
  Reliability Dashboard
</h2>

<p class="sub">
  Conflict and inconclusive results are counted as abstentions rather than silently treated as correct predictions.
</p>

<div class="metrics">

  <div class="metric">
    <span>Binary samples</span>
    <b>${results.total}</b>
  </div>

  <div class="metric">
    <span>Decided samples</span>
    <b>${results.decided}</b>
  </div>

  <div class="metric">
    <span>Abstentions</span>
    <b>${results.abstained}</b>
  </div>

  <div class="metric">
    <span>Coverage</span>
    <b>${results.coverage}%</b>
  </div>

  <div class="metric">
    <span>Selective accuracy</span>
    <b>${results.selectiveAccuracy}%</b>
  </div>

  <div class="metric">
    <span>Precision</span>
    <b>${results.precision}%</b>
  </div>

  <div class="metric">
    <span>AI recall</span>
    <b>${results.recall}%</b>
  </div>

  <div class="metric">
    <span>Human specificity</span>
    <b>${results.specificity}%</b>
  </div>

  <div class="metric">
    <span>False-positive rate</span>
    <b>${results.fpr}%</b>
  </div>

  <div class="metric">
    <span>False-negative rate</span>
    <b>${results.fnr}%</b>
  </div>

  <div class="metric">
    <span>Human abstention rate</span>
    <b>${results.humanAbstainRate}%</b>
  </div>

  <div class="metric">
    <span>AI abstention rate</span>
    <b>${results.aiAbstainRate}%</b>
  </div>

</div>


<h3>
  Recent benchmark records
</h3>


<div class="evidence">

  ${
    recent ||
    `
    <div class="ev">
      <small>
        No V5.2 benchmark records yet.
      </small>
    </div>
    `
  }

</div>

`;
}


/* ============================================================
   HISTORY
============================================================ */

function saveHistory(scan) {

  try {

    const history =
      loadJSON(
        HISTORY_KEY
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


    localStorage.setItem(
      HISTORY_KEY,
      JSON.stringify(
        history.slice(
          -100
        )
      )
    );

  } catch (error) {

    console.warn(
      'History save failed:',
      error
    );
  }
}


/* ============================================================
   DEVELOPER UTILITIES
============================================================ */

window.AITraceV52 = {

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
      HISTORY_KEY
    );
  },


  clearBenchmark() {

    if (
      confirm(
        'Delete all V5.2 benchmark records from this device?'
      )
    ) {

      localStorage.removeItem(
        BENCH_KEY
      );

      renderBenchmarkPanel();
    }
  },


  exportBenchmark() {

    const data =
      JSON.stringify(
        this.benchmark(),
        null,
        2
      );


    const blob =
      new Blob(
        [data],
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
      `AI-Trace-V52-Benchmark-${Date.now()}.json`;


    anchor.click();


    URL.revokeObjectURL(
      url
    );
  }
};


/* ============================================================
   INITIALIZE
============================================================ */

updateCount();

setTimeout(
  renderBenchmarkPanel,
  350
);
