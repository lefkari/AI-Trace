/*
  AI TRACE V5.1 — MOBILE SAFE SMART ENGINE

  Mobile:
  - TMR
  - E5-small
  - Human counter-evidence
  - Domain context
  - Segment analysis
  - Calibration

  Desktop:
  - TMR
  - E5-small
  - ModernBERT conditional judge

  Zero paid API.
*/

import {
  pipeline,
  env
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;

const VERSION = '5.1';

const MODEL_TMR =
  'onnx-community/tmr-ai-text-detector-ONNX';

const MODEL_E5 =
  'onnx-community/e5-small-lora-ai-generated-detector-ONNX';

const MODEL_MODERN =
  'onnx-community/modernbert-ai-detection-raid-mage-ONNX';

const BENCH_KEY =
  'aiTraceBenchmarkV51';

const HISTORY_KEY =
  'aiTraceHistoryV51';

let tmr = null;
let e5 = null;
let modern = null;

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

    /Android|iPhone|iPad|iPod/i
      .test(
        navigator.userAgent
      )
  );
}


/* =========================================================
   UI
========================================================= */

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


function updateCount() {

  const words =
    wordCount(
      textEl.value
    );

  $('count').textContent =
    `${words} words`;
}


function loadDemo() {

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

  if ($('progress')) {

    $('progress')
      .classList
      .remove('hidden');
  }

  if ($('bar')) {

    $('bar').style.width =
      `${percent}%`;
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
    500
  );
}


function setState(label) {

  if ($('modelState')) {

    $('modelState').textContent =
      label;
  }
}


/* =========================================================
   HELPERS
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

  return (
    usable.reduce(
      (sum, value) =>
        sum + value,
      0
    ) /
    usable.length
  );
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
    average(
      usable
    );

  return Math.sqrt(
    average(
      usable.map(
        value =>
          (
            value -
            mean
          ) ** 2
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
      .sort(
        (a, b) =>
          a - b
      );

  if (!usable.length) {

    return 50;
  }

  const middle =
    Math.floor(
      usable.length /
      2
    );

  if (
    usable.length % 2
  ) {

    return usable[
      middle
    ];
  }

  return (
    usable[
      middle - 1
    ] +
    usable[
      middle
    ]
  ) / 2;
}


function wordCount(value) {

  if (
    !value.trim()
  ) {

    return 0;
  }

  return value
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .length;
}


function escapeHTML(value) {

  return value.replace(
    /[&<>"']/g,

    character =>
      ({
        '&':
          '&amp;',

        '<':
          '&lt;',

        '>':
          '&gt;',

        '"':
          '&quot;',

        "'":
          '&#039;'
      })[
        character
      ]
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
    latin /
    total >
    0.82
  )
    ? 'English'
    : 'Non-English';
}


function countMatches(
  value,
  regex
) {

  return (
    value.match(regex) ||
    []
  ).length;
}


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
        wordCount(
          sentence
        )
    );

  const paragraphLengths =
    paragraphs.map(
      paragraph =>
        wordCount(
          paragraph
        )
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
            value.match(
              regex
            ) || []
          ).length > 0
      )
      .length;

  const quoteCount =
    (
      value.match(
        /["“”‘’]/g
      ) || []
    ).length;

  const dialogueLines =
    lines.filter(
      line =>

        /^[“"'—-]/
          .test(
            line
          ) ||

        /[”"']$/
          .test(
            line
          )
    ).length;

  const firstPerson =
    (
      value.match(
        /\b(I|me|my|mine|we|us|our|ours)\b/gi
      ) || []
    ).length;

  const contractions =
    (
      value.match(
        /\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s)\b/gi
      ) || []
    ).length;

  const transitions =
    (
      value.match(
        /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|in conclusion|additionally|nevertheless|as a result)\b/gi
      ) || []
    ).length;

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

    dialogueLines,

    firstPerson,

    contractions,

    transitions
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
        /\b(method|methods|results|conclusion|study|participants|dataset|experiment|analysis|significant|hypothesis|abstract)\b/g
      ),

    recipes:
      countMatches(
        content,
        /\b(cup|tablespoon|teaspoon|ingredients|preheat|oven|bake|stir|chop|minutes|serve)\b/g
      ),

    reviews:
      countMatches(
        content,
        /\b(review|rating|stars|recommend|purchase|product|quality|price|experience)\b/g
      ),

    reddit:
      countMatches(
        content,
        /\b(aita|tldr|subreddit|upvote|downvote|edit:|throwaway|imo|lol)\b/g
      ),

    wiki:
      countMatches(
        content,
        /\b(was born|is a|refers to|located in|population|history of|known for|founded|species)\b/g
      ),

    news:
      countMatches(
        content,
        /\b(reuters|reported|according to|officials|government|minister|president|said on|announced|agency)\b/g
      ),

    poetry:
      (
        profile.lineBreaks >=
        6 &&
        profile.averageLineLength <
        60
      )
        ? 4
        : 0,

    books:
      (
        profile.quoteCount >=
        6 ||
        profile.dialogueLines >=
        2
      )
        ? 4
        : 0
  };

  const sorted =
    Object.entries(
      signals
    )
      .sort(
        (
          a,
          b
        ) =>
          b[1] -
          a[1]
      );

  const [
    domain,
    score
  ] = sorted[0];

  if (
    !score ||
    score < 2
  ) {

    if (
      profile.quoteCount >=
      4 ||
      profile.dialogueLines >
      0
    ) {

      return {

        domain:
          'books',

        confidence:
          'low'
      };
    }

    return {

      domain:
        'general',

      confidence:
        'low'
    };
  }

  return {

    domain,

    confidence:
      score >= 5
        ? 'medium'
        : 'low'
  };
}


/* =========================================================
   HUMAN COUNTER-EVIDENCE
========================================================= */

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

    score += 20;

    reasons.push(
      'high sentence-length variation'
    );

  } else if (
    profile.sentenceBurstiness >=
    0.45
  ) {

    score += 12;

    reasons.push(
      'moderate sentence-length variation'
    );
  }


  if (
    profile.punctuationTypes >=
    5
  ) {

    score += 12;

    reasons.push(
      'rich punctuation variety'
    );

  } else if (
    profile.punctuationTypes >=
    3
  ) {

    score += 6;
  }


  if (
    profile.quoteCount >=
    8 ||
    profile.dialogueLines >=
    2
  ) {

    score += 16;

    reasons.push(
      'dialogue / quotation structure'
    );

  } else if (
    profile.quoteCount >=
    3
  ) {

    score += 7;
  }


  if (
    profile.firstPerson >=
    4
  ) {

    score += 8;

    reasons.push(
      'personal voice'
    );

  } else if (
    profile.firstPerson >
    0
  ) {

    score += 4;
  }


  if (
    profile.contractions >=
    4
  ) {

    score += 6;
  }


  if (
    profile.paragraphDeviation >=
    25 &&
    profile.paragraphs >=
    3
  ) {

    score += 8;

    reasons.push(
      'irregular paragraph rhythm'
    );
  }


  if (
    profile.lexicalDiversity >=
    0.62
  ) {

    score += 6;
  }


  if (
    profile.transitions >=
    4
  ) {

    score -= 7;
  }


  if (
    domain ===
      'books' ||
    domain ===
      'poetry'
  ) {

    score += 8;

    reasons.push(
      'literary-domain caution'
    );
  }


  return {

    score:
      clamp(
        Math.round(
          score
        )
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


/* =========================================================
   MODEL LOADERS
========================================================= */

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


/* =========================================================
   MODEL OUTPUT
========================================================= */

function aiProbability(
  output
) {

  const results =
    (
      Array.isArray(
        output
      )
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
      label.includes(
        'ai'
      ) ||

      label.includes(
        'machine'
      ) ||

      label.includes(
        'generated'
      ) ||

      label ===
        'label_1'
    ) {

      ai =
        Math.max(
          ai ?? 0,
          score
        );
    }


    if (
      label.includes(
        'human'
      ) ||

      label ===
        'label_0'
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
    results.length >=
    2
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
    aiProbability(
      result
    ) *
    100
  );
}


/* =========================================================
   THIRD MODEL ROUTING
========================================================= */

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
    median(
      [
        quickScores.tmr,
        quickScores.e5
      ]
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
    standardDeviation(
      segmentScores
    );


  return (

    words <
      180 ||

    modelGap >=
      18 ||

    (
      raw >=
        35 &&
      raw <=
        85
    ) ||

    segmentRange >=
      45 ||

    segmentSD >=
      20 ||

    humanScore >=
      45 ||

    domain ===
      'books' ||

    domain ===
      'poetry'
  );
}


/* =========================================================
   CONSENSUS
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
      median(
        active
      )
    );


  const modelSpread =
    active.length >
    1

      ? Math.max(
          ...active
        ) -
        Math.min(
          ...active
        )

      : 100;


  const modelSD =
    Math.round(
      standardDeviation(
        active
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


  const segmentSD =
    Math.round(
      standardDeviation(
        segmentScores
      )
    );


  const domainRisk =
    (
      domain ===
        'books' ||
      domain ===
        'poetry'
    )

      ? 18

      : domain ===
          'general'
        ? 8
        : 4;


  let quality =
    100;


  quality -=
    Math.min(
      38,
      modelSpread *
      0.75
    );


  quality -=
    Math.min(
      30,

      segmentSD *
      0.65 +

      segmentRange *
      0.18
    );


  quality -=
    profile.words <
      150

      ? 15

      : profile.words <
          250
        ? 7
        : 0;


  quality -=
    language ===
      'English'

      ? 0

      : 35;


  quality -=
    domainRisk;


  quality -=
    thirdUsed

      ? 0

      : 8;


  quality =
    clamp(
      Math.round(
        quality
      )
    );


  const instability =
    1 -
    quality /
    100;


  const humanPenalty =
    human.score *
    (
      0.10 +
      instability *
      0.55
    ) *
    (
      raw /
      100
    );


  let calibrated =
    clamp(
      Math.round(
        raw -
        humanPenalty
      )
    );


  if (
    (
      domain ===
        'books' ||
      domain ===
        'poetry'
    ) &&

    human.score >=
      45 &&

    (
      segmentRange >=
        45 ||

      modelSpread >=
        20
    )
  ) {

    calibrated =
      Math.min(
        calibrated,
        64
      );
  }


  const uncertainty =
    clamp(
      Math.round(

        (
          100 -
          quality
        ) +

        (
          raw >=
            70 &&

          human.score >=
            50

            ? (
                human.score -
                45
              ) *
              0.30

            : 0
        )
      ),

      5,
      95
    );


  const confidence =
    100 -
    uncertainty;


  const highConflict =
    modelSpread >=
      32 ||

    segmentRange >=
      70 ||

    segmentSD >=
      28;


  let verdict =
    'INCONCLUSIVE';


  if (
    language !==
      'English'
  ) {

    verdict =
      'INCONCLUSIVE';

  } else if (

    calibrated >=
      86 &&

    quality >=
      74 &&

    !highConflict &&

    human.score <
      50

  ) {

    verdict =
      'Strong AI evidence';

  } else if (

    calibrated >=
      74 &&

    quality >=
      62 &&

    modelSpread <
      26 &&

    segmentRange <
      60 &&

    human.score <
      55

  ) {

    verdict =
      'Likely AI';

  } else if (

    calibrated <=
      20 &&

    quality >=
      55 &&

    human.score >=
      45

  ) {

    verdict =
      'Strong human evidence';

  } else if (

    calibrated <=
      38 &&

    human.score >=
      42 &&

    modelSpread <
      30

  ) {

    verdict =
      'Likely human';
  }


  if (
    verdict.includes(
      'AI'
    ) &&

    (
      highConflict ||

      (
        (
          domain ===
            'books' ||

          domain ===
            'poetry'
        ) &&

        human.score >=
          45
      )
    )
  ) {

    verdict =
      'INCONCLUSIVE';
  }


  return {

    raw,

    calibrated,

    quality,

    uncertainty,

    confidence,

    verdict,

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


/* =========================================================
   MAIN SCAN
========================================================= */

async function runSmartScan() {

  const value =
    textEl.value.trim();

  const words =
    wordCount(
      value
    );


  if (
    words <
    80
  ) {

    alert(
      'Paste at least 80 words. 150+ words is recommended.'
    );

    return;
  }


  $('scan').disabled =
    true;


  try {

    setProgress(
      3,
      'Profiling document…'
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
      humanEvidence(
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


    /*
      MODEL A
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
            30
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

    } catch (
      error
    ) {

      console.error(
        'TMR failed:',
        error
      );
    }


    /*
      MODEL B
    */

    try {

      const modelB =
        await loadE5();


      setProgress(
        62,
        'Running Model B…'
      );


      scores.e5 =
        await classify(
          modelB,
          value
        );

    } catch (
      error
    ) {

      console.error(
        'E5 failed:',
        error
      );
    }


    /*
      Mobile-safe third model rule
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
      MODEL C
      Desktop only
    */

    if (
      thirdUsed
    ) {

      try {

        const modelC =
          await loadModern();


        setProgress(
          78,
          'Running Model C…'
        );


        scores.modern =
          await classify(
            modelC,
            value
          );

      } catch (
        error
      ) {

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
        new Date()
          .toISOString(),

      words,

      language,

      domain:
        domainInfo.domain,

      domainConfidence:
        domainInfo.confidence,

      scores,

      segmentScores,

      human,

      consensus
    };


    /*
      Rendering is isolated so a UI issue
      cannot leave the scan frozen.
    */

    try {

      render(
        scan
      );

    } catch (
      renderError
    ) {

      console.error(
        'Render error:',
        renderError
      );


      alert(
        'The analysis completed, but an interface error occurred. Please refresh and try again.'
      );
    }


    try {

      saveHistory(
        scan
      );

    } catch (
      historyError
    ) {

      console.warn(
        'History error:',
        historyError
      );
    }


    setProgress(
      100,
      'Trace complete'
    );


    if (
      mobile
    ) {

      setState(
        'V5.1 Mobile Safe • 2-model engine ✓'
      );

    } else if (
      thirdUsed
    ) {

      setState(
        'V5.1 Deep 3-model engine ✓'
      );

    } else {

      setState(
        'V5.1 Smart 2-model engine ✓'
      );
    }


    setTimeout(
      () => {

        try {

          benchmarkPrompt(
            scan
          );

        } catch (
          benchmarkError
        ) {

          console.warn(
            'Benchmark prompt error:',
            benchmarkError
          );
        }

      },
      700
    );

  } catch (
    fatalError
  ) {

    console.error(
      'Fatal scan error:',
      fatalError
    );


    setState(
      'Scan error'
    );


    alert(
      'AI Trace encountered an error during analysis. Refresh the page and try again.'
    );

  } finally {

    $('scan').disabled =
      false;


    hideProgress();
  }
}


/* =========================================================
   RENDER
========================================================= */

function render(scan) {

  const {

    consensus,

    scores,

    human,

    language,

    domain,

    domainConfidence,

    segmentScores

  } = scan;


  const profile =
    createProfile(
      textEl.value
    );


  $('report')
    .classList
    .remove('hidden');


  $('score').textContent =
    `${consensus.calibrated}%`;


  $('scaleFill').style.width =
    `${consensus.calibrated}%`;


  $('verdict').textContent =
    consensus.verdict;


  const confidenceLabel =

    consensus.confidence >=
      75

      ? 'High'

      : consensus.confidence >=
          50

        ? 'Medium'

        : 'Low';


  $('confidence').textContent =
    `Evidence confidence: ${confidenceLabel} (${consensus.confidence}%)`;


  $('explain').textContent =
    `Raw ensemble signal ${consensus.raw}%; calibrated to ${consensus.calibrated}% after model agreement, segment stability, domain context and human counter-evidence. ${
      consensus.verdict ===
      'INCONCLUSIVE'
        ? 'AI Trace abstained because the available evidence was not reliable enough for a confident AI/Human verdict.'
        : ''
    }`;


  const humanDisplay =
    clamp(
      Math.round(

        human.score *
        0.72 +

        (
          100 -
          consensus.calibrated
        ) *
        0.28
      )
    );


  $('humanVal').textContent =
    `${humanDisplay}%`;


  $('aiVal').textContent =
    `${consensus.calibrated}%`;


  $('uncertainVal').textContent =
    `${consensus.uncertainty}%`;


  $('humanBar').style.width =
    `${humanDisplay}%`;


  $('aiBar').style.width =
    `${consensus.calibrated}%`;


  $('uncertainBar').style.width =
    `${consensus.uncertainty}%`;


  $('engineBadge').textContent =

    consensus.thirdUsed

      ? 'V5.1 • 3-MODEL CONSENSUS'

      : isMobileDevice()

        ? 'V5.1 • MOBILE SAFE'

        : 'V5.1 • SMART CONSENSUS';


  const humanReasons =

    human.reasons.length

      ? human.reasons
          .slice(
            0,
            4
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


  const evidence = [

    [
      'Calibrated AI signal',

      `${consensus.calibrated}%`,

      'Primary'
    ],

    [
      'Raw ensemble signal',

      `${consensus.raw}%`,

      'Diagnostic'
    ],

    [
      'Evidence quality',

      `${consensus.quality}%`,

      consensus.quality >=
        75
        ? 'Strong'
        : consensus.quality >=
            50
          ? 'Medium'
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

      modelCText,

      'Model C'
    ],

    [
      'Human counter-evidence',

      `${human.score}% — ${humanReasons}`,

      human.score >=
        55
        ? 'Strong'
        : human.score >=
            35
          ? 'Moderate'
          : 'Low'
    ],

    [
      'Model spread',

      `${consensus.modelSpread} points`,

      consensus.modelSpread >=
        32
        ? 'High conflict'
        : 'Acceptable'
    ],

    [
      'Segment range',

      `${consensus.segmentRange} points`,

      consensus.segmentRange >=
        70
        ? 'High variation'
        : 'Acceptable'
    ],

    [
      'Domain context',

      `${domain} (${domainConfidence} confidence)`,

      'Routing only'
    ],

    [
      'Language fit',

      language ===
        'English'

        ? 'English — strongest supported path'

        : 'Non-English — reduced reliability',

      'Context'
    ]
  ];


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

</div>

`
      )
      .join('');


  const metrics = {

    Words:
      profile.words,

    Sentences:
      profile.sentences,

    Domain:
      domain,

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
        .toFixed(
          2
        ),

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
      `${consensus.quality}%`
  };


  $('metrics').innerHTML =

    Object.entries(
      metrics
    )
      .map(
        (
          [
            key,
            value
          ]
        ) => `

<div class="metric">

  <span>${key}</span>

  <b>${value}</b>

</div>

`
      )
      .join('');


  const chunks =
    chunkText(
      textEl.value.trim()
    );


  $('segments').innerHTML =

    chunks
      .map(
        (
          chunk,
          index
        ) => {

          const score =
            segmentScores[
              index
            ] ??
            50;


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
      chunk.slice(
        0,
        320
      )
    )}

    ${
      chunk.length >
      320
        ? '…'
        : ''
    }

  </p>

</div>

`;

        }
      )
      .join('');


  $('report')
    .scrollIntoView({

      behavior:
        'smooth',

      block:
        'start'
    });


  renderBenchmarkPanel();
}


/* =========================================================
   BENCHMARK
========================================================= */

function loadBench() {

  try {

    return JSON.parse(
      localStorage.getItem(
        BENCH_KEY
      ) || '[]'
    );

  } catch {

    return [];
  }
}


function saveBench(
  records
) {

  localStorage.setItem(

    BENCH_KEY,

    JSON.stringify(
      records
    )
  );
}


function benchmarkPrompt(
  scan
) {

  const answer =
    prompt(
`AI TRACE V5.1 BENCHMARK

Only label samples whose origin you KNOW.

AI      = definitely AI-generated
HUMAN   = definitely human-written
MIXED   = known mixture
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


  records.push({

    id:
      `${truth[0]}-${String(
        records.length +
        1
      ).padStart(
        3,
        '0'
      )}`,

    truth,

    source,

    ...scan
  });


  saveBench(
    records
  );


  renderBenchmarkPanel();


  const results =
    benchmarkMetrics(
      records
    );


  alert(
`Saved.

Binary samples: ${results.total}
Coverage: ${results.coverage}%
Selective accuracy: ${results.selectiveAccuracy}%
False-positive rate: ${results.fpr}%
False-negative rate: ${results.fnr}%
Human abstention rate: ${results.humanAbstainRate}%

Development data only.`
  );
}


function prediction(
  record
) {

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


  return 'ABSTAIN';
}


function benchmarkMetrics(
  records =
    loadBench()
) {

  const rows =
    records.filter(
      record =>
        record.truth ===
          'AI' ||

        record.truth ===
          'HUMAN'
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
      prediction(
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
      record.truth ===
        'AI' &&
      predicted ===
        'AI'
    ) {

      TP++;
    }


    if (
      record.truth ===
        'HUMAN' &&
      predicted ===
        'HUMAN'
    ) {

      TN++;
    }


    if (
      record.truth ===
        'HUMAN' &&
      predicted ===
        'AI'
    ) {

      FP++;
    }


    if (
      record.truth ===
        'AI' &&
      predicted ===
        'HUMAN'
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


  return {

    total:
      rows.length,

    totalAI,

    totalHuman,

    TP,

    TN,

    FP,

    FN,

    aiAbstain,

    humanAbstain,


    coverage:

      rows.length

        ? Math.round(
            decided /
            rows.length *
            100
          )

        : 0,


    selectiveAccuracy:

      decided

        ? Math.round(
            (
              TP +
              TN
            ) /
            decided *
            100
          )

        : 0,


    precision:

      TP + FP

        ? Math.round(
            TP /
            (
              TP +
              FP
            ) *
            100
          )

        : 0,


    recall:

      totalAI

        ? Math.round(
            TP /
            totalAI *
            100
          )

        : 0,


    specificity:

      totalHuman

        ? Math.round(
            TN /
            totalHuman *
            100
          )

        : 0,


    fpr:

      totalHuman

        ? Math.round(
            FP /
            totalHuman *
            100
          )

        : 0,


    fnr:

      totalAI

        ? Math.round(
            FN /
            totalAI *
            100
          )

        : 0,


    humanAbstainRate:

      totalHuman

        ? Math.round(
            humanAbstain /
            totalHuman *
            100
          )

        : 0
  };
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
        10
      )
      .map(
        record => `

<div class="ev">

  <div class="evTop">

    <span>${record.id}</span>

    <span>${record.truth}</span>

  </div>

  <small>

    ${record.domain || 'general'}

    · calibrated
    ${record.consensus?.calibrated ?? '?'}%

    · quality
    ${record.consensus?.quality ?? '?'}%

    · ${record.consensus?.verdict || '?'}

  </small>

</div>

`
      )
      .join('');


  panel.innerHTML = `

<span class="over">
  V5.1 BENCHMARK • DEVELOPMENT ONLY
</span>

<h2>
  Reliability Dashboard
</h2>

<div class="metrics">

  <div class="metric">
    <span>Binary samples</span>
    <b>${results.total}</b>
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
        No V5.1 benchmark records yet.
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

  try {

    const history =
      JSON.parse(
        localStorage.getItem(
          HISTORY_KEY
        ) || '[]'
      );


    history.push({

      timestamp:
        scan.timestamp,

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

  } catch (
    error
  ) {

    console.warn(
      'History save failed:',
      error
    );
  }
}


/* =========================================================
   DEVELOPER UTILITIES
========================================================= */

window.AITraceV51 = {

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


  clearBenchmark() {

    if (
      confirm(
        'Delete all V5.1 benchmark records?'
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
      `AI-Trace-V51-Benchmark-${Date.now()}.json`;


    anchor.click();


    URL.revokeObjectURL(
      url
    );
  }
};


setTimeout(
  renderBenchmarkPanel,
  350
);
