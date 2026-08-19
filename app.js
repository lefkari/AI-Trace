import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;

const TMR_MODEL = 'onnx-community/tmr-ai-text-detector-ONNX';
const MODERN_MODEL = 'onnx-community/modernbert-ai-detection-raid-mage-ONNX';

let tmrClassifier = null;
let modernClassifier = null;

const $ = id => document.getElementById(id);
const text = $('text');

text.oninput = () => {
  const count = text.value.trim()
    ? text.value.trim().split(/\s+/).length
    : 0;

  $('count').textContent = count + ' words';
};

$('clear').onclick = () => {
  text.value = '';
  text.oninput();
  $('report').classList.add('hidden');
};

$('demo').onclick = () => {
  text.value = `Artificial intelligence is increasingly becoming an integral part of modern organizations. Moreover, its ability to process large volumes of information enables companies to identify patterns, improve decision-making, and automate repetitive processes. However, successful adoption requires more than simply deploying new technology. Organizations must also establish appropriate governance frameworks, train employees, monitor outcomes, and ensure that automated systems remain transparent and accountable. Ultimately, businesses that combine technological innovation with responsible implementation are more likely to create sustainable long-term value while minimizing operational and ethical risks.`;
  text.oninput();
};

$('scan').onclick = run;

function progress(percent, label) {
  $('progress').classList.remove('hidden');
  $('bar').style.width = percent + '%';
  $('progressText').textContent = label;
}

function detectLanguage(value) {
  const latin = (value.match(/[A-Za-z]/g) || []).length;
  const total = (value.match(/\p{L}/gu) || []).length;

  return total && latin / total > 0.8
    ? 'English'
    : 'Non-English';
}

function profile(value) {
  const words = value.trim().split(/\s+/);

  const sentences = value
    .split(/[.!?]+/)
    .map(x => x.trim())
    .filter(Boolean);

  const clean = words
    .map(w =>
      w
        .toLowerCase()
        .replace(/[^\p{L}\p{N}]/gu, '')
    )
    .filter(Boolean);

  const avg =
    words.length /
    Math.max(1, sentences.length);

  const lexical =
    new Set(clean).size /
    Math.max(clean.length, 1);

  const lengths = sentences.map(
    sentence => sentence.split(/\s+/).length
  );

  const mean =
    lengths.reduce((a, b) => a + b, 0) /
    Math.max(1, lengths.length);

  const variance =
    lengths.reduce(
      (sum, current) =>
        sum + (current - mean) ** 2,
      0
    ) /
    Math.max(1, lengths.length);

  const transitions =
    (
      value.match(
        /\b(however|moreover|furthermore|therefore|overall|ultimately|consequently|in conclusion)\b/gi
      ) || []
    ).length;

  return {
    words: words.length,
    sentences: sentences.length,
    avg,
    lexical,
    variance,
    transitions
  };
}

function chunkText(value, max = 1450) {
  const sentences =
    value.match(
      /[^.!?]+[.!?]+|[^.!?]+$/g
    ) || [value];

  const chunks = [];
  let current = '';

  for (const sentence of sentences) {
    if (
      (current + sentence).length >
        max &&
      current
    ) {
      chunks.push(current.trim());
      current = sentence;
    } else {
      current += sentence;
    }
  }

  if (current.trim()) {
    chunks.push(current.trim());
  }

  return chunks.slice(0, 8);
}

async function loadTMR() {
  if (tmrClassifier) {
    return tmrClassifier;
  }

  $('modelState').textContent =
    'Loading TMR engine…';

  progress(
    12,
    'Loading Quick Scan model…'
  );

  tmrClassifier = await pipeline(
    'text-classification',
    TMR_MODEL,
    {
      dtype: 'q8'
    }
  );

  return tmrClassifier;
}

async function loadModern() {
  if (modernClassifier) {
    return modernClassifier;
  }

  $('modelState').textContent =
    'Loading Deep Scan engine…';

  progress(
    55,
    'Loading second detector…'
  );

  modernClassifier = await pipeline(
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

  for (const item of results) {
    const label =
      String(item.label || '')
        .toLowerCase();

    const score =
      Number(item.score) || 0;

    if (
      label.includes('ai') ||
      label.includes('machine') ||
      label.includes('generated') ||
      label === 'label_1'
    ) {
      ai = Math.max(
        ai ?? 0,
        score
      );
    }

    if (
      label.includes('human') ||
      label === 'label_0'
    ) {
      human = Math.max(
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

  /*
   ModernBERT documentation defines
   probability index 1 as machine-generated.
   */
  if (
    results.length >= 2
  ) {
    return Number(
      results[1].score || 0.5
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

function standardDeviation(values) {
  if (!values.length) {
    return 0;
  }

  const mean =
    values.reduce(
      (a, b) => a + b,
      0
    ) / values.length;

  return Math.sqrt(
    values.reduce(
      (sum, current) =>
        sum +
        (current - mean) ** 2,
      0
    ) / values.length
  );
}

function heuristicScore(p) {
  let score = 42;

  if (p.variance < 35) {
    score += 10;
  }

  if (p.transitions > 2) {
    score += 8;
  }

  if (
    p.avg > 18 &&
    p.avg < 32
  ) {
    score += 8;
  }

  return Math.min(
    78,
    score
  );
}

function buildConsensus({
  tmr,
  modern,
  segmentScores,
  profileData,
  language
}) {
  const modelGap =
    Math.abs(
      tmr - modern
    );

  const segmentSpread =
    standardDeviation(
      segmentScores
    );

  const average =
    Math.round(
      tmr * 0.48 +
      modern * 0.48 +
      heuristicScore(
        profileData
      ) *
        0.04
    );

  let uncertainty = 12;

  uncertainty +=
    Math.min(
      35,
      modelGap * 0.65
    );

  uncertainty +=
    Math.min(
      25,
      segmentSpread * 0.65
    );

  if (
    profileData.words < 150
  ) {
    uncertainty += 12;
  }

  if (
    language !== 'English'
  ) {
    uncertainty += 25;
  }

  uncertainty =
    Math.min(
      95,
      Math.round(
        uncertainty
      )
    );

  const confidence =
    Math.max(
      5,
      100 - uncertainty
    );

  const strongDisagreement =
    modelGap >= 35;

  const unstableSegments =
    segmentSpread >= 28;

  let verdict;

  if (
    strongDisagreement ||
    unstableSegments
  ) {
    verdict =
      'INCONCLUSIVE';
  } else if (
    average >= 85
  ) {
    verdict =
      'Strong AI evidence';
  } else if (
    average >= 65
  ) {
    verdict =
      'Likely AI';
  } else if (
    average <= 15
  ) {
    verdict =
      'Strong human evidence';
  } else if (
    average <= 35
  ) {
    verdict =
      'Likely human';
  } else {
    verdict =
      'INCONCLUSIVE';
  }

  return {
    score: average,
    uncertainty,
    confidence,
    modelGap,
    segmentSpread:
      Math.round(
        segmentSpread
      ),
    verdict
  };
}

async function run() {
  const value =
    text.value.trim();

  const wordCount =
    value
      ? value.split(/\s+/)
          .length
      : 0;

  if (wordCount < 80) {
    alert(
      'Paste at least 80 words for a meaningful analysis.'
    );
    return;
  }

  $('scan').disabled = true;

  const documentProfile =
    profile(value);

  const language =
    detectLanguage(value);

  const chunks =
    chunkText(value);

  let tmrDocument = 50;
  let modernDocument = 50;

  const tmrSegments = [];

  let tmrWorked = true;
  let modernWorked = true;

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
            (i /
              chunks.length) *
              25
          ),
        `TMR segment ${
          i + 1
        }/${chunks.length}`
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
      'TMR error',
      error
    );

    tmrWorked = false;

    tmrDocument =
      heuristicScore(
        documentProfile
      );

    for (
      let i = 0;
      i < chunks.length;
      i++
    ) {
      tmrSegments.push(
        tmrDocument
      );
    }
  }

  /*
   Deep Scan:
   load the second detector only after
   the fast model has finished.
  */

  try {
    const modern =
      await loadModern();

    progress(
      72,
      'ModernBERT Deep Scan…'
    );

    modernDocument =
      await classify(
        modern,
        value
      );
  } catch (error) {
    console.error(
      'ModernBERT error',
      error
    );

    modernWorked = false;

    modernDocument =
      tmrDocument;
  }

  progress(
    88,
    'Building consensus…'
  );

  let consensus;

  if (
    tmrWorked &&
    modernWorked
  ) {
    consensus =
      buildConsensus({
        tmr:
          tmrDocument,
        modern:
          modernDocument,
        segmentScores:
          tmrSegments,
        profileData:
          documentProfile,
        language
      });
  } else {
    /*
     If only one model works,
     confidence is intentionally capped.
    */

    const spread =
      Math.round(
        standardDeviation(
          tmrSegments
        )
      );

    const fallbackScore =
      tmrWorked
        ? tmrDocument
        : heuristicScore(
            documentProfile
          );

    consensus = {
      score:
        fallbackScore,
      uncertainty:
        Math.max(
          50,
          spread
        ),
      confidence:
        Math.min(
          50,
          100 - spread
        ),
      modelGap: 0,
      segmentSpread:
        spread,
      verdict:
        'INCONCLUSIVE'
    };
  }

  renderV4({
    consensus,
    documentProfile,
    chunks,
    segmentScores:
      tmrSegments,
    language,
    tmrDocument,
    modernDocument,
    tmrWorked,
    modernWorked
  });

  progress(
    100,
    'Consensus complete'
  );

  $('modelState').textContent =
    tmrWorked &&
    modernWorked
      ? 'V4 consensus engine ready ✓'
      : 'Limited evidence mode';

  setTimeout(() => {
    $('progress')
      .classList.add(
        'hidden'
      );
  }, 500);

  $('scan').disabled = false;
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
      }[character])
  );
}

function renderV4({
  consensus,
  documentProfile,
  chunks,
  segmentScores,
  language,
  tmrDocument,
  modernDocument,
  tmrWorked,
  modernWorked
}) {
  $('report')
    .classList.remove(
      'hidden'
    );

  $('score').textContent =
    consensus.score + '%';

  $('scaleFill').style.width =
    consensus.score + '%';

  $('verdict').textContent =
    consensus.verdict;

  $('confidence').textContent =
    `Confidence: ${
      consensus.confidence >= 70
        ? 'High'
        : consensus.confidence >= 45
        ? 'Medium'
        : 'Low'
    } (${consensus.confidence}%)`;

  if (
    consensus.verdict ===
    'INCONCLUSIVE'
  ) {
    $('explain').textContent =
      `Conflicting or unstable evidence detected. Model disagreement: ${consensus.modelGap} points. Segment instability: ${consensus.segmentSpread}. AI Trace will not force an AI/Human verdict when the evidence is inconsistent.`;
  } else {
    $('explain').textContent =
      `Two independent detection signals were combined. TMR: ${tmrDocument}% AI. ModernBERT: ${modernDocument}% AI. Model disagreement: ${consensus.modelGap} points.`;
  }

  const human =
    100 -
    consensus.score;

  $('humanVal').textContent =
    human + '%';

  $('aiVal').textContent =
    consensus.score + '%';

  $('uncertainVal').textContent =
    consensus.uncertainty +
    '%';

  $('humanBar').style.width =
    human + '%';

  $('aiBar').style.width =
    consensus.score + '%';

  $('uncertainBar').style.width =
    consensus.uncertainty +
    '%';

  $('engineBadge').textContent =
    tmrWorked &&
    modernWorked
      ? 'V4 • 2-MODEL CONSENSUS'
      : 'LIMITED EVIDENCE';

  const evidence = [
    [
      'TMR detector',
      tmrWorked
        ? `${tmrDocument}% AI signal`
        : 'Unavailable',
      tmrWorked
        ? 'Model A'
        : 'Error'
    ],

    [
      'ModernBERT detector',
      modernWorked
        ? `${modernDocument}% AI signal`
        : 'Unavailable',
      modernWorked
        ? 'Model B'
        : 'Error'
    ],

    [
      'Model disagreement',
      `${consensus.modelGap} percentage points`,
      consensus.modelGap >= 35
        ? 'High conflict'
        : 'Acceptable'
    ],

    [
      'Segment stability',
      `Spread score: ${consensus.segmentSpread}`,
      consensus.segmentSpread >=
      28
        ? 'Unstable'
        : 'Stable'
    ],

    [
      'Language fit',
      language === 'English'
        ? 'English detected — strongest supported path.'
        : 'Non-English detected — reliability is reduced.',
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
      documentProfile.words,

    Sentences:
      documentProfile.sentences,

    'Avg. words / sentence':
      documentProfile.avg.toFixed(
        1
      ),

    'Lexical diversity':
      Math.round(
        documentProfile.lexical *
          100
      ) + '%',

    Language:
      language,

    'Models active':
      `${Number(
        tmrWorked
      ) +
        Number(
          modernWorked
        )}/2`,

    'Model disagreement':
      consensus.modelGap +
      ' pts',

    'Segment instability':
      consensus.segmentSpread
  };

  $('metrics').innerHTML =
    Object.entries(metrics)
      .map(
        ([key, value]) => `
        <div class="metric">
          <span>${key}</span>
          <b>${value}</b>
        </div>
      `
      )
      .join('');

  $('segments').innerHTML =
    chunks
      .map(
        (
          chunk,
          index
        ) => `
        <div class="segment">
          <div class="segmentHead">
            <b>Segment ${
              index + 1
            }</b>

            <span>
              ${
                segmentScores[
                  index
                ]
              }% TMR signal
            </span>
          </div>

          <div class="segmentMeter">
            <i
              style="width:${
                segmentScores[
                  index
                ]
              }%"
            ></i>
          </div>

          <p>
            ${escapeHTML(
              chunk.slice(
                0,
                300
              )
            )}

            ${
              chunk.length >
              300
                ? '…'
                : ''
            }
          </p>
        </div>
      `
      )
      .join('');

  /*
   Local benchmark log.
   No server / no paid database.
  */

  try {
    const history =
      JSON.parse(
        localStorage.getItem(
          'aiTraceBenchmarks'
        ) || '[]'
      );

    history.push({
      date:
        new Date().toISOString(),

      words:
        documentProfile.words,

      language,

      tmr:
        tmrDocument,

      modern:
        modernDocument,

      score:
        consensus.score,

      verdict:
        consensus.verdict,

      confidence:
        consensus.confidence,

      modelGap:
        consensus.modelGap,

      segmentSpread:
        consensus.segmentSpread
    });

    localStorage.setItem(
      'aiTraceBenchmarks',
      JSON.stringify(
        history.slice(-100)
      )
    );
  } catch (error) {
    console.warn(
      'Benchmark logging unavailable',
      error
    );
  }

  $('report').scrollIntoView({
    behavior: 'smooth',
    block: 'start'
  });
}
