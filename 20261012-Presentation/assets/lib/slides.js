const printMode = new URLSearchParams(location.search).has('print-pdf');
const markdownPlugin = RevealMarkdown();

Reveal.initialize({
  width: 1280, height: printMode ? 800 : 720, margin: 0.04,
  controls: false, progress: false, center: false,
  hash: true, hashOneBasedIndex: true,
  transition: 'none', pdfMaxPagesPerSlide: 1,
  pdfSeparateFragments: false,
  plugins: [markdownPlugin, RevealMath.MathJax3]
}).then(async () => {
  window.highlightCodeBlocks();
  await window.initializeCentroidClouds();
  const clouds = [...document.querySelectorAll('.centroid-cloud')];
  if (!printMode) {
    const distributionControl = document.querySelector('#distributions');
    const records = await (await fetch('assets/data/centroids.json')).json();
    for (const distribution of [...new Set(records.map(record => record.distribution))]) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = distribution;
      button.setAttribute('aria-pressed', String(distribution === 'Normal'));
      button.addEventListener('click', () => {
        clouds.forEach(cloud => cloud.selectDistribution(distribution));
        distributionControl.querySelectorAll('button').forEach(other => {
          other.setAttribute('aria-pressed', String(other === button));
        });
        document.querySelector('#print-distribution').textContent = distribution;
      });
      distributionControl.append(button);
    }
    document.querySelector('#reset').addEventListener('click', () => {
      clouds.forEach(cloud => cloud.resetView());
    });
  }
  window.highlightCodeBlocks();
  window.slidesReady = true;
});
